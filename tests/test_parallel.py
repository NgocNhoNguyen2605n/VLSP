"""Hai tiến trình FSDP thật trên CPU: gradient, padding, overflow, checkpoint và retrain."""

import csv
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "toy-template"

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        # Độ dài assistant khác nhau giữa rank để kiểm tra chunked head không deadlock.
        index = int(messages[1]["content"].split()[-1])
        prefix = [1, 3, 4] + [3] * (index % 5)
        length = 3 + index % 4
        return prefix if add_generation_prompt else prefix + [5] * length + [2]

    def decode(self, tokens, **kwargs):
        return str(tokens)

    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text('{}')


def fsdp_worker(rank, directory, mode):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from baseline.parallel import ParallelScaler, local_tensor, train_parallel, synchronize_adapter_gradients
    from baseline.runtime import read_config, arguments
    root = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        if mode in ('preflight', 'preflight_tied'):
            from baseline.parallel_check import check_parallel_backend
            torch.set_num_threads(1)
            with patch('torch.distributed.reduce_scatter_tensor', side_effect=AssertionError('reduce-scatter bị cấm')):
                result = check_parallel_backend(torch.device('cpu'), tied_embeddings=mode == 'preflight_tied')
            assert result['status'] == 'passed' and result['backend'] == 'gloo'
            assert result['tied_word_embeddings'] == (mode == 'preflight_tied')
            return
        if mode == 'replicated':
            weight = torch.nn.Parameter(torch.ones(4))
            conditional = torch.nn.Parameter(torch.ones(3))
            unused = torch.nn.Parameter(torch.ones(2))
            params = [weight, conditional, unused]
            weight.grad = torch.full_like(weight, rank + 1.)
            conditional.grad = torch.full_like(conditional, 2.) if rank else None
            synchronize_adapter_gradients(params, torch.device('cpu'), bucket_bytes=16)
            torch.testing.assert_close(weight.grad, torch.full_like(weight, 1.5))
            torch.testing.assert_close(conditional.grad, torch.ones_like(conditional))
            assert unused.grad is None  # Không decay tham số unused trên cả hai rank.
            optimizer = torch.optim.SGD(params, lr=.1, weight_decay=.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
            scaler = ParallelScaler(scale=1, replicated=True)
            skipped, _, _ = scaler.step(params, optimizer, scheduler, 1., torch.device('cpu'))
            assert not skipped
            torch.testing.assert_close(weight, torch.ones_like(weight) - .1 * (1.5 / (12**.5 + 1e-6) + .1))
            torch.testing.assert_close(conditional, torch.ones_like(conditional) - .1 * (1 / (12**.5 + 1e-6) + .1))
            torch.testing.assert_close(unused, torch.ones_like(unused))
            original = weight.detach().clone()
            weight.grad = torch.full_like(weight, float('inf') if rank else 1.)
            synchronize_adapter_gradients(params, torch.device('cpu'), bucket_bytes=16)
            skipped, before, after = scaler.step(params, optimizer, scheduler, 1., torch.device('cpu'))
            assert skipped and (before, after) == (1, .5) and scheduler.last_epoch == 1
            torch.testing.assert_close(weight, original)
            # Nhóm cuối 5 mẫu thật: zero-weight dummy vẫn giữ gradient đúng.
            layer = torch.nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                layer.weight.fill_(1)
            reference = torch.nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                reference.weight.fill_(1)
            inputs = torch.arange(20, dtype=torch.float32).view(5, 4) / 10
            reference(inputs).square().mean().backward()
            for step in range(3):
                index = 2 * step + rank
                valid = index < 5
                loss = layer(inputs[index if valid else 0:index + 1 if valid else 1]).square().mean()
                (loss * (2 * int(valid) / 5)).backward()
            synchronize_adapter_gradients(list(layer.parameters()), torch.device('cpu'))
            torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
            return
        if mode in ("gradient", "gradient_tied"):
            import copy
            from peft import LoraConfig, get_peft_model
            from transformers import Qwen2Config, Qwen2ForCausalLM
            from baseline.model import ChunkedLoss, TARGETS, configure_attention
            from baseline.parallel import shard_model, synchronize_adapter_gradients
            from baseline.data import collate_batch
            torch.manual_seed(42)
            torch.set_num_threads(1)
            configure_attention()
            base = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16,
                intermediate_size=24, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, use_cache=False,
                tie_word_embeddings=mode == 'gradient_tied')).half().requires_grad_(False)
            for name, parameter in base.named_parameters():
                if 'norm' in name:
                    parameter.data = parameter.data.float()
            model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=TARGETS,
                                                  task_type='CAUSAL_LM'))
            reference = ChunkedLoss(copy.deepcopy(model), 2)
            # Ép .to() thay Parameter identity để kiểm tra ignored_params sau chuyển device.
            old_overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            try:
                shard_model(model, init_device_mesh('cpu', (2,)), torch.device('cpu'))
            finally:
                torch.__future__.set_overwrite_module_params_on_conversion(old_overwrite)
            if mode == 'gradient_tied':
                assert base.lm_head.weight is base.model.embed_tokens.weight
                assert not isinstance(base.lm_head.weight, torch.distributed.tensor.DTensor)
                assert not base.lm_head.weight.requires_grad
                shared_weight = base.lm_head.weight.detach().clone()
            learner = ChunkedLoss(model, 2)
            def batch(step, owner):
                features = []
                for i in range(4):
                    count = 6 + (step * 3 + owner * 7 + i) % 11
                    features.append({'input_ids': [1, 3, 4] + [5] * count,
                                     'attention_mask': [1] * (3 + count),
                                     'labels': [-100] * 3 + [5] * count})
                return collate_batch(features, 0)
            params = [p for p in model.parameters() if p.requires_grad]
            expected = dict(reference.model.named_parameters())
            optimizer = torch.optim.SGD(params, lr=.01)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
            from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
            original_prefetch = FSDPParamGroup._prefetch_unshard
            gather_name = 'all_gather_single' if hasattr(dist, 'all_gather_single') else 'all_gather_into_tensor'
            original_gather = getattr(dist, gather_name)
            in_prefetch, trace = False, []
            def checked_prefetch(*args, **kwargs):
                nonlocal in_prefetch
                in_prefetch = True
                try:
                    return original_prefetch(*args, **kwargs)
                finally:
                    in_prefetch = False
            def checked_gather(output, tensor, *args, **kwargs):
                assert not in_prefetch, 'Prefetch không được tạo all-gather'
                trace.append((output.numel(), tensor.numel(), str(tensor.dtype)))
                return original_gather(output, tensor, *args, **kwargs)
            with patch.object(FSDPParamGroup, '_prefetch_unshard', side_effect=checked_prefetch), \
                 patch.object(dist, gather_name, side_effect=checked_gather):
                for update in range(20):  # 80 micro-batches: vượt vị trí batch 69 trong log lỗi.
                    # No-grad validation giữa optimizer steps rồi trở lại train.
                    learner.eval()
                    if update:
                        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
                            learner(**batch(update, 0))
                    learner.train()
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    for step in range(4):
                        with torch.autocast('cpu', dtype=torch.float16):
                            loss = learner(**batch(step + update, rank))
                        # Lệch nhịp hai rank + số head chunks khác nhau.
                        import time
                        if (step + update) % 2 == rank:
                            time.sleep(.01)
                        before = len(trace)
                        (loss * 128 / 4).backward()
                        assert len(trace) - before == 4, (rank, update, step, trace[before:])
                        for owner in range(2):
                            with torch.autocast('cpu', dtype=torch.float16):
                                loss = reference(**batch(step + update, owner))
                            (loss * 128 / 8).backward()
                    synchronize_adapter_gradients(params, torch.device('cpu'), bucket_bytes=256)
                    for parameter in [*params, *reference.parameters()]:
                        if parameter.grad is not None:
                            parameter.grad.div_(128)
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            key = name.replace('._checkpoint_wrapped_module', '')
                            torch.testing.assert_close(parameter.grad, expected[key].grad, rtol=.03, atol=5e-5)
                            if 'lora_B' in name:
                                assert parameter.grad.abs().sum() > 0
                    optimizer.step()
                    reference_optimizer.step()
                    if mode == 'gradient_tied':
                        assert base.lm_head.weight is base.model.embed_tokens.weight
                        torch.testing.assert_close(base.lm_head.weight, shared_weight)
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            key = name.replace('._checkpoint_wrapped_module', '')
                            torch.testing.assert_close(parameter, expected[key], rtol=.03, atol=5e-5)
            traces = [None, None]
            dist.all_gather_object(traces, trace)
            assert traces[0] == traces[1]  # cùng số/thứ tự/shape collective trên hai rank
            return
        if mode == "scaler":
            mesh = init_device_mesh("cpu", (2,))
            layer = torch.nn.Linear(4, 2, bias=False)
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
            # Mỗi rank giữ một hàng, khởi tạo cùng trọng số để so bước update.
            with torch.no_grad():
                local_tensor(layer.weight).fill_(1)
            optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
            scaler = ParallelScaler()
            params = list(layer.parameters())
            (layer(torch.ones(1, 4)).sum() * scaler.scale).backward()
            original = local_tensor(layer.weight).detach().clone()
            if rank == 1:
                local_tensor(layer.weight.grad).fill_(float("inf"))
            skipped, before, after = scaler.step(params, optimizer, scheduler, 1.0, torch.device("cpu"))
            assert skipped and (before, after) == (128, 64)
            torch.testing.assert_close(local_tensor(layer.weight), original)
            assert scheduler.last_epoch == 0
            (layer(torch.ones(1, 4)).sum() * scaler.scale).backward()
            skipped, _, _ = scaler.step(params, optimizer, scheduler, 1.0, torch.device("cpu"))
            assert not skipped and scheduler.last_epoch == 1
            # Global norm sqrt(8), không clip theo local norm sqrt(4).
            torch.testing.assert_close(local_tensor(layer.weight), original - 0.1 / (8**0.5 + 1e-6))
            # Nhóm 5 mẫu trên 2 rank: dummy không đóng góp, gradient khớp mean của 5 mẫu thật.
            with torch.no_grad():
                local_tensor(layer.weight).fill_(1)
            optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
            reference = torch.nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                reference.weight.fill_(1)
            inputs = torch.arange(20, dtype=torch.float32).view(5, 4) / 10
            reference(inputs).square().mean().backward()
            for step in range(3):
                index = 2 * step + rank
                valid = index < 5
                layer.set_requires_gradient_sync(step == 2)
                loss = layer(inputs[index if valid else 0:index + 1 if valid else 1]).square().mean()
                (loss * (2 * int(valid) / 5) * scaler.scale).backward()
            skipped, _, _ = scaler.step(params, optimizer, scheduler, 1e6, torch.device("cpu"))
            assert not skipped
            torch.testing.assert_close(layer.weight.full_tensor(), reference.weight.detach() - 0.01 * reference.weight.grad)
            return
        cfg = read_config(arguments("test").parse_args(["--config", str(root / 'config.json')]))
        initialized = []
        from baseline.model import load_model
        from baseline.evaluate import generate_batch
        def fresh(config, **kwargs):
            model, plan = load_model(config, **kwargs)
            assert kwargs == {"training": True, "cpu_init": True}
            assert all(torch.count_nonzero(p) == 0 for name, p in model.named_parameters() if 'lora_B' in name)
            initialized.append(config['training_stage'])
            return model, plan
        generated, token_counts, batch_sizes = [], [], []
        validation_count = len(json.loads(Path(cfg['split_file']).read_text())['validation_ids'])
        def generation(model, tokenizer, questions, config):
            # Chạy greedy thật qua FSDP/KV cache/synced_gpus; metric giả để chọn epoch 3 chắc chắn.
            with patch("baseline.evaluate.synchronize"):
                generation_cfg = {**config, 'timeout_seconds': 0 if rank == 0 else 600} if not generated else config
                outputs = generate_batch(model, tokenizer, questions, generation_cfg)
            epoch = len(generated) // validation_count + 1
            batch_sizes.append(len(questions))
            answer = '1' if epoch == 3 else '99'
            results = []
            for raw, elapsed, count, finished in outputs:
                generated.append(raw)
                token_counts.append(count)
                results.append((json.dumps({'answer': answer, 'unit': 'V', 'explanation': 'Giải'}), 0.1, count, finished))
            return results
        with patch("baseline.parallel.load_tokenizer", return_value=ToyTokenizer()), \
             patch("baseline.parallel.load_model", fresh), \
             patch("baseline.validation.generate_batch", generation), \
             patch("torch.distributed.reduce_scatter_tensor", side_effect=AssertionError('FSDP không được reduce-scatter LoRA')):
            train_parallel(cfg, torch.device('cpu'))
            best = json.loads((Path(cfg['output_dir']) / 'best_checkpoint.json').read_text())
            final = {**cfg, 'training_stage': 'final', 'output_dir': str(root / 'final'), 'epochs': best['epoch']}
            final.pop('split_file')
            train_parallel(final, torch.device('cpu'))
        assert initialized == ['search', 'final']
        assert len(generated) == 5 * validation_count
        expected_batches = [min(4, validation_count - start) for start in range(0, validation_count, 4)]
        assert batch_sizes == expected_batches * 5
        all_generated = [None, None]
        dist.all_gather_object(all_generated, generated)
        assert all_generated[0][batch_sizes[0]:] == all_generated[1][batch_sizes[0]:]
        counts = [None, None]
        dist.all_gather_object(counts, token_counts)
        assert counts[0][0] == 1 and counts[1][0] == 2  # Deadline khác nhau vẫn không deadlock.
        # Full adapter tensors sau FSDP phải bằng adapter khi nạp bằng PEFT thông thường.
        from safetensors.torch import load_file
        from peft import PeftModel
        from transformers import Qwen2ForCausalLM
        saved = load_file(str(root / 'final/adapter/adapter_model.safetensors'))
        loaded = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(root / 'backbone', dtype=torch.float16),
                                          root / 'final/adapter')
        for name, p in loaded.named_parameters():
            if 'lora_' in name:
                key = name.replace('.default', '')
                torch.testing.assert_close(p.float(), saved[key])
        assert any(tensor.abs().sum() > 0 for name, tensor in saved.items() if 'lora_B' in name)
    finally:
        dist.destroy_process_group()


class ParallelTests(unittest.TestCase):
    def test_launcher_and_experiment_dispatch_two_workers(self):
        from baseline.parallel import launch_training
        from baseline.experiment import run_training
        cfg = json.loads(Path('configs/baseline.json').read_text())
        def inspect_launch(command, **kwargs):
            self.assertIn('--nproc_per_node=2', command)
            self.assertEqual(command[command.index('--module') + 1], 'baseline.parallel')
            worker_cfg = json.loads(Path(command[command.index('--config') + 1]).read_text())
            self.assertEqual(worker_cfg, cfg)
            self.assertEqual(kwargs['env']['PYTHONHASHSEED'], '42')
            self.assertGreater(int(kwargs['env']['TORCH_NCCL_TRACE_BUFFER_SIZE']), 0)
            self.assertEqual(kwargs['env']['TORCH_NCCL_DUMP_ON_TIMEOUT'], '1')
            self.assertTrue(kwargs['check'])
        with patch('baseline.parallel.torch.cuda.device_count', return_value=2), \
             patch('baseline.parallel.subprocess.run', side_effect=inspect_launch) as runner:
            launch_training(cfg=cfg)
            self.assertEqual(runner.call_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.json'
            path.write_text(json.dumps(cfg))
            with patch('baseline.parallel.launch_training') as launcher:
                run_training(path)
                launcher.assert_called_once_with(path)
        with patch('baseline.parallel.torch.cuda.device_count', return_value=1):
            with self.assertRaisesRegex(RuntimeError, '2 GPU'):
                launch_training(cfg=cfg)

    def test_disjoint_batches_cover_dataset_without_padding_duplicates(self):
        from baseline.parallel import parallel_batches
        for count in (1, 3, 10, 13):
            features = list(range(count))
            combined, sequences = [], []
            for rank in range(2):
                batches = list(parallel_batches(features, 1, 2, rank, 42, 1))
                sequences.append(batches)
                for batch, valid in batches:
                    if valid:
                        combined.extend(batch)
            self.assertEqual(sorted(combined), features)
            self.assertEqual(len(sequences[0]), len(sequences[1]))
            self.assertEqual(sequences[0], list(parallel_batches(features, 1, 2, 0, 42, 1)))
        self.assertNotEqual(list(parallel_batches(list(range(13)), 1, 2, 0, 42, 1)),
                            list(parallel_batches(list(range(13)), 1, 2, 0, 42, 2)))

    def test_two_rank_overflow_sync_and_global_clipping(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(directory, 'scaler'), nprocs=2, join=True)

    def test_checkpointed_fsdp_gradients_match_unsharded_reference(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(directory, 'gradient'), nprocs=2, join=True)

    def test_tied_embedding_head_fsdp_gradients_and_frozen_shared_weight(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(directory, 'gradient_tied'), nprocs=2, join=True)

    def test_replicated_gradients_missing_parameters_clipping_overflow_and_partial_batch(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(directory, 'replicated'), nprocs=2, join=True)

    def test_backend_preflight_against_reference_without_reduce_scatter(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(directory, 'preflight'), nprocs=2, join=True)

    def test_tied_backend_preflight_against_reference(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(directory, 'preflight_tied'), nprocs=2, join=True)

    def test_length_grouped_batches_reduce_padding_and_keep_partial_batch_last(self):
        from baseline.parallel import parallel_batches
        features = [{'input_ids': list(range(2 + (i * 71) % 509)), 'id': i} for i in range(133)]
        def sample(grouped, epoch):
            seen, padded, count = [], 0, []
            for rank in range(2):
                batches = list(parallel_batches(features, 4, 2, rank, 42, epoch, grouped))
                count.append(len(batches))
                for step, (batch, valid) in enumerate(batches):
                    if step < len(batches) - 1:
                        self.assertEqual(valid, 4)
                    seen.extend(row['id'] for row in batch[:valid])
                    padded += max(len(row['input_ids']) for row in batch) * valid
            self.assertEqual(sorted(seen), list(range(133)))
            self.assertEqual(count[0], count[1])
            return seen, padded
        grouped, padded = sample(True, 1)
        self.assertEqual((grouped, padded), sample(True, 1))
        self.assertNotEqual(grouped, sample(True, 2)[0])
        self.assertLess(padded, sample(False, 1)[1] * .8)

    def test_two_rank_five_epochs_and_full_data_retrain(self):
        self.run_search_and_retrain(batch_size=1)

    def test_two_rank_batch_four_variable_lengths_and_partial_batches(self):
        self.run_search_and_retrain(batch_size=4)

    def test_tied_model_batch_four_search_validation_retrain_and_adapter_reload(self):
        self.run_search_and_retrain(batch_size=4, tied=True)

    def run_search_and_retrain(self, batch_size, tied=False):
        import torch.multiprocessing as mp
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from baseline.data import load_data, split_training_rows
        from baseline.runtime import save_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(42)
            Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=128, use_cache=False, tie_word_embeddings=tied)).save_pretrained(root / 'backbone')
            sample_count = 43 if batch_size == 4 else 13
            with (root / 'train_dataset.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['id', 'question', 'cot', 'answer', 'unit'])
                writer.writeheader()
                writer.writerows([{'id': str(i), 'question': f'Câu hỏi {i}', 'cot': 'Giải', 'answer': '1', 'unit': 'V'}
                                  for i in range(sample_count)])
            cfg = json.loads(Path('configs/baseline.json').read_text())
            cfg.update(model_id=str(root / 'backbone'), revision='local-test', data_root=str(root),
                       output_dir=str(root / 'search'), max_length=32, lora_rank=2, lora_alpha=4,
                       loss_chunk_size=2, batch_size=batch_size, gradient_accumulation_steps=4,
                       max_new_tokens=2, timeout_seconds=600, training_stage='search')
            splits, info = load_data(cfg, splits=('train',))
            fit, val = split_training_rows(splits['train'])
            split_path = root / 'split.json'
            save_json(split_path, {'train_csv_sha256': info['train']['sha256'],
                'train_ids': [r['id'] for r in fit], 'validation_ids': [r['id'] for r in val]})
            cfg['split_file'] = str(split_path)
            save_json(root / 'config.json', cfg)
            mp.spawn(fsdp_worker, args=(directory, 'training'), nprocs=2, join=True)
            search = json.loads((root / 'search/run.json').read_text())
            final = json.loads((root / 'final/run.json').read_text())
            self.assertEqual((search['training_samples'], final['training_samples']), (len(fit), sample_count))
            self.assertEqual((search['effective_batch_size'], final['effective_batch_size']), (8 * batch_size,) * 2)
            self.assertEqual((search['config']['epochs'], final['config']['epochs']), (5, 3))
            self.assertEqual(search['model']['parallelism'], 'FSDP2')
            self.assertEqual(search['model']['world_size'], 2)
            self.assertEqual(final['training_status'], 'complete')
            import math
            self.assertEqual(final['successful_optimizer_steps'], 3 * math.ceil(sample_count / (8 * batch_size)))
            self.assertEqual(search['model']['activation_checkpointing'], 'inside FSDP; non-reentrant')
            self.assertTrue(search['group_by_length'])
            self.assertTrue(search['model']['replicated_lora'])
            self.assertEqual(search['model']['replicated_frozen_embedding'], tied)
            for worker in range(2):
                progress = json.loads((root / f'final/rank-{worker}-progress.json').read_text())
                self.assertEqual(progress['phase'], 'complete')
                self.assertEqual(progress['epoch'], 3)
            self.assertIsNone(final['split'])
            self.assertEqual(set(final['data']), {'train'})
            self.assertEqual(len(json.loads((root / 'search/train_log.json').read_text())), 5)
            self.assertEqual(len(list((root / 'search/validation').glob('epoch-*/metrics.json'))), 5)


if __name__ == '__main__':
    unittest.main()
