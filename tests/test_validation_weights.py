"""Generation FSDP thật: giảm all-gather, giữ output và tiếp tục training sau cleanup."""

import contextlib
import io
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class Tokenizer:
    pad_token_id = 0
    eos_token_id = None  # Tắt EOS để sinh đủ số token khi đếm communication.

    def apply_chat_template(self, messages, **kwargs):
        return [1, 3] + [4] * int(messages[1]["content"])

    def decode(self, tokens, **kwargs):
        return str(tokens)


def generation_worker(rank, directory, tied):
    import copy
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from torch.distributed.device_mesh import init_device_mesh
    from baseline.evaluate import generate_batch
    from baseline.model import ChunkedLoss, TARGETS, configure_attention
    from baseline.parallel import (shard_model, synchronize_adapter_gradients,
                                   unsharded_validation_generation)

    dist.init_process_group("gloo", init_method=f"file://{Path(directory) / 'rendezvous'}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        configure_attention()
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16,
            intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=64,
            tie_word_embeddings=tied, eos_token_id=None, pad_token_id=0)).half().requires_grad_(False)
        for name, parameter in base.named_parameters():
            if "norm" in name:
                parameter.data = parameter.data.float()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                              target_modules=TARGETS, task_type="CAUSAL_LM"))
        reference = ChunkedLoss(copy.deepcopy(model), 4)
        root = shard_model(model, init_device_mesh("cpu", (2,)), torch.device("cpu"))
        learner = ChunkedLoss(model, 4)
        params = [p for p in model.parameters() if p.requires_grad]
        expected = dict(reference.model.named_parameters())
        optimizer = torch.optim.SGD(params, lr=.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        ids = torch.tensor([[1, 3, 4, 5, 6, 7]])
        labels = ids.clone()
        labels[:, :2] = -100
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

        def train_step():
            learner.train()
            reference.train()
            model.config.use_cache = reference.model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.float16):
                actual = learner(**batch)
                gold = reference(**batch)
            torch.testing.assert_close(actual, gold, rtol=.003, atol=1e-4)
            actual.backward()
            gold.backward()
            synchronize_adapter_gradients(params, torch.device("cpu"))
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    key = name.replace("._checkpoint_wrapped_module", "")
                    torch.testing.assert_close(parameter.grad, expected[key].grad, rtol=.03, atol=5e-5)
            optimizer.step()
            reference_optimizer.step()

        train_step()
        learner.eval()
        reference.eval()
        model.config.use_cache = reference.model.config.use_cache = True
        cfg = {"gpu_count": 2, "max_new_tokens": 4, "timeout_seconds": 600}
        gather_name = "all_gather_single" if hasattr(dist, "all_gather_single") else "all_gather_into_tensor"
        original_gather = getattr(dist, gather_name)
        trace = []

        def gather(output, tensor, *args, **kwargs):
            trace.append((output.numel(), tensor.numel()))
            return original_gather(output, tensor, *args, **kwargs)

        def generate(target, questions):
            with patch("baseline.evaluate.synchronize"):
                return [(raw, count, finished) for raw, _, count, finished
                        in generate_batch(target, Tokenizer(), questions, cfg)]

        with patch.object(dist, gather_name, side_effect=gather):
            normal = generate(model, ["0", "3"]) + generate(model, ["1"])
            normal_gathers = len(trace)
            trace.clear()
            with unsharded_validation_generation(root, torch.device("cpu")):
                retained = generate(model, ["0", "3"])
                first_batch_gathers = len(trace)
                retained += generate(model, ["1"])
                assert first_batch_gathers > 0
                assert len(trace) == first_batch_gathers, "Batch sau không được all-gather lại trọng số"
            retained_gathers = len(trace)
            assert retained == normal == generate(reference.model, ["0", "3"]) + generate(reference.model, ["1"])
            assert retained_gathers < normal_gathers, (retained_gathers, normal_gathers)
            trace.clear()
            assert generate(model, ["0", "3"]) == normal[:2]
            assert len(trace) > retained_gathers, "Sau cleanup phải trở lại gather mỗi forward"
            # Lỗi giữa generation cũng phải khôi phục policy và giải phóng full weights.
            try:
                with unsharded_validation_generation(root, torch.device("cpu")):
                    generate(model, ["1"])
                    raise RuntimeError("generation interrupted")
            except RuntimeError as error:
                assert str(error) == "generation interrupted"
            trace.clear()
            generate(model, ["0", "3"])
            assert len(trace) > retained_gathers
        train_step()  # Gradient vẫn khớp reference sau validation thành công và thất bại.
        if tied:
            assert base.lm_head.weight is base.model.embed_tokens.weight
        if rank == 0:
            print(f"Validation weights PASS: tied={tied}, all-gather {normal_gathers} -> {retained_gathers}", flush=True)
    finally:
        dist.destroy_process_group()


class ValidationWeightsTests(unittest.TestCase):
    def test_generation_output_gathers_and_training_recovery(self):
        import torch.multiprocessing as mp
        for tied in (False, True):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as directory:
                mp.spawn(generation_worker, args=(directory, tied), nprocs=2, join=True)

    def test_context_only_wraps_generation_and_cleans_up_on_error(self):
        from baseline.validation import validate_epoch
        active = False
        entered = 0

        @contextlib.contextmanager
        def retain():
            nonlocal active, entered
            active = True
            entered += 1
            try:
                yield True
            finally:
                active = False

        class Learner:
            model = SimpleNamespace(config=SimpleNamespace(use_cache=False),
                                    get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)))
            def eval(self): pass
            def __call__(self, **kwargs):
                assert not active, "Dev loss phải giữ sharding"
                return torch.tensor(2.)

        def generation(model, tokenizer, questions, cfg):
            assert active
            return [(' {"answer":"1","unit":"V","explanation":"Giải"}', .1, 4, True)
                    for _ in questions]

        cfg = {"seed": 42, "validation_batch_size": 1, "rtol": .05, "atol": 0,
               "timeout_seconds": 600}
        rows = [{"id": str(i), "question": "0", "answer": "1", "unit": "V"} for i in range(2)]
        loader = [{"labels": torch.tensor([[-100, 1]])}]
        learner = Learner()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("baseline.validation.generate_batch", side_effect=generation):
                result = validate_epoch(learner, loader, rows, None, cfg, 1, directory,
                                        generation_context=retain())
            self.assertEqual(entered, 1)
            self.assertTrue(result["generation_keep_unsharded"])
            self.assertEqual(result["content_tolerant"], 2)
            self.assertFalse(active)
            with patch("baseline.validation.generate_batch", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    validate_epoch(learner, loader, rows, None, cfg, 1, directory,
                                   generation_context=retain())
            self.assertFalse(active)
            self.assertFalse(learner.model.config.use_cache)
            fixed = {**cfg, "checkpoint_selection": "fixed_epoch", "selected_epoch": 1}
            validate_epoch(learner, loader, rows, None, fixed, 1, directory,
                           generation_context=retain())
            self.assertEqual(entered, 2)  # fixed_epoch không mở context generation.


if __name__ == "__main__":
    unittest.main()
