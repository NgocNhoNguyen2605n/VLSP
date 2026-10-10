"""Kiểm tra train/dev, xếp hạng và xuất best epoch bằng Qwen2 nhỏ trên CPU."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.validation import checkpoint_rank


class SplitTests(unittest.TestCase):
    @staticmethod
    def rows(count=100):
        return [{"id": str(i), "question": f"Câu hỏi {i}", "cot": "Giải", "answer": "1", "unit": "V"}
                for i in range(count)]

    def test_accuracy_ranking_and_ties(self):
        base = {"total": 10, "content_tolerant": 5, "content_exact": 4,
                "validation_loss": 1.0, "epoch": 2, "seed": 42}
        self.assertGreater(checkpoint_rank({**base, "content_tolerant": 6, "validation_loss": 9}), checkpoint_rank(base))
        for change in ({"content_exact": 5}, {"validation_loss": 0.9}, {"epoch": 1}):
            self.assertGreater(checkpoint_rank({**base, **change}), checkpoint_rank(base))
        with self.assertRaises(ValueError):
            checkpoint_rank({**base, "validation_loss": float("nan")})


class ExperimentTests(unittest.TestCase):
    def test_worker_uses_all_train_and_only_dev_for_validation_and_checks_both_csvs(self):
        from baseline.data import load_data, load_training_data, validation_manifest
        from baseline.runtime import save_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = SplitTests.rows(10)
            dev = [{**row, 'question': f"Dev {row['id']}"} for row in SplitTests.rows(3)]
            for name, rows in (('train', train), ('dev', dev)):
                with (root / f'{name}_dataset.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['id', 'question', 'cot', 'answer', 'unit'])
                    writer.writeheader()
                    writer.writerows(rows)
            cfg = {'data_root': str(root), 'skip_missing_answer': True, 'training_stage': 'search',
                   'split_file': str(root / 'split.json')}
            splits, info = load_data(cfg)
            manifest = validation_manifest(splits, info)
            save_json(cfg['split_file'], manifest)
            self.assertEqual(load_training_data(cfg)[:2], (train, dev))
            # ID scopes belong to separate CSVs; equal IDs do not imply an internal split.
            self.assertTrue(set(manifest['train_ids']) & set(manifest['validation_ids']))
            for name in ('train', 'dev'):
                dataset = root / f'{name}_dataset.csv'
                original = dataset.read_bytes()
                dataset.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'CSV hoặc danh sách ID đã đổi'):
                    load_training_data(cfg)
                dataset.write_bytes(original)
            save_json(cfg['split_file'], {**manifest, 'validation_ids': ['wrong']})
            with self.assertRaisesRegex(ValueError, 'CSV hoặc danh sách ID đã đổi'):
                load_training_data(cfg)

    def test_cli_runs_one_experiment_and_has_no_retrain_option(self):
        from baseline.experiment import main
        cfg = json.loads(Path('configs/baseline.json').read_text())
        with patch('sys.argv', ['baseline.experiment']), \
             patch('baseline.experiment.read_config', return_value=cfg), \
             patch('baseline.experiment.run_experiment') as run:
            main()
            run.assert_called_once_with(cfg)
        with patch('sys.argv', ['baseline.experiment', '--retrain-from', '/old/run']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()

    def test_legacy_train_cli_runs_experiment_and_internal_stages_do_not_recurse(self):
        from baseline.train import main
        cfg = json.loads(Path('configs/baseline.json').read_text())
        with patch('sys.argv', ['baseline.train']), patch('baseline.train.read_config', return_value=cfg), \
             patch('baseline.experiment.run_experiment') as experiment, \
             patch('baseline.parallel.launch_training') as launcher, contextlib.redirect_stdout(io.StringIO()):
            main()
            experiment.assert_called_once_with(cfg)
            launcher.assert_not_called()
        for stage in ('single', 'search', 'final'):
            worker_cfg = {**cfg, 'training_stage': stage}
            with patch('sys.argv', ['baseline.train']), patch('baseline.train.read_config', return_value=worker_cfg), \
                 patch('baseline.experiment.run_experiment') as experiment, \
                 patch('baseline.parallel.launch_training') as launcher:
                main()
                experiment.assert_not_called()
                launcher.assert_called_once_with(cfg=worker_cfg)
        worker_cfg = {**cfg, 'parallel_training': 'model', 'training_stage': 'search'}
        with patch('sys.argv', ['baseline.train']), patch('baseline.train.read_config', return_value=worker_cfg), \
             patch('baseline.experiment.run_experiment') as experiment, patch('baseline.train.train') as train:
            main()
            experiment.assert_not_called()
            train.assert_called_once_with(worker_cfg)

    def test_legacy_evaluate_parent_uses_final_adapter_and_rejects_incomplete_experiment(self):
        import hashlib
        from baseline.evaluate import evaluate
        from baseline.runtime import save_json
        cfg = json.loads(Path('configs/baseline.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / 'experiment.json', {'status': 'retraining'})
            with self.assertRaisesRegex(RuntimeError, 'Experiment chưa hoàn tất'):
                evaluate(root)
            save_json(root / 'experiment.json', {'status': 'complete'})
            save_json(root / 'final/adapter/adapter_config.json', {})
            save_json(root / 'final/run.json', {'training_status': 'complete', 'config': cfg, 'data': {},
                      'template_sha256': hashlib.sha256(b'toy-template').hexdigest()})
            class Tokenizer:
                chat_template = 'toy-template'
            rows = SplitTests.rows(1)
            with patch('baseline.evaluate.setup_seed'), patch('baseline.evaluate.environment', return_value={}), \
                 patch('baseline.evaluate.load_data', return_value=({'train': rows, 'dev': rows}, {})), \
                 patch('baseline.evaluate.load_tokenizer', return_value=Tokenizer()), \
                 patch('baseline.evaluate.load_model', return_value=(object(), {})) as loader, \
                 patch('baseline.evaluate.generate_once', return_value=(
                     '{"answer":"1","unit":"V","explanation":"Giải"}', .1, 5, True)), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(root)
            loader.assert_called_once_with(cfg, adapter=root / 'final/adapter')
            self.assertEqual(result['content_tolerant'], 1)
            self.assertTrue((root / 'final/evaluation/metrics.json').is_file())

    def test_evaluate_parent_uses_published_best_adapter_without_final_directory(self):
        import hashlib
        from baseline.evaluate import evaluate
        from baseline.runtime import save_json
        cfg = json.loads(Path('configs/baseline.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / 'experiment.json', {'status': 'complete', 'workflow': 'train_dev',
                                                'best_checkpoint': {'epoch': 3}})
            save_json(root / 'adapter/adapter_config.json', {})
            save_json(root / 'run.json', {'training_status': 'complete', 'config': cfg, 'data': {},
                      'split': {'validation_source': 'dev'},
                      'template_sha256': hashlib.sha256(b'toy-template').hexdigest()})
            class Tokenizer:
                chat_template = 'toy-template'
            rows = SplitTests.rows(1)
            with patch('baseline.evaluate.setup_seed'), patch('baseline.evaluate.environment', return_value={}), \
                 patch('baseline.evaluate.load_data', return_value=({'train': rows, 'dev': rows}, {})), \
                 patch('baseline.evaluate.load_tokenizer', return_value=Tokenizer()), \
                 patch('baseline.evaluate.load_model', return_value=(object(), {})) as loader, \
                 patch('baseline.evaluate.generate_once', return_value=(
                     '{"answer":"1","unit":"V","explanation":"Giải"}', .1, 5, True)), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(root)
            loader.assert_called_once_with(cfg, adapter=root / 'adapter')
            self.assertEqual(result['content_tolerant'], 1)
            self.assertIn('không phải test độc lập', result['notes'][-1])
            self.assertTrue((root / 'evaluation/metrics.json').is_file())
            self.assertFalse((root / 'final').exists())

    def test_failed_search_does_not_start_final_training(self):
        from baseline.experiment import run_experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "train_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question", "cot", "answer", "unit"])
                writer.writeheader()
                writer.writerows(SplitTests.rows(10))
            (root / "dev_dataset.csv").write_bytes((root / "train_dataset.csv").read_bytes())
            cfg = json.loads(Path("configs/baseline.json").read_text())
            cfg.update(data_root=str(root), output_dir=str(root / "experiment"))
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    "baseline.experiment.run_training", side_effect=RuntimeError("GPU failure")) as runner:
                with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                    run_experiment(cfg)
            self.assertEqual(runner.call_count, 1)
            result = json.loads((root / "experiment/experiment.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse((root / "experiment/configs/final.json").exists())

    def test_requires_valid_seed_and_epoch_count(self):
        from baseline.experiment import run_experiment

        cfg = json.loads(Path("configs/baseline.json").read_text())
        for seed in (True, -1, 2**32):
            with self.assertRaisesRegex(ValueError, "seed"):
                run_experiment({**cfg, "seed": seed})
        for epochs in (True, 0, -1, 2.5):
            with self.assertRaisesRegex(ValueError, "epochs"):
                run_experiment({**cfg, "epochs": epochs})

    def test_five_epochs_all_train_dev_validation_and_export_best_without_retrain(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from baseline.experiment import run_experiment
        from baseline.model import backbone_plan, load_model
        from baseline.runtime import arguments, read_config
        from baseline.train import train

        class Tokenizer:
            pad_token_id = 0
            chat_template = "toy-template"

            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                prefix = [1, 3, 4]
                return prefix if add_generation_prompt else prefix + [5, 6, 7, 8, 9, 2]

            def save_pretrained(self, directory):
                Path(directory, "tokenizer_config.json").write_text('{}')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                 max_position_embeddings=128, use_cache=False)
            Qwen2ForCausalLM(config).save_pretrained(root / "backbone")
            rows = SplitTests.rows(30)
            rows.append({"id": "missing", "question": "Thiếu", "cot": "Giải", "answer": "", "unit": ""})
            with (root / "train_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question", "cot", "answer", "unit"])
                writer.writeheader()
                writer.writerows(rows)
            dev_rows = [{**row, "question": f"Dev {row['id']}"} for row in SplitTests.rows(3)]
            with (root / "dev_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question", "cot", "answer", "unit"])
                writer.writeheader()
                writer.writerows(dev_rows)
            cfg = json.loads(Path("configs/baseline.json").read_text())
            cfg.update(model_id=str(root / "backbone"), revision="local-test", data_root=str(root),
                       output_dir=str(root / "experiment"), max_length=32, lora_rank=2, lora_alpha=4,
                       loss_chunk_size=3, batch_size=3, gradient_accumulation_steps=2)
            calls, initialized = [], []

            def cpu_plan(config):
                text, mapping, plan = backbone_plan(config)
                return text, {key: "cpu" for key in mapping}, plan

            def cpu_model(config, **kwargs):
                model, plan = load_model(config, **kwargs)
                self.assertNotIn("adapter", kwargs)
                self.assertTrue(all(torch.count_nonzero(p) == 0 for name, p in model.named_parameters()
                                    if "lora_B" in name))
                initialized.append((config["training_stage"], config["seed"]))
                return model, plan

            def generation(model, tokenizer, questions, config):
                epoch = len(calls) // 3 + 1
                calls.extend(questions)
                answer = "1" if epoch == 3 else "99"
                return [(json.dumps({"answer": answer, "unit": "V", "explanation": "Giải"}), 0.1, 10, True)
                        for question in questions]

            def runner(path):
                worker_cfg = read_config(arguments("test").parse_args(["--config", str(path)]))
                # Như worker CUDA: metadata được thêm sau khi đọc cấu hình search.
                worker_cfg['distributed_preflight'] = {'status': 'passed', 'device': 'cuda:0'}
                train(worker_cfg)

            with (patch("baseline.model.require_gpus"),
                  patch("baseline.model.backbone_plan", cpu_plan),
                  patch("baseline.train.load_model", cpu_model),
                  patch("baseline.train.load_tokenizer", return_value=Tokenizer()),
                  patch("baseline.validation.generate_batch", generation),
                  contextlib.redirect_stdout(io.StringIO())):
                # Trường cấu hình cũ không được kích hoạt thêm 5 run theo seed.
                result = run_experiment({**cfg, "search_seeds": [42, 43, 44, 45, 46]}, runner=runner)
                self.assertEqual(result["status"], "complete")
            self.assertEqual(initialized, [("search", 42)])  # One backbone load, no retrain.
            self.assertFalse((root / "experiment/final").exists())
            self.assertFalse((root / "experiment/configs/final.json").exists())
            self.assertEqual(result["completed_epochs"], [1, 2, 3, 4, 5])
            self.assertEqual(calls, [row["question"] for row in dev_rows] * 5)
            self.assertNotIn("search_seeds", result["config"])
            self.assertEqual((result["best_checkpoint"]["seed"], result["best_checkpoint"]["epoch"]), (42, 3))
            search = json.loads((root / "experiment/search/run.json").read_text())
            self.assertEqual(search["training_samples"], 30)  # All valid training rows.
            self.assertEqual(search["config"]["epochs"], 5)
            self.assertEqual(set(search["data"]), {"train", "dev"})
            self.assertEqual(len(search["split"]["train_ids"]), 30)
            self.assertEqual(len(search["split"]["validation_ids"]), 3)
            published = root / "experiment/adapter"
            self.assertTrue((published / "adapter_config.json").is_file())
            self.assertTrue((published / "tokenizer_config.json").is_file())
            from safetensors.torch import load_file
            best_weights = load_file(str(root / "experiment/search/epoch-3/adapter_model.safetensors"))
            exported = load_file(str(published / "adapter_model.safetensors"))
            last_weights = load_file(str(root / "experiment/search/epoch-5/adapter_model.safetensors"))
            self.assertEqual(set(exported), set(best_weights))
            for name in best_weights:
                torch.testing.assert_close(exported[name], best_weights[name], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(exported[name], last_weights[name]) for name in exported))
            published_manifest = json.loads((root / "experiment/run.json").read_text())
            self.assertEqual(published_manifest["adapter_epoch"], 3)
            self.assertEqual(len(json.loads((root / "experiment/selection.json").read_text())["candidates"]), 5)
            with self.assertRaises(FileExistsError):
                run_experiment(cfg, runner=runner)


if __name__ == "__main__":
    unittest.main()
