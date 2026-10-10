"""Kiểm tra split, xếp hạng và luồng 5 epoch + train cuối bằng Qwen2 nhỏ trên CPU."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.data import apply_training_split, split_training_rows
from baseline.validation import checkpoint_rank


class SplitTests(unittest.TestCase):
    @staticmethod
    def rows(count=100):
        return [{"id": str(i), "question": f"Câu hỏi {i}", "cot": "Giải", "answer": "1", "unit": "V"}
                for i in range(count)]

    def test_fixed_split_grouped_duplicates(self):
        from baseline.metrics import canonical_text

        rows = self.rows()
        rows[1]["question"] = "  Câu hỏi  0\n"
        training, validation = split_training_rows(rows)
        self.assertEqual((len(training), len(validation)), (90, 10))
        self.assertEqual((training, validation), split_training_rows(rows))
        self.assertNotEqual(validation, split_training_rows(rows, seed=43)[1])
        self.assertEqual({r["id"] for r in training + validation}, {r["id"] for r in rows})
        self.assertFalse({canonical_text(r["question"]) for r in training}
                         & {canonical_text(r["question"]) for r in validation})

    def test_split_integrity_and_changed_data(self):
        rows = self.rows(10)
        training, validation = split_training_rows(rows)
        split = {"train_csv_sha256": "hash", "train_ids": [r["id"] for r in training],
                 "validation_ids": [r["id"] for r in validation]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(split))
            self.assertEqual(apply_training_split(rows, {"sha256": "hash"}, path)[:2], (training, validation))
            with self.assertRaisesRegex(ValueError, "CSV train đã đổi"):
                apply_training_split(rows, {"sha256": "changed"}, path)
            split["validation_ids"].append(training[0]["id"])
            path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, "rời nhau"):
                apply_training_split(rows, {"sha256": "hash"}, path)
        for rows in ([], self.rows(1)):
            with self.assertRaises(ValueError):
                split_training_rows(rows)

    def test_accuracy_ranking_and_ties(self):
        base = {"total": 10, "content_tolerant": 5, "content_exact": 4,
                "validation_loss": 1.0, "epoch": 2, "seed": 42}
        self.assertGreater(checkpoint_rank({**base, "content_tolerant": 6, "validation_loss": 9}), checkpoint_rank(base))
        for change in ({"content_exact": 5}, {"validation_loss": 0.9}, {"epoch": 1}):
            self.assertGreater(checkpoint_rank({**base, **change}), checkpoint_rank(base))
        with self.assertRaises(ValueError):
            checkpoint_rank({**base, "validation_loss": float("nan")})


class ExperimentTests(unittest.TestCase):
    def test_failed_search_does_not_start_final_training(self):
        from baseline.experiment import run_experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "train_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question", "cot", "answer", "unit"])
                writer.writeheader()
                writer.writerows(SplitTests.rows(10))
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

    def test_five_epochs_and_fresh_final_training(self):
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

            def generation(model, tokenizer, question, config):
                calls.append(question)
                epoch = (len(calls) - 1) // 3 + 1
                answer = "1" if epoch == 3 else "99"
                return json.dumps({"answer": answer, "unit": "V", "explanation": "Giải"}), 0.1, 10, True

            def runner(path):
                train(read_config(arguments("test").parse_args(["--config", str(path)])))

            with (patch("baseline.model.require_gpus"),
                  patch("baseline.model.backbone_plan", cpu_plan),
                  patch("baseline.train.load_model", cpu_model),
                  patch("baseline.train.load_tokenizer", return_value=Tokenizer()),
                  patch("baseline.validation.generate_once", generation),
                  contextlib.redirect_stdout(io.StringIO())):
                # Trường cấu hình cũ không được kích hoạt thêm 5 run theo seed.
                result = run_experiment({**cfg, "search_seeds": [42, 43, 44, 45, 46]}, runner=runner)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_epochs"], [1, 2, 3, 4, 5])
            self.assertEqual(initialized, [("search", 42), ("final", 42)])
            self.assertEqual(len(calls), 15)  # 3 câu validation, chấm sau mỗi epoch.
            self.assertNotIn("search_seeds", result["config"])
            self.assertEqual((result["best_checkpoint"]["seed"], result["best_checkpoint"]["epoch"]), (42, 3))
            final = json.loads((root / "experiment/final/run.json").read_text())
            self.assertEqual(final["training_samples"], 30)
            self.assertEqual(final["config"]["epochs"], 3)
            self.assertEqual(len(json.loads((root / "experiment/final/train_log.json").read_text())), 3)
            search = json.loads((root / "experiment/search/run.json").read_text())
            self.assertEqual(search["training_samples"], 27)
            self.assertEqual(search["config"]["epochs"], 5)
            self.assertIsNone(final["split"])
            self.assertEqual(set(final["data"]), {"train"})  # Không cần dev CSV cho cả search/final.
            self.assertTrue((root / "experiment/final/adapter/adapter_config.json").is_file())
            selection = json.loads((root / "experiment/selection.json").read_text())
            self.assertEqual(len(selection["candidates"]), 5)
            with self.assertRaises(FileExistsError):
                run_experiment(cfg, runner=runner)


if __name__ == "__main__":
    unittest.main()
