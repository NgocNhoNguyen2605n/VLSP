"""Sweep: đủ tổ hợp, cô lập config, xếp hạng, fail-fast và resume."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.data import load_training_data
from baseline.runtime import save_json
from baseline.sweep import combinations, main, run_sweep


class SweepTests(unittest.TestCase):
    grid = {"learning_rates": [1e-3, 1e-4, 2e-4], "weight_decays": [0.1], "max_new_tokens": [300, 1000]}

    def config(self, root):
        for name, count in (("train", 10), ("dev", 5)):
            with (root / f"{name}_dataset.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question", "cot", "answer", "unit"])
                writer.writeheader()
                writer.writerows({"id": str(i), "question": f"{name} {i}", "cot": "Giải",
                                  "answer": "1", "unit": "V"} for i in range(count))
        cfg = json.loads(Path("configs/baseline.json").read_text())
        cfg.update(data_root=str(root), output_dir=str(root / "sweep"), checkpoint_selection="answer_accuracy")
        return cfg

    def runner(self, calls, failure=None):
        def train(path):
            cfg = json.loads(Path(path).read_text())
            calls.append(cfg)
            if failure and failure(cfg):
                raise RuntimeError("NCCL failure")
            rows, dev, data, split = load_training_data(cfg)
            output = Path(cfg["output_dir"])
            history = []
            for epoch in range(1, cfg["epochs"] + 1):
                correct = (4 if cfg["learning_rate"] == 2e-4 and cfg["max_new_tokens"] == 1000 else 2) if epoch == 3 else 1
                checkpoint = output / f"epoch-{epoch}"
                save_json(checkpoint / "adapter_config.json", {})
                (checkpoint / "adapter_model.safetensors").write_bytes(f"weights-{epoch}".encode())
                history.append({"epoch": epoch, "validation": {
                    "epoch": epoch, "seed": cfg["seed"], "total": len(dev), "content_tolerant": correct,
                    "content_exact": correct, "validation_loss": 1.0, "accuracy": correct / len(dev),
                    "valid_json": len(dev), "unfinished_generation": 0}})
            save_json(output / "run.json", {"config": cfg, "data": data, "split": split,
                      "training_samples": len(rows), "training_status": "complete"})
            save_json(output / "train_log.json", history)
            save_json(output / "adapter/tokenizer_config.json", {})
            (output / "adapter/adapter_model.safetensors").write_bytes(b"last-weights")
        return train

    def test_grid_rejects_invalid_and_duplicate_values(self):
        self.assertEqual(len(combinations(self.grid)), 6)
        for key, invalid in (("learning_rates", [0]), ("learning_rates", [float("nan")]),
                             ("weight_decays", [-1]), ("weight_decays", [True]),
                             ("max_new_tokens", [300.0]), ("max_new_tokens", [0]),
                             ("max_new_tokens", []), ("max_new_tokens", [300, 300])):
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                combinations({**self.grid, key: invalid})

    def test_six_fresh_trials_and_ranked_results_use_exact_best_weights(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg = self.config(root)
            original = cfg.copy()
            calls = []
            result = run_sweep(cfg, self.grid, runner=self.runner(calls))
            self.assertEqual(cfg, original)
            self.assertEqual(len(calls), 6)
            self.assertEqual([(c['learning_rate'], c['max_new_tokens']) for c in calls],
                             [(lr, tokens) for lr in self.grid['learning_rates'] for tokens in (300, 1000)])
            self.assertEqual(len({c['output_dir'] for c in calls}), 6)
            self.assertTrue(all(c['seed'] == 42 and c['epochs'] == 5 and c['weight_decay'] == .1 for c in calls))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['best_trial']['parameters']['learning_rate'], 2e-4)
            self.assertEqual(result['best_trial']['parameters']['max_new_tokens'], 1000)
            self.assertEqual(result['best_trial']['best']['epoch'], 3)
            for trial in result['trials']:
                self.assertEqual((Path(trial['run_dir']) / 'adapter/adapter_model.safetensors').read_bytes(), b'weights-3')
                self.assertFalse((Path(trial['run_dir']) / 'final').exists())
            with (root / 'sweep/results.csv').open() as handle:
                results = list(csv.DictReader(handle))
            self.assertEqual(len(results), 6)
            self.assertEqual(float(results[0]['learning_rate']), 2e-4)
            self.assertEqual(int(results[0]['max_new_tokens']), 1000)
            self.assertEqual(int(results[0]['best_epoch']), 3)
            with self.assertRaises(FileExistsError):
                run_sweep(cfg, self.grid, runner=self.runner(calls))

    def test_failure_preserves_results_and_resume_skips_complete_trials(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg = self.config(root)
            calls = []
            with self.assertRaisesRegex(RuntimeError, 'NCCL failure'):
                run_sweep(cfg, self.grid, runner=self.runner(calls, lambda c: c['learning_rate'] == 1e-4))
            self.assertEqual(len(calls), 3)
            saved = json.loads((root / 'sweep/sweep.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual([t['status'] for t in saved['trials']], ['complete', 'complete', 'failed', 'pending', 'pending', 'pending'])
            resumed = []
            result = run_sweep(cfg, self.grid, runner=self.runner(resumed), resume=True)
            self.assertEqual(len(resumed), 4)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['trials'][2]['run_dir'].endswith('_attempt-02'))
            self.assertEqual(len(result['trials'][2]['attempts']), 2)
            self.assertTrue(Path(result['trials'][2]['attempts'][0]).exists())
            with self.assertRaisesRegex(ValueError, 'đã đổi'):
                run_sweep({**cfg, 'epochs': 3}, self.grid, runner=self.runner([]), resume=True)
            dataset = root / 'dev_dataset.csv'
            dataset.write_bytes(dataset.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'đã đổi'):
                run_sweep(cfg, self.grid, runner=self.runner([]), resume=True)

    def test_cli_uses_grid_and_output_override(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = Path(directory) / 'grid.json'
            save_json(grid, self.grid)
            with patch('sys.argv', ['baseline.sweep', '--grid-config', str(grid), '--output-dir', '/tmp/trials', '--resume']), \
                 patch('baseline.sweep.run_sweep') as run:
                main()
            cfg = run.call_args.args[0]
            self.assertEqual(cfg['output_dir'], '/tmp/trials')
            self.assertEqual(run.call_args.args[1], self.grid)
            self.assertTrue(run.call_args.kwargs['resume'])


if __name__ == '__main__':
    unittest.main()
