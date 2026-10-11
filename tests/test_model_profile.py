"""Training/serving phải dùng cùng model; không xuất adapter 3B với profile 7B."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from submission.config import BASE_MODEL, BASE_PARAMETERS, DEFAULT_MODEL_NAME, REVISION, ServingConfig
from submission.prepare import prepare


ROOT = Path(__file__).resolve().parents[1]


class ModelProfileTests(unittest.TestCase):
    def test_training_serving_and_eda_use_the_same_pinned_model(self):
        training = json.loads((ROOT / "configs/baseline.json").read_text())
        serving = ServingConfig.load(ROOT / "configs/submission.json")
        self.assertEqual(training["model_id"], BASE_MODEL)
        self.assertEqual(training["revision"], REVISION)
        self.assertEqual((serving.base_model, serving.revision), (BASE_MODEL, REVISION))
        self.assertEqual(serving.model_name, DEFAULT_MODEL_NAME)
        self.assertLessEqual(training["max_length"] + training["max_new_tokens"], serving.max_model_len)
        self.assertFalse(training["validation_keep_unsharded"])
        notebook = json.loads((ROOT / "01_data_exploration.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn(f'TOKENIZER_ID = "{BASE_MODEL}"', source)
        self.assertIn(f'TOKENIZER_REVISION = "{REVISION}"', source)

    def test_serving_rejects_an_old_backbone_or_revision(self):
        for overrides in ({"base_model": "Qwen/Qwen2.5-3B-Instruct"},
                          {"revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "backbone/revision"):
                ServingConfig(**overrides)

    def test_prepare_rejects_old_training_before_writing_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            for model, revision in (("Qwen/Qwen2.5-3B-Instruct", REVISION),
                                    (BASE_MODEL, "aa8e72537993ba99e69dfaafa59ed015b17504d1")):
                manifest = {"training_status": "complete", "adapter_epoch": 5,
                            "config": {"model_id": model, "revision": revision}}
                (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
                target = root / "deployment"
                with self.subTest(model=model, revision=revision), self.assertRaisesRegex(ValueError, "Checkpoint khác"):
                    prepare(target, run_dir=run)
                self.assertFalse(target.exists())

    def test_base_only_export_declares_the_new_backbone_and_capacity(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = prepare(Path(directory) / "deployment", base_only=True)
            serving = ServingConfig.load(output / "serving.json")
            serving.require_prepared()
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(serving.model_name, BASE_MODEL)
            self.assertEqual(provenance["base_parameters"], 7_615_616_512)
            self.assertEqual(provenance["total_parameters"], BASE_PARAMETERS)
            self.assertLess(provenance["total_parameters"], 8_000_000_000)
            self.assertEqual(provenance["active_llm_instances"], 1)

    @unittest.skipUnless(importlib.util.find_spec("httpx"), "Cần requirements-serving.txt")
    def test_vllm_uses_the_new_model_and_matching_lora_parent(self):
        from submission.launch import vllm_command

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text('{"r":16}')
            cfg = ServingConfig(adapter_path=str(adapter))
            command = vllm_command(cfg)
            self.assertEqual(command[:3], ["vllm", "serve", BASE_MODEL])
            for flag in ("--revision", "--tokenizer-revision"):
                self.assertEqual(command[command.index(flag) + 1], REVISION)
            lora = json.loads(command[command.index("--lora-modules") + 1])
            self.assertEqual(lora, {"name": DEFAULT_MODEL_NAME, "path": str(adapter),
                                    "base_model_name": BASE_MODEL})


if __name__ == "__main__":
    unittest.main()
