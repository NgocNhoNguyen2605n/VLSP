"""Kiểm tra các điểm dễ làm lệch kết quả; không cần tải trọng số 9B."""

import csv  # Tạo CSV mẫu có BOM và xuống dòng.
import json  # Tạo phản hồi JSON mẫu.
import tempfile  # Không sửa dữ liệu thật khi test.
import unittest  # Chạy tests bằng thư viện chuẩn Python.
from pathlib import Path  # File tạm cho loader và checkpoint.

from baseline.data import find_csv, load_csv, prompt_messages  # Loader và ranh giới train/dev.
from baseline.metrics import numeric_value, parse_prediction, score_prediction, unit_ascii  # Chấm P1.


class MetricsTests(unittest.TestCase):
    """Kiểm tra số rất nhỏ, đơn vị, nhiều phần và output không hợp lệ."""

    @staticmethod
    def prediction(answer, unit):
        """Tạo output có explanation hợp lệ cho các ca kiểm tra P1."""
        return parse_prediction(json.dumps({"answer": answer, "unit": unit,
                                            "explanation": "Áp dụng công thức.", "reasoning": None}))

    def test_numeric_notation(self):
        """Các cách ghi số mũ hợp lệ phải cho cùng giá trị."""
        for text in ("1.2 * 10^-6", "1.2e-6", "1.2 × 10⁻⁶", "1.2 * 10^{-6}"):
            self.assertAlmostEqual(numeric_value(text), 1.2e-6)
        self.assertAlmostEqual(numeric_value("sqrt(2)*3"), 2**0.5 * 3)

    def test_tiny_value_is_not_zero(self):
        """Không dùng atol lớn khiến đáp án 0 khớp lực cực nhỏ."""
        score = score_prediction(self.prediction("0", "N"), {"answer": "1e-27", "unit": "N"})
        self.assertFalse(score["tolerant"])

    def test_relative_tolerance_uses_gold(self):
        """Dung sai tính theo độ lớn gold; exact vẫn báo riêng."""
        score = score_prediction(self.prediction("100.5", "V"), {"answer": "100", "unit": "V"})
        self.assertTrue(score["tolerant"])
        self.assertFalse(score["exact"])

    def test_unit_case_and_scale(self):
        """Sai đơn vị/thang đo bị tính sai; không tự đổi mJ sang J."""
        for unit in ("mV", "v", "A"):
            self.assertFalse(score_prediction(self.prediction("1", unit),
                                             {"answer": "1", "unit": "V"})["tolerant"])
        self.assertEqual(unit_ascii("N*m^-2"), "N*m^-2")
        self.assertEqual(unit_ascii("µF"), "uF")
        self.assertEqual(unit_ascii("μF"), "uF")
        self.assertEqual(unit_ascii("—"), "")

    def test_all_parts_required(self):
        """Một phần sai, đảo thứ tự hay thiếu đơn vị làm cả câu sai."""
        gold = {"answer": "0.50; 1.25", "unit": "uC; uJ"}
        self.assertTrue(score_prediction(self.prediction("0.5; 1.25", "µC; μJ"), gold)["tolerant"])
        for answer, unit in (("0.5; 1.3", "uC; uJ"), ("1.25; 0.5", "uJ; uC"),
                             ("0.5", "uC"), ("0.5; 1.25", "uC")):
            self.assertFalse(score_prediction(self.prediction(answer, unit), gold)["tolerant"])

    def test_output_schema(self):
        """JSON hỏng, answer kiểu số hay explanation rỗng không được chấm đúng."""
        for text in ("không phải JSON", "[]", '{"answer": 1, "unit":"V", "explanation":"x"}',
                     '{"answer":"1", "unit":"V", "explanation":""}'):
            self.assertIsNone(parse_prediction(text))

    def test_safe_expressions(self):
        """Không thực thi Python hoặc coi biểu thức có biến là số."""
        for text in ("__import__('os').system('echo unsafe')", "sqrt(2)*F0", "10**10000", "nan"):
            self.assertIsNone(numeric_value(text))
        score = score_prediction(self.prediction("F0*sqrt(2)", "N"),
                                 {"answer": "sqrt(2)*F0", "unit": "N"})
        self.assertFalse(score["tolerant"])
        self.assertTrue(score["unsupported_symbolic"])


class LoaderTests(unittest.TestCase):
    """Giữ nguyên raw text và không tự chọn nhầm file Kaggle."""

    def test_raw_text_and_missing_label(self):
        """Không strip/normalize question/cot; mẫu thiếu nhãn được ghi ID."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_dataset.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "question", "cot", "answer", "unit"])
                writer.writerow(["a", "  Câu  hỏi\n", "Lời giải\n\n", "1", "μF"])
                writer.writerow(["b", "Câu hỏi", "Lời giải", "", ""])
            rows, skipped = load_csv(path, "train", True)
            self.assertEqual(rows[0]["question"], "  Câu  hỏi\n")
            self.assertEqual(rows[0]["cot"], "Lời giải\n\n")
            self.assertEqual(rows[0]["unit"], "μF")
            self.assertEqual(skipped, ["b"])
            with self.assertRaises(ValueError):
                load_csv(path, "dev", True)

    def test_ambiguous_input(self):
        """Hai dataset cùng tên phải yêu cầu chọn đường dẫn cụ thể."""
        with tempfile.TemporaryDirectory() as directory:
            for name in ("one", "two"):
                folder = Path(directory) / name
                folder.mkdir()
                (folder / "train_dataset.csv").touch()
            with self.assertRaises(ValueError):
                find_csv(directory, "train")

    def test_inference_has_no_gold(self):
        """Prompt sinh dev nhận duy nhất question ngoài system instruction."""
        messages = prompt_messages("Điện dung bằng bao nhiêu?")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "Điện dung bằng bao nhiêu?")


class NativeModelTests(unittest.TestCase):
    """Model Qwen3.5 rất nhỏ để kiểm tra loss, gradient và mapping checkpoint."""

    @staticmethod
    def config():
        """Giữ cả DeltaNet/full attention nhưng dùng ít tham số để test CPU."""
        from transformers import Qwen3_5TextConfig

        return Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2],
                             "mrope_interleaved": True}, use_cache=False)

    def test_loss_and_gradient(self):
        """Chunked loss phải bằng loss chuẩn và chỉ adapter nhận gradient."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5ForCausalLM
        from baseline.model import ChunkedLoss

        torch.manual_seed(42)
        base = Qwen3_5ForCausalLM(self.config())
        base.requires_grad_(False)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                                              target_modules=["q_proj", "in_proj_qkv", "gate_proj"]))
        ids = torch.randint(0, 64, (1, 12))
        labels = ids.clone()
        labels[:, :4] = -100
        mask = torch.ones_like(ids)
        expected = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
        learner = ChunkedLoss(model, 3)
        actual = learner(ids, mask, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for name, parameter in model.named_parameters() if "lora_" in name))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if "lora_" not in name))

    def test_multimodal_checkpoint_to_text(self):
        """Nạp text backbone phải giữ đúng trọng số từ checkpoint multimodal."""
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

        full_config = Qwen3_5Config(
            text_config=self.config().to_dict(), vision_config={
                "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                "out_hidden_size": 32, "num_position_embeddings": 16, "patch_size": 2,
                "spatial_merge_size": 2, "temporal_patch_size": 1})
        full = Qwen3_5ForConditionalGeneration(full_config)
        with tempfile.TemporaryDirectory() as directory:
            full.save_pretrained(directory)
            text, info = Qwen3_5ForCausalLM.from_pretrained(
                directory, config=full_config.text_config, output_loading_info=True)
            self.assertFalse(info["missing_keys"])
            for name, weight in full.model.language_model.state_dict().items():
                torch.testing.assert_close(text.model.state_dict()[name], weight)

    def test_adapter_round_trip(self):
        """Adapter sau một optimizer step phải giữ dự đoán khi lưu/nạp lại."""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3_5ForCausalLM

        torch.manual_seed(42)
        base = Qwen3_5ForCausalLM(self.config())
        original = {name: weight.detach().clone() for name, weight in base.state_dict().items()}
        base.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                                              target_modules=["q_proj", "in_proj_qkv", "gate_proj"]))
        ids = torch.randint(0, 64, (1, 12))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        optimizer.step()
        model.eval()
        expected = model(input_ids=ids, use_cache=False).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored_base = Qwen3_5ForCausalLM(self.config())
            restored_base.load_state_dict(original)
            restored = PeftModel.from_pretrained(restored_base, directory).eval()
            actual = restored(input_ids=ids, use_cache=False).logits.detach()
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
