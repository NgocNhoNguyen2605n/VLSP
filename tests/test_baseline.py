"""Kiểm tra các điểm dễ làm lệch kết quả; không cần tải trọng số backbone."""

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


class TrainingTests(unittest.TestCase):
    """Kiểm tra overflow thực sự với GradScaler CPU và chặn chấm run chưa hoàn tất."""

    def test_overflow_then_recovery(self):
        """Overflow không đổi trọng số/scheduler; gradient hữu hạn cập nhật bình thường."""
        import torch
        from baseline.train import optimizer_step

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
        original = parameter.detach().clone()
        old_epoch = scheduler.last_epoch
        scaler.scale(parameter.square().sum()).backward()
        parameter.grad.fill_(float("inf"))
        skipped, old_scale, scale = optimizer_step([parameter], optimizer, scaler, scheduler, 1.0)
        self.assertTrue(skipped)
        self.assertEqual((old_scale, scale), (128.0, 64.0))
        torch.testing.assert_close(parameter, original)
        self.assertEqual(scheduler.last_epoch, old_epoch)
        self.assertIsNone(parameter.grad)

        scaler.scale(parameter.square().sum()).backward()
        skipped, _, scale = optimizer_step([parameter], optimizer, scaler, scheduler, 1.0)
        self.assertFalse(skipped)
        self.assertEqual(scale, 64.0)
        self.assertLess(parameter.item(), original.item())
        self.assertEqual(scheduler.last_epoch, old_epoch + 1)

    def test_evaluate_requires_completed_training(self):
        """Sai thư mục hoặc run đang dở phải dừng trước khi tải backbone lên GPU."""
        from baseline.evaluate import evaluate

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "--run-dir"):
                evaluate(path)
            (path / "run.json").write_text('{"training_status":"running"}')
            with self.assertRaisesRegex(RuntimeError, "adapter cuối"):
                evaluate(path)
            (path / "adapter").mkdir()
            (path / "adapter" / "adapter_config.json").write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "Train chưa hoàn tất"):
                evaluate(path)


class AttentionStabilityTests(unittest.TestCase):
    """Tái hiện tràn QK^T FP16 và kiểm tra đường attention thay thế."""

    def test_sdpa_math_survives_fp16_attention_overflow(self):
        import torch
        from types import SimpleNamespace
        from baseline.model import configure_attention
        from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        configure_attention()
        module = SimpleNamespace(num_key_value_groups=1, training=False, is_causal=False)
        query = torch.full((1, 1, 2, 8), 300.0, dtype=torch.float16, requires_grad=True)
        key = torch.full_like(query, 300.0, requires_grad=True)
        value = torch.ones_like(query, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.float16):
            broken, _ = eager_attention_forward(module, query, key, value, None, scaling=8**-0.5)
            stable, _ = sdpa_attention_forward(module, query, key, value, None, scaling=8**-0.5)
        self.assertFalse(torch.isfinite(broken).all())
        torch.testing.assert_close(stable, torch.ones_like(stable))
        stable.float().sum().backward()
        self.assertTrue(all(tensor.grad is not None and torch.isfinite(tensor.grad).all()
                            for tensor in (query, key, value)))

    def test_diagnostic_names_layer_and_removes_hooks(self):
        import torch
        from baseline.model import diagnose_nonfinite

        class Learner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Sequential(torch.nn.Linear(2, 2))

            def forward(self, input_ids):
                return self.model(input_ids.float()).float().sum()

        learner = Learner()
        batch = {"input_ids": torch.ones((1, 2), dtype=torch.long)}
        with torch.no_grad():
            learner.model[0].weight.fill_(float("inf"))
        detail = diagnose_nonfinite(learner, batch)
        self.assertIn("Output không hữu hạn ở 0", detail)
        self.assertFalse(learner.model[0]._forward_hooks)
        with torch.no_grad():
            learner.model[0].weight.fill_(1.0)
        self.assertIn("projection/norm hữu hạn", diagnose_nonfinite(learner, batch))
        self.assertFalse(learner.model[0]._forward_hooks)


class NativeModelTests(unittest.TestCase):
    """Model Qwen2 rất nhỏ để kiểm tra loss, gradient và checkpoint."""

    @staticmethod
    def config(tied=False):
        """Kiến trúc Qwen2.5-Math thu nhỏ, chạy trên CPU."""
        from transformers import Qwen2Config

        return Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            tie_word_embeddings=tied, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=128, use_cache=False)

    def test_loss_and_gradient(self):
        """Chunked loss phải bằng loss chuẩn và chỉ adapter nhận gradient."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2ForCausalLM
        from baseline.model import ChunkedLoss

        for tied in (False, True):
            with self.subTest(tied_embeddings=tied):
                torch.manual_seed(42)
                base = Qwen2ForCausalLM(self.config(tied))
                base.requires_grad_(False)
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                      "gate_proj", "up_proj", "down_proj"]))
                ids = torch.randint(0, 64, (2, 12))
                labels = ids.clone()
                labels[:, :4] = -100
                labels[1, 9:] = -100
                mask = torch.ones_like(ids)
                mask[1, 9:] = 0
                expected = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
                learner = ChunkedLoss(model, 3)
                actual = learner(ids, mask, labels)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                    for name, parameter in model.named_parameters() if "lora_" in name))
                self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                                    if "lora_" not in name))

    def test_checkpoint_and_backbone_plan(self):
        """Đếm đúng tham số, chia đủ lớp và nạp checkpoint text không thiếu key."""
        import torch
        from transformers import Qwen2ForCausalLM
        from baseline.model import backbone_plan, layer_map

        for tied in (False, True):
            with self.subTest(tied_embeddings=tied):
                config = self.config(tied)
                model = Qwen2ForCausalLM(config)
                mapping = layer_map(config, 2)
                self.assertEqual(mapping["model.layers.0"], 0)
                self.assertEqual(mapping["model.layers.1"], 1)
                self.assertEqual(mapping["lm_head"], 0 if tied else 1)
                self.assertEqual(mapping["model.norm"], mapping["lm_head"])
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    cfg = {"model_id": directory, "revision": "local-test",
                           "gpu_count": 2, "internal_experiment": False}
                    text_config, actual_map, plan = backbone_plan(cfg)
                    self.assertEqual(actual_map, mapping)
                    count = sum(p.numel() for p in model.parameters())
                    self.assertEqual(plan["base_parameters"], count)
                    self.assertAlmostEqual(sum(plan["fp16_weight_gib_per_gpu"]), count * 2 / 2**30)
                    self.assertTrue(plan["within_8b_backbone"])
                    restored, info = Qwen2ForCausalLM.from_pretrained(
                        directory, config=text_config, output_loading_info=True)
                    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                        self.assertFalse(info.get(key))
                    for name, weight in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], weight)
        with self.assertRaises(ValueError):
            layer_map(self.config(), 3)

    def test_adapter_round_trip(self):
        """Adapter sau một optimizer step phải giữ dự đoán khi lưu/nạp lại."""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen2ForCausalLM

        for tied in (False, True):
            with self.subTest(tied_embeddings=tied):
                torch.manual_seed(42)
                base = Qwen2ForCausalLM(self.config(tied))
                original = {name: weight.detach().clone() for name, weight in base.state_dict().items()}
                base.requires_grad_(False)
                model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                      "gate_proj", "up_proj", "down_proj"]))
                ids = torch.randint(0, 64, (1, 12))
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
                model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
                optimizer.step()
                model.eval()
                expected = model(input_ids=ids, use_cache=False).logits.detach()
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored_base = Qwen2ForCausalLM(self.config(tied))
                    restored_base.load_state_dict(original)
                    restored = PeftModel.from_pretrained(restored_base, directory).eval()
                    actual = restored(input_ids=ids, use_cache=False).logits.detach()
                    torch.testing.assert_close(actual, expected)

    def test_training_loader_fp16(self):
        """Luồng load_model thật giữ backbone FP16 và chỉ học LoRA FP32."""
        import torch
        from unittest.mock import patch
        from transformers import Qwen2ForCausalLM
        from baseline.model import ChunkedLoss, backbone_plan, load_model

        with tempfile.TemporaryDirectory() as directory:
            Qwen2ForCausalLM(self.config()).save_pretrained(directory)
            cfg = {"model_id": directory, "revision": "local-test", "gpu_count": 2,
                   "internal_experiment": False, "lora_rank": 2, "lora_alpha": 4,
                   "lora_dropout": 0.0}

            def cpu_plan(config):
                text, mapping, plan = backbone_plan(config)
                return text, {key: "cpu" for key in mapping}, plan

            # Chỉ thay vị trí GPU bằng CPU, giữ nguyên nạp checkpoint và tạo LoRA.
            with patch("baseline.model.require_gpus"), patch("baseline.model.backbone_plan", cpu_plan):
                model, plan = load_model(cfg, training=True)
            self.assertGreater(plan["trainable_parameters"], 0)
            self.assertEqual(model.config._attn_implementation, "sdpa")
            self.assertEqual(plan["attention"], "sdpa math; FP32 intermediates")
            self.assertFalse(model.config.use_cache)
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    self.assertTrue(parameter.requires_grad)
                    self.assertEqual(parameter.dtype, torch.float32)
                else:
                    self.assertFalse(parameter.requires_grad)
                    self.assertEqual(parameter.dtype, torch.float32 if "norm" in name else torch.float16)
            ids = torch.randint(0, 64, (1, 12))
            labels = ids.clone()
            labels[:, :4] = -100
            # Train dùng autocast FP16 khi norm/adapter ở FP32.
            with torch.autocast("cpu", dtype=torch.float16):
                loss = ChunkedLoss(model, 3)(ids, torch.ones_like(ids), labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

if __name__ == "__main__":
    unittest.main()
