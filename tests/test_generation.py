"""Batch generation: left padding, EOS từng câu, batch cuối và chấm đúng ID."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from baseline.evaluate import generate_batch, generate_once
from baseline.validation import validate_epoch


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [1] + [3] * int(messages[1]["content"])

    def decode(self, tokens, **kwargs):
        return str([token for token in tokens if token not in (0, 2)])


class GenerationTests(unittest.TestCase):
    cfg = {"gpu_count": 2, "max_new_tokens": 4, "timeout_seconds": 60}

    def test_left_padding_and_each_eos_without_counting_trailing_padding(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64),
                                get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)))

        def generate(**kwargs):
            self.assertEqual(kwargs["input_ids"].tolist(), [[0, 0, 1, 3], [1, 3, 3, 3],
                                                           [0, 1, 3, 3], [0, 0, 0, 1]])
            self.assertEqual(kwargs["attention_mask"].tolist(), [[0, 0, 1, 1], [1, 1, 1, 1],
                                                                [0, 1, 1, 1], [0, 0, 0, 1]])
            self.assertFalse(kwargs["generation_config"].do_sample)
            self.assertFalse(kwargs["synced_gpus"])
            tail = torch.tensor([[9, 2, 0, 0], [8, 7, 6, 2], [5, 4, 3, 1], [2, 0, 0, 0]])
            return torch.cat((kwargs["input_ids"], tail), dim=1)

        model.generate = generate
        with patch("baseline.evaluate.synchronize"), patch("baseline.evaluate.time.perf_counter", side_effect=[10, 13]):
            outputs = generate_batch(model, Tokenizer(), ["1", "3", "2", "0"], self.cfg)
        self.assertEqual(outputs, [("[9]", 3, 2, True), ("[8, 7, 6]", 3, 4, True),
                                   ("[5, 4, 3, 1]", 3, 4, False), ("[]", 3, 1, True)])
        with patch("baseline.evaluate.generate_batch", return_value=[outputs[0]]) as batch:
            self.assertEqual(generate_once(model, Tokenizer(), "1", self.cfg), outputs[0])
            batch.assert_called_once_with(model, unittest.mock.ANY, ["1"], self.cfg)

    def test_real_qwen_batch_greedy_matches_single_queries(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                    max_position_embeddings=64)).eval()
        questions = ["1", "3", "2", "0", "4"]
        with patch("baseline.evaluate.synchronize"):
            singles = [generate_once(model, Tokenizer(), question, self.cfg) for question in questions]
            outputs = generate_batch(model, Tokenizer(), questions[:4], self.cfg)
            outputs += generate_batch(model, Tokenizer(), questions[4:], self.cfg)
        self.assertEqual([(raw, count, finished) for raw, _, count, finished in outputs],
                         [(raw, count, finished) for raw, _, count, finished in singles])

    def test_empty_batch_and_context_limit_are_rejected_before_generation(self):
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=6))
        with self.assertRaisesRegex(ValueError, "rỗng"):
            generate_batch(model, Tokenizer(), [], self.cfg)
        with patch('baseline.evaluate.synchronize'), self.assertRaisesRegex(ValueError, "vượt context"):
            generate_batch(model, Tokenizer(), ['3'], self.cfg)

    def test_validation_four_plus_one_scores_every_id_and_restores_cache(self):
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False),
                                get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)))
        class Learner:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                return torch.tensor(2.)
        learner = Learner()
        learner.model = model
        rows = [{"id": str(i), "question": str(i), "answer": str(i), "unit": "V"} for i in range(5)]
        loader = [{"labels": torch.tensor([[-100, 1]])}]
        cfg = {**self.cfg, "seed": 42, "validation_batch_size": 4, "rtol": .01, "atol": 0}
        calls = []
        def generation(model, tokenizer, questions, config):
            calls.append(questions)
            return [(json.dumps({"answer": question, "unit": "V", "explanation": "Giải"}), .1, 10, True)
                    for question in questions]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch("baseline.validation.generate_batch", generation):
            result = validate_epoch(learner, loader, rows, Tokenizer(), cfg, 1, directory)
            records = [json.loads(line) for line in (Path(directory) / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(calls, [["0", "1", "2", "3"], ["4"]])
            self.assertEqual(result["content_tolerant"], 5)
            self.assertEqual(result["generation_batch_size"], 4)
            self.assertEqual(result["unfinished_generation"], 0)
            self.assertEqual([r["query_id"] for r in records], ["0", "1", "2", "3", "4"])
            self.assertEqual([r["generation_batch_size"] for r in records], [4, 4, 4, 4, 1])
            self.assertFalse(model.config.use_cache)
            with patch("baseline.validation.generate_batch", side_effect=RuntimeError("generation failed")):
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    validate_epoch(learner, loader, rows, Tokenizer(), cfg, 1, directory)
            self.assertFalse(model.config.use_cache)


if __name__ == "__main__":
    unittest.main()
