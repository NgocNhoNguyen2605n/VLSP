"""Bỏ phiếu nhiều phần, biểu diễn số, mẫu lỗi và cấu hình sampling."""

import json
import unittest

from baseline.consistency import decoding_settings, select_consistent


def candidate(answer, unit="N", finished=True, explanation="Áp dụng công thức."):
    return {"raw_output": json.dumps({"answer": answer, "unit": unit,
                                      "explanation": explanation, "reasoning": None}, ensure_ascii=False),
            "finished": finished}


class ConsistencyTests(unittest.TestCase):
    def test_equivalent_numeric_expressions_vote_together_but_not_five_percent_tolerance(self):
        rows = [candidate("2"), candidate("2.0"), candidate("4/2"), candidate("2.01"), candidate("2.02")]
        selected, info = select_consistent(rows)
        self.assertEqual(selected, 0)
        self.assertEqual(info["vote_count"], 3)
        self.assertTrue(info["majority"])
        self.assertEqual(len(info["groups"]), 3)

    def test_units_are_ascii_normalized_without_prefix_conversion(self):
        rows = [candidate("2", "μF"), candidate("2", "uF"), candidate("2", "F"),
                candidate("2e-6", "F"), candidate("2", "UF")]
        selected, info = select_consistent(rows)
        self.assertEqual(selected, 0)
        self.assertEqual(info["vote_count"], 2)
        self.assertEqual(len(info["groups"]), 4)

    def test_multipart_tuple_wins_without_mixing_parts_from_different_explanations(self):
        rows = [candidate("1; 2", "A; W", explanation="Lời giải thứ nhất."),
                candidate("1; 3", "A; W"), candidate("4; 2", "A; W"),
                candidate("1.0; 3.00", "A; W"), candidate("1; 3", "A; W")]
        selected, info = select_consistent(rows)
        self.assertEqual(selected, 1)
        self.assertEqual(info["vote_count"], 3)
        self.assertEqual(json.loads(rows[selected]["raw_output"])["answer"], "1; 3")

    def test_invalid_unfinished_and_mismatched_parts_are_excluded(self):
        rows = [candidate("8", finished=False), {"raw_output": "broken", "finished": True},
                candidate("1; 2", "N"), candidate("6"), candidate("6.0")]
        selected, info = select_consistent(rows)
        self.assertEqual(selected, 3)
        self.assertEqual(info["valid_samples"], 2)
        self.assertFalse(info["majority"])
        self.assertEqual(info["candidates"][0]["error"], "generation_unfinished")

    def test_ties_use_first_candidate_and_report_no_majority(self):
        selected, info = select_consistent([candidate("8"), candidate("6"), candidate("6"),
                                            candidate("8"), candidate("9")])
        self.assertEqual(selected, 0)
        self.assertTrue(info["tied"])
        self.assertFalse(info["majority"])

    def test_no_valid_candidates_and_very_small_nonzero_numbers(self):
        selected, info = select_consistent([candidate("1", finished=False)] * 5)
        self.assertIsNone(selected)
        self.assertEqual(info["valid_samples"], 0)
        _, info = select_consistent([candidate("0"), candidate("1e-15"), candidate("1.0e-15")])
        self.assertEqual(info["vote_count"], 2)
        self.assertEqual(len(info["groups"]), 2)

    def test_sampling_parameters_are_validated_and_one_sample_is_greedy(self):
        self.assertEqual(decoding_settings({"self_consistency_samples": 5}),
                         {"samples": 5, "temperature": 0.7, "top_p": 0.8})
        self.assertEqual(decoding_settings({"self_consistency_samples": 1}),
                         {"samples": 1, "temperature": 0.0, "top_p": 1.0})
        for invalid in ({"self_consistency_samples": True}, {"self_consistency_samples": 0},
                        {"sampling_temperature": 0}, {"sampling_temperature": float("nan")},
                        {"sampling_top_p": 1.1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decoding_settings(invalid)
