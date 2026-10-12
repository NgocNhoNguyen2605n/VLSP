"""Sampling và bỏ phiếu chỉ từ đầu ra model, không nhận nhãn tham chiếu."""

import math
from collections import defaultdict
from collections.abc import Mapping

from .metrics import canonical_text, numeric_value, parse_prediction, unit_ascii


def decoding_settings(config):
    """Một mẫu là greedy; nhiều mẫu dùng cùng tham số ở Transformers và vLLM."""
    get = config.get if isinstance(config, Mapping) else lambda key, default: getattr(config, key, default)
    samples = get("self_consistency_samples", 1)
    temperature = get("sampling_temperature", 0.7)
    top_p = get("sampling_top_p", 0.8)
    if type(samples) is not int or samples <= 0:
        raise ValueError("self_consistency_samples phải là số nguyên dương.")
    for name, value in (("sampling_temperature", temperature), ("sampling_top_p", top_p)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} phải là số dương hữu hạn.")
    if top_p > 1:
        raise ValueError("sampling_top_p phải nằm trong (0, 1].")
    return {"samples": samples, "temperature": float(temperature) if samples > 1 else 0.0,
            "top_p": float(top_p) if samples > 1 else 1.0}


def prediction_key(prediction):
    """Gộp biểu diễn số tương đương; không dùng dung sai chấm 5% hay đổi thang đơn vị."""
    answers = [part.strip() for part in prediction["answer"].split(";")]
    units = [unit_ascii(part) for part in prediction["unit"].split(";")]
    if len(answers) != len(units) or any(not answer for answer in answers):
        raise ValueError("Số phần answer và unit không khớp hoặc answer rỗng.")
    parts = []
    for answer, unit in zip(answers, units):
        value = numeric_value(answer)
        # 12 chữ số có nghĩa chỉ bỏ sai số biểu diễn float của các biểu thức số.
        normalized = ("number", format(value if value != 0 else 0.0, ".12g")) if value is not None else (
            "text", canonical_text(answer))
        parts.append((normalized, unit))
    return tuple(parts)


def select_consistent(candidates):
    """Chọn nhóm đông nhất; hòa lấy mẫu đầu tiên, không trộn đáp án giữa các phần."""
    groups = defaultdict(list)
    records = []
    for index, candidate in enumerate(candidates):
        record = {**candidate, "index": index, "valid": False}
        raw = candidate.get("raw_output")
        if not candidate.get("finished", False):
            record["error"] = "generation_unfinished"
        elif not isinstance(raw, str):
            record["error"] = "content_not_string"
        else:
            prediction = parse_prediction(raw)
            if prediction is None:
                record["error"] = "invalid_prediction"
            else:
                try:
                    key = prediction_key(prediction)
                except ValueError as error:
                    record["error"] = str(error)
                else:
                    record.update(valid=True, answer=prediction["answer"], unit=prediction["unit"])
                    groups[key].append(index)
        records.append(record)
    ranked = sorted(groups.values(), key=lambda indexes: (-len(indexes), indexes[0]))
    selected = ranked[0][0] if ranked else None
    votes = len(ranked[0]) if ranked else 0
    details = {"samples": len(candidates), "valid_samples": sum(r["valid"] for r in records),
               "selected_index": selected, "vote_count": votes,
               "majority": votes > len(candidates) / 2,
               "tied": len(ranked) > 1 and len(ranked[1]) == votes,
               "tie_break": "first_valid_candidate_in_tied_groups",
               "groups": [{"indices": indexes, "votes": len(indexes),
                           "answer": records[indexes[0]]["answer"],
                           "unit": records[indexes[0]]["unit"]} for indexes in ranked],
               "candidates": records}
    return selected, details
