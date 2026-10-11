"""Chấm dev sau mỗi epoch và chọn checkpoint theo đáp án + đơn vị."""

import contextlib
import json
import math
from pathlib import Path

import torch

from .evaluate import generate_batch
from .metrics import parse_prediction, score_prediction
from .runtime import save_json
from .progress import BatchProgress


def checkpoint_rank(result):
    """Đúng đáp án/đơn vị, exact, loss, rồi epoch sớm hơn."""
    total = result["total"]
    loss = result["validation_loss"]
    if total <= 0 or not math.isfinite(loss):
        raise ValueError("Checkpoint thiếu validation hợp lệ.")
    return (result["content_tolerant"] / total, result["content_exact"] / total,
            -loss, -result["epoch"])


def validate_epoch(learner, loader, rows, tokenizer, cfg, epoch, result_dir, primary=True):
    """Loss có trọng số token; sinh một lần mỗi câu và so đáp án + đơn vị."""
    batch_size = cfg.get("validation_batch_size", 4)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("validation_batch_size phải là số nguyên dương.")
    learner.eval()
    model = learner.model
    device = model.get_input_embeddings().weight.device
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad(), BatchProgress(f"Validation loss epoch {epoch}", len(loader), enabled=primary,
                                       log_every=cfg.get("log_every", 10)) as progress:
        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16):
                loss = learner(**batch)
            failed = (~torch.isfinite(loss)).to(dtype=torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
            if failed.item():
                raise FloatingPointError(f"Validation loss không hữu hạn ở epoch {epoch}.")
            tokens = int(batch["labels"][:, 1:].ne(-100).sum())
            total_loss += loss.item() * tokens
            total_tokens += tokens
            progress.update(index)
    if not total_tokens:
        raise ValueError("Validation không có token assistant.")
    result_dir = Path(result_dir)
    if primary:
        result_dir.mkdir(parents=True, exist_ok=True)
    result = {"seed": cfg["seed"], "epoch": epoch, "total": len(rows),
              "validation_loss": total_loss / total_tokens,
              "content_tolerant": 0, "content_exact": 0, "valid_json": 0,
              "over_time": 0, "p1_estimate": 0, "unfinished_generation": 0}
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    result.update(generation_batch_size=batch_size,
                  generation_timing="seconds là thời gian chung của batch, không chia cho số câu")
    try:
        handle_context = ((result_dir / "predictions.jsonl").open("w", encoding="utf-8")
                          if primary else contextlib.nullcontext(None))
        # Hai rank phải cùng sinh đúng batch, kể cả batch cuối thiếu 4 câu.
        with handle_context as handle, BatchProgress(f"Validation đáp án epoch {epoch}", math.ceil(len(rows) / batch_size), enabled=primary,
                                                     log_every=cfg.get("log_every", 10)) as progress:
            for index, start in enumerate(range(0, len(rows), batch_size), start=1):
                batch_rows = rows[start:start + batch_size]
                outputs = generate_batch(model, tokenizer, [row["question"] for row in batch_rows], cfg)
                if len(outputs) != len(batch_rows):
                    raise RuntimeError("Generation trả sai số câu trong batch validation.")
                for row, (raw, elapsed, token_count, finished) in zip(batch_rows, outputs):
                    prediction = parse_prediction(raw)
                    score = score_prediction(prediction, row, cfg["rtol"], cfg["atol"])
                    over_time = elapsed > cfg["timeout_seconds"]
                    result["content_tolerant"] += score["tolerant"]
                    result["content_exact"] += score["exact"]
                    result["valid_json"] += prediction is not None
                    result["over_time"] += over_time
                    result["unfinished_generation"] += not finished
                    result["p1_estimate"] += score["tolerant"] and not over_time
                    if handle is not None:
                        handle.write(json.dumps({"query_id": row["id"], "prediction": prediction,
                                                 "raw_output": raw, "gold_answer": row["answer"],
                                                 "gold_unit": row["unit"], "score": score,
                                                 "seconds": elapsed, "generated_tokens": token_count,
                                                 "finished": finished, "over_time": over_time,
                                                 "generation_batch": index,
                                                 "generation_batch_size": len(batch_rows)}, ensure_ascii=False) + "\n")
                        handle.flush()
                progress.update(index)
    finally:
        model.config.use_cache = previous_cache
    result["accuracy"] = result["content_tolerant"] / result["total"]
    result["selection_metric"] = "content_tolerant accuracy; không áp dụng timeout khi xếp hạng"
    if primary:
        save_json(result_dir / "metrics.json", result)
    return result
