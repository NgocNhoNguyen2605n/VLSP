"""Sinh đáp án dev một lần mỗi câu và báo P1 ước lượng, không dùng API."""

from .runtime import environment, save_json, setup_seed  # Seed, môi trường và báo cáo.

import argparse  # Chọn thư mục adapter và kết quả.
import hashlib  # Kiểm tra template giống lúc train.
import json  # Lưu từng prediction ngay sau khi sinh.
import time  # Đo thời gian mỗi câu theo giới hạn 60 giây.
from pathlib import Path  # Đọc manifest và lưu JSONL.

import torch  # Inference không gradient, dùng FP16 trên hai GPU.
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList  # Greedy và deadline.

from .data import load_data, prompt_ids  # Prompt chỉ có câu hỏi.
from .metrics import canonical_text, parse_prediction, score_prediction  # Evaluator P1 tạm thời.
from .model import load_model, load_tokenizer  # Nạp backbone và adapter đã lưu.


class Deadline(StoppingCriteria):
    """Dừng sinh ở ranh giới token; không bảo đảm ngắt kernel GPU đang chạy."""

    def __init__(self, deadline):
        """Lưu thời điểm kết thúc truy vấn, tính bằng perf_counter."""
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        """Kết quả boolean đúng device để Transformers dừng batch một câu."""
        return torch.full((input_ids.shape[0],), time.perf_counter() >= self.deadline,
                          dtype=torch.bool, device=input_ids.device)


def synchronize(gpu_count):
    """Đợi cả hai GPU để thời gian đo gồm toàn bộ tính toán CUDA."""
    distributed = torch.distributed.is_initialized()
    devices = [torch.cuda.current_device()] if distributed else range(gpu_count)
    for i in devices:
        torch.cuda.synchronize(i)


def generate_once(model, tokenizer, question, cfg):
    """Greedy một lần; không retry JSON hỏng, không dùng answer/cot của dev."""
    return generate_batch(model, tokenizer, [question], cfg)[0]


def generate_batch(model, tokenizer, questions, cfg):
    """Greedy với left padding; mỗi câu một output, thời gian chung của batch."""
    if not questions:
        raise ValueError("Batch generation không được rỗng.")
    synchronize(cfg["gpu_count"])
    start = time.perf_counter()
    prompts = [prompt_ids(tokenizer, question) for question in questions]
    width = max(map(len, prompts))
    if width + cfg["max_new_tokens"] > model.config.max_position_embeddings:
        raise ValueError("Prompt và max_new_tokens vượt context của model.")
    device = model.get_input_embeddings().weight.device
    tensor = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(tensor)
    for index, ids in enumerate(prompts):
        tensor[index, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[index, -len(ids):] = 1
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=cfg["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True, temperature=None, top_p=None, top_k=None)
    with torch.inference_mode(), torch.autocast(tensor.device.type, dtype=torch.float16):
        output = model.generate(
            input_ids=tensor, attention_mask=mask,
            generation_config=generation,
            synced_gpus=torch.distributed.is_initialized(),
            stopping_criteria=StoppingCriteriaList([Deadline(start + cfg["timeout_seconds"])]))
    synchronize(cfg["gpu_count"])
    elapsed = time.perf_counter() - start
    results = []
    for tokens in output[:, width:].tolist():
        # Câu kết thúc sớm được pad trong khi các câu còn lại tiếp tục sinh.
        finished = tokenizer.eos_token_id in tokens
        if finished:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append((text, elapsed, len(tokens), finished))
    return results


def evaluate(run_dir, result_dir=None, data_root=None):
    """Chấm mọi dòng dev; báo riêng content đúng và đúng trong giới hạn thời gian."""
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").is_file() and (run_dir / "experiment.json").is_file():
        experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
        if experiment.get("status") != "complete":
            raise RuntimeError("Experiment chưa hoàn tất; hãy retrain cấu hình đã chọn bằng "
                               f"python -m baseline.experiment --retrain-from {run_dir} trước khi chấm dev.")
        run_dir = run_dir / "final"
        print(f"Evaluate adapter retrain cuối: {run_dir}", flush=True)
    if not (run_dir / "run.json").is_file():
        raise FileNotFoundError(f"{run_dir} thiếu run.json; --run-dir phải trùng --output-dir lúc train.")
    if not (run_dir / "adapter" / "adapter_config.json").is_file():
        raise RuntimeError(f"{run_dir} chưa có adapter cuối; hãy train thành công trước khi evaluate.")
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("training_status", "complete") != "complete":
        raise RuntimeError("Train chưa hoàn tất; không chấm adapter của lần chạy đang dở.")
    cfg = manifest["config"].copy()
    if data_root:
        cfg.update(data_root=data_root, train_file=None, dev_file=None)
    setup_seed(cfg["seed"])
    result_dir = Path(result_dir) if result_dir else run_dir / "evaluation"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"{result_dir} đã có kết quả; truyền --result-dir khác để chạy lại.")
    splits, info = load_data(cfg)
    for split in ("train", "dev"):
        if split in manifest["data"] and info[split]["sha256"] != manifest["data"][split]["sha256"]:
            raise ValueError(f"{split} đã đổi từ lúc train; không so sánh các lần chạy khác dữ liệu.")
    tokenizer = load_tokenizer(cfg)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != manifest["template_sha256"]:
        raise ValueError("Chat template đã đổi từ lúc train.")
    model, _ = load_model(cfg, adapter=run_dir / "adapter")
    result_dir.mkdir(parents=True, exist_ok=True)
    train_questions = {canonical_text(row["question"]) for row in splits["train"]}
    counts = {"total": len(splits["dev"]), "valid_json": 0, "content_exact": 0,
              "content_tolerant": 0, "p1_estimate": 0, "over_time": 0,
              "unfinished_generation": 0, "unsupported_symbolic": 0,
              "overlap_samples": 0, "overlap_correct": 0, "unseen_correct": 0}
    print(f"Evaluate {counts['total']} mẫu | greedy | seed={cfg['seed']} | "
          f"rtol={cfg['rtol']}, atol={cfg['atol']} (tạm thời).", flush=True)
    with (result_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(splits["dev"], start=1):
            raw, elapsed, token_count, finished = generate_once(model, tokenizer, row["question"], cfg)
            prediction = parse_prediction(raw)
            if prediction is not None:
                prediction["query_id"] = row["id"]
            score = score_prediction(prediction, row, cfg["rtol"], cfg["atol"])
            timed_out = elapsed > cfg["timeout_seconds"]
            correct = score["tolerant"] and not timed_out
            overlap = canonical_text(row["question"]) in train_questions
            counts["valid_json"] += prediction is not None
            counts["content_exact"] += score["exact"]
            counts["content_tolerant"] += score["tolerant"]
            counts["p1_estimate"] += correct
            counts["over_time"] += timed_out
            counts["unfinished_generation"] += not finished
            counts["unsupported_symbolic"] += score["unsupported_symbolic"]
            counts["overlap_samples"] += overlap
            counts["overlap_correct"] += overlap and correct
            counts["unseen_correct"] += not overlap and correct
            record = {"query_id": row["id"], "prediction": prediction, "raw_output": raw,
                      "gold_answer": row["answer"], "gold_unit": row["unit"],
                      "seconds": elapsed, "generated_tokens": token_count, "finished": finished,
                      "score": score, "over_time": timed_out,
                      "p1_estimate_correct": correct, "overlap_train": overlap}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == counts["total"]:
                print(f"Dev {index}/{counts['total']} | đúng ước lượng={counts['p1_estimate']}", flush=True)
    summary = {**counts, "accuracy_estimate": counts["p1_estimate"] / counts["total"],
               "rtol": cfg["rtol"], "atol": cfg["atol"], "timeout_seconds": cfg["timeout_seconds"],
               "official_metric": False, "environment": environment(), "data": info,
               "notes": ["BTC chưa công bố dung sai và evaluator chính thức.",
                         "Exact: chuỗi đáp án sau NFC/khoảng trắng + đơn vị alias ASCII.",
                         "Tolerant: biểu thức số an toàn; không đổi thang đơn vị, không suy luận tương đương biểu thức có biến.",
                         "Không chấm chất lượng explanation; yêu cầu explanation không rỗng.",
                         "Mẫu quá thời gian bị tính sai; deadline chỉ dừng ở ranh giới token."]}
    save_json(result_dir / "metrics.json", summary)
    print(f"P1 ước lượng: {counts['p1_estimate']}/{counts['total']} | "
          f"content đúng={counts['content_tolerant']} | JSON hợp lệ={counts['valid_json']} | "
          f"quá thời gian={counts['over_time']}. Kết quả: {result_dir}", flush=True)
    return summary


def main():
    """Entry point; mặc định dùng toàn bộ cấu hình đã lưu của lần train."""
    parser = argparse.ArgumentParser(description="Evaluate dev ViPRA")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    evaluate(args.run_dir, args.result_dir, args.data_root)


if __name__ == "__main__":
    main()
