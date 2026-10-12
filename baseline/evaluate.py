"""Sinh và bỏ phiếu đáp án dev; báo P1 ước lượng, không dùng API."""

from .runtime import environment, save_json, setup_seed  # Seed, môi trường và báo cáo.

import argparse  # Chọn thư mục adapter và kết quả.
import hashlib  # Kiểm tra template giống lúc train.
import json  # Lưu từng prediction ngay sau khi sinh.
import time  # Đo thời gian mỗi câu theo deadline generation trong cấu hình (60 giây mặc định).
from pathlib import Path  # Đọc manifest và lưu JSONL.

import torch  # Inference không gradient, dùng FP16 trên hai GPU.
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList  # Greedy và deadline.

from .data import load_data, prompt_ids  # Prompt chỉ có câu hỏi.
from .consistency import decoding_settings, select_consistent
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


def generate_once(model, tokenizer, question, cfg, diagnostics=None):
    """Một lượt generation cho một câu; không retry hoặc dùng nhãn dev."""
    if diagnostics is None:
        return generate_batch(model, tokenizer, [question], cfg)[0]
    return generate_batch(model, tokenizer, [question], cfg, diagnostics=diagnostics)[0]


def generate_batch(model, tokenizer, questions, cfg, diagnostics=None):
    """Sinh nhiều mẫu trong một batch, trả một kết quả đã bỏ phiếu cho mỗi câu."""
    if not questions:
        raise ValueError("Batch generation không được rỗng.")
    synchronize(cfg["gpu_count"])
    start = time.perf_counter()
    decoding = decoding_settings(cfg)
    samples = decoding["samples"]
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
        do_sample=samples > 1, num_beams=1, num_return_sequences=samples,
        max_new_tokens=cfg["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True, temperature=decoding["temperature"] if samples > 1 else None,
        top_p=decoding["top_p"] if samples > 1 else None, top_k=0 if samples > 1 else None)
    with torch.inference_mode(), torch.autocast(tensor.device.type, dtype=torch.float16):
        output = model.generate(
            input_ids=tensor, attention_mask=mask,
            generation_config=generation,
            synced_gpus=torch.distributed.is_initialized(),
            stopping_criteria=StoppingCriteriaList([Deadline(start + cfg["timeout_seconds"])]))
    synchronize(cfg["gpu_count"])
    if output.shape[0] != len(questions) * samples:
        raise ValueError("Generation không trả đủ số mẫu cho mỗi câu hỏi.")
    candidates = []
    for tokens in output[:, width:].tolist():
        # Câu kết thúc sớm được pad trong khi các câu còn lại tiếp tục sinh.
        finished = tokenizer.eos_token_id in tokens
        if finished:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        candidates.append({"raw_output": text, "generated_tokens": len(tokens), "finished": finished})
    results = []
    for start_index in range(0, len(candidates), samples):
        group = candidates[start_index:start_index + samples]
        selected, details = select_consistent(group)
        if samples == 1:
            # Giữ hành vi greedy cũ, kể cả output bị cắt, để so sánh baseline.
            selected = 0
        winner = group[selected] if selected is not None else {
            "raw_output": "", "finished": all(candidate["finished"] for candidate in group)}
        results.append((winner["raw_output"], 0.0,
                        sum(candidate["generated_tokens"] for candidate in group), winner["finished"]))
        if diagnostics is not None:
            diagnostics.append(details)
    elapsed = time.perf_counter() - start
    return [(raw, elapsed, tokens, finished) for raw, _, tokens, finished in results]


def evaluate(run_dir, result_dir=None, data_root=None, train_file=None, dev_file=None,
             self_consistency_samples=None, sampling_temperature=None, sampling_top_p=None):
    """Chấm mọi dòng dev; báo riêng content đúng và đúng trong giới hạn thời gian."""
    run_dir = Path(run_dir)
    if (run_dir / "experiment.json").is_file():
        experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
        if experiment.get("status") != "complete":
            raise RuntimeError("Experiment chưa hoàn tất; cần train và chọn best checkpoint trước khi chấm dev.")
        if experiment.get("workflow") != "train_dev":
            run_dir = run_dir / "final"  # Đọc được kết quả workflow cũ đã hoàn tất.
        else:
            print(f"Evaluate best epoch {experiment['best_checkpoint']['epoch']}: {run_dir}", flush=True)
    if not (run_dir / "run.json").is_file():
        raise FileNotFoundError(f"{run_dir} thiếu run.json; --run-dir phải trùng --output-dir lúc train.")
    if not (run_dir / "adapter" / "adapter_config.json").is_file():
        raise RuntimeError(f"{run_dir} chưa có adapter cuối; hãy train thành công trước khi evaluate.")
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("training_status", "complete") != "complete":
        raise RuntimeError("Train chưa hoàn tất; không chấm adapter của lần chạy đang dở.")
    cfg = manifest["config"].copy()
    for key, value, default in (("self_consistency_samples", self_consistency_samples, 5),
                                ("sampling_temperature", sampling_temperature, 0.7),
                                ("sampling_top_p", sampling_top_p, 0.8)):
        if value is not None:
            cfg[key] = value
        else:
            cfg.setdefault(key, default)
    decoding = decoding_settings(cfg)
    if data_root:
        cfg.update(data_root=data_root, train_file=None, dev_file=None)
    if train_file is not None:
        cfg["train_file"] = train_file
    if dev_file is not None:
        cfg["dev_file"] = dev_file
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
    print(f"Evaluate {counts['total']} câu | samples={decoding['samples']} | seed={cfg['seed']} | "
          f"rtol={cfg['rtol']}, atol={cfg['atol']} (tạm thời).", flush=True)
    with (result_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(splits["dev"], start=1):
            diagnostics = []
            raw, elapsed, token_count, finished = generate_once(model, tokenizer, row["question"], cfg,
                                                              diagnostics=diagnostics)
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
                      "p1_estimate_correct": correct, "overlap_train": overlap,
                      "self_consistency": diagnostics[0] if diagnostics else None}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == counts["total"]:
                print(f"Dev {index}/{counts['total']} | đúng ước lượng={counts['p1_estimate']}", flush=True)
    summary = {**counts, "accuracy_estimate": counts["p1_estimate"] / counts["total"],
               "rtol": cfg["rtol"], "atol": cfg["atol"], "timeout_seconds": cfg["timeout_seconds"],
               "decoding": decoding,
               "official_metric": False, "environment": environment(), "data": info,
               "notes": ["BTC chưa công bố dung sai và evaluator chính thức.",
                         "Exact: chuỗi đáp án sau NFC/khoảng trắng + đơn vị alias ASCII.",
                         "Tolerant: biểu thức số an toàn; không đổi thang đơn vị, không suy luận tương đương biểu thức có biến.",
                         "Không chấm chất lượng explanation; yêu cầu explanation không rỗng.",
                         "Mẫu quá thời gian bị tính sai; deadline chỉ dừng ở ranh giới token."]}
    if (manifest.get("split") or {}).get("validation_source") == "dev":
        summary["notes"].append("Dev đã dùng chọn checkpoint; điểm này là validation, không phải test độc lập.")
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
    parser.add_argument("--train-file")
    parser.add_argument("--dev-file")
    parser.add_argument("--self-consistency-samples", type=int, help="Số mẫu/câu; mặc định run hoặc 5, dùng 1 để so greedy")
    parser.add_argument("--sampling-temperature", type=float)
    parser.add_argument("--sampling-top-p", type=float)
    args = parser.parse_args()
    evaluate(args.run_dir, args.result_dir, args.data_root, args.train_file, args.dev_file,
             args.self_consistency_samples, args.sampling_temperature, args.sampling_top_p)


if __name__ == "__main__":
    main()
