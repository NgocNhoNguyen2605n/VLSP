"""Chấm toàn bộ dev qua API, chỉ gửi ID/câu hỏi; giữ nhãn ở local."""
import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from baseline.data import load_csv
from baseline.metrics import score_prediction
from .schema import validate_response


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(records, total, rtol, atol):
    times = sorted(r["seconds"] for r in records)
    statuses = dict(Counter(r["status"] for r in records))
    groups = defaultdict(list)
    for record in records:
        prefix = re.match(r"[A-Za-z]+", record["query_id"])
        groups[prefix[0] if prefix else "other"].append(record)
    exact = sum(r["score"]["exact"] and r["status"] == "ok" for r in records)
    correct = sum(r["score"]["tolerant"] and r["status"] == "ok" for r in records)
    return {"total": total, "processed": len(records), "complete": len(records) == total,
            "strict_exact_correct": exact, "strict_exact_accuracy": exact / total,
            "tolerant_correct": correct, "tolerant_accuracy": correct / total,
            "content_tolerant_correct": sum(r["score"]["tolerant"] for r in records),
            "valid_responses": sum(r["prediction"] is not None for r in records), "statuses": statuses,
            "unsupported_symbolic": sum(r["score"]["unsupported_symbolic"] for r in records),
            "rtol": rtol, "atol": atol, "request_limit_seconds": 60,
            "scorer": "baseline.metrics.score_prediction; all parts and ASCII-normalized units; no SI prefix conversion",
            "official_score": False,
            "latency_seconds": {"mean": statistics.mean(times) if times else None,
                "median": statistics.median(times) if times else None,
                "p95": times[min(len(times)-1, int(.95*(len(times)-1)))] if times else None,
                "max": max(times) if times else None},
            "by_prefix": {key: {"total": len(rows),
                "correct": sum(r["score"]["tolerant"] and r["status"] == "ok" for r in rows),
                "errors": sum(r["status"] != "ok" for r in rows)} for key, rows in sorted(groups.items())}}


def evaluate(dev_file, base_url, output, model_name, rtol=.05, atol=0, resume=False):
    gold, _ = load_csv(dev_file, "dev")
    base_url = base_url.rstrip("/")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"dev_sha256": hashlib.sha256(Path(dev_file).read_bytes()).hexdigest(),
                "dev_file": str(Path(dev_file).resolve()), "base_url": base_url,
                "model_name": model_name, "rtol": rtol, "atol": atol, "total": len(gold)}
    records = []
    if resume:
        previous = json.loads((folder / "run.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Không resume run khác dữ liệu/model/endpoint/scorer.")
        records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines() if line]
        if [r["query_id"] for r in records] != [r["id"] for r in gold[:len(records)]]:
            raise ValueError("Log không khớp thứ tự dev hoặc có ID trùng.")
    elif any(folder.iterdir()):
        raise FileExistsError("Output đã có dữ liệu; chọn folder mới hoặc --resume.")
    with httpx.Client(timeout=60) as client:
        health = client.get(base_url + "/healthz"); health.raise_for_status()
        health = health.json()
        if health.get("status") != "ready" or health.get("model") != model_name:
            raise ValueError("API chưa ready hoặc model không khớp.")
        models = client.get(base_url + "/v1/models"); models.raise_for_status()
        run = {"identity": identity, "health": health, "models": models.json(), "started_unix": time.time()}
        if resume and previous["health"] != health:
            raise ValueError("Model/precision đã thay đổi so với run trước.")
        save_json(folder / "run.json", previous if resume else run)
        try:
            with (folder / "predictions.jsonl").open("a", encoding="utf-8") as log:
                for index in range(len(records), len(gold)):
                    row = gold[index]
                    if index % 20 == 0:
                        current = client.get(base_url + "/healthz"); current.raise_for_status()
                        if current.json() != health:
                            raise ValueError("API/model/precision đổi giữa run; dừng để tránh chấm lẫn.")
                    start = time.perf_counter()
                    record = {"query_id": row["id"], "question": row["question"],
                              "gold_answer": row["answer"], "gold_unit": row["unit"],
                              "prediction": None, "status": "ok", "http_status": None}
                    try:
                        # Payload chỉ có hai trường; cot/answer/unit không được gửi.
                        response = client.post(base_url + "/predict", json={"query_id": row["id"], "query": row["question"]})
                        record["http_status"] = response.status_code
                        response.raise_for_status()
                        record["prediction"] = validate_response(response.json(), row["id"])
                    except httpx.TimeoutException as error:
                        record.update(status="timeout", error=str(error))
                    except httpx.HTTPStatusError as error:
                        code = error.response.status_code
                        record.update(status="timeout" if code == 504 else "invalid_output" if code == 502 else "http_error",
                                      error=f"HTTP {code}: {error.response.text[:500]}")
                    except httpx.HTTPError as error:
                        record.update(status="network_error", error=str(error))
                    except ValueError as error:
                        record.update(status="invalid_output", error=str(error))
                    record["seconds"] = time.perf_counter() - start
                    if record["seconds"] >= 60 and record["status"] == "ok":
                        record["status"] = "timeout"
                    record["score"] = score_prediction(record["prediction"], row, rtol, atol)
                    log.write(json.dumps(record, ensure_ascii=False) + "\n"); log.flush()
                    records.append(record)
                    result = summarize(records, len(gold), rtol, atol)
                    save_json(folder / "metrics.json", result)
                    if (index+1) % 10 == 0 or index+1 == len(gold):
                        print(f"DEV {index+1}/{len(gold)} | correct={result['tolerant_correct']}/{index+1} | statuses={result['statuses']} | mean={result['latency_seconds']['mean']:.2f}s", flush=True)
        finally:
            result = summarize(records, len(gold), rtol, atol)
            save_json(folder / "metrics.json", result)
            fields = ["query_id", "question", "gold_answer", "gold_unit", "predicted_answer", "predicted_unit", "exact", "tolerant", "status", "seconds", "explanation", "error"]
            with (folder / "details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for record in records:
                    p = record["prediction"] or {}
                    writer.writerow({**{k: record[k] for k in ["query_id", "question", "gold_answer", "gold_unit", "status", "seconds"]},
                        "predicted_answer": p.get("answer", ""), "predicted_unit": p.get("unit", ""),
                        "exact": record["score"]["exact"] and record["status"] == "ok",
                        "tolerant": record["score"]["tolerant"] and record["status"] == "ok",
                        "explanation": p.get("explanation", ""), "error": record.get("error", "")})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Chấm full dev qua API đang phục vụ")
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rtol", type=float, default=.05)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate(args.dev_file, args.base_url, args.output_dir, args.model_name, args.rtol, args.atol, args.resume)


if __name__ == "__main__":
    main()
