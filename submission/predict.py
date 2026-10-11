"""Đọc public test không nhãn; gọi /predict tuần tự và lưu tiến độ từng câu."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .schema import Query, validate_response


def load_queries(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, strict=True))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(rows, dict):
            rows = [rows]
    else:
        raise ValueError("Public test cần CSV, JSON list hoặc JSONL.")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Không có câu hỏi trong public test.")
    queries, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Mỗi câu hỏi phải là object.")
        # Chỉ lấy đúng hai trường. answer/cot/unit nếu có trong CSV không đi vào prompt.
        item = Query(query_id=row.get("query_id", row.get("id")),
                     query=row.get("query", row.get("question")))
        if item.query_id in seen:
            raise ValueError(f"ID trùng: {item.query_id}")
        seen.add(item.query_id)
        queries.append(item)
    return queries


def run_predictions(input_file, endpoint, output_dir, resume=False, client=None):
    queries = load_queries(input_file)
    if urlsplit(endpoint).scheme not in ("http", "https"):
        raise ValueError("endpoint phải là URL HTTP/HTTPS tới /predict.")
    folder = Path(output_dir)
    manifest = {"input_sha256": hashlib.sha256(Path(input_file).read_bytes()).hexdigest(),
                "endpoint": endpoint, "total": len(queries)}
    recorded, results = {}, []
    if resume:
        if json.loads((folder / "input.json").read_text()) != manifest:
            raise ValueError("Input hoặc endpoint đã đổi; dùng output-dir mới.")
        for line in (folder / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["query_id"] in recorded:
                raise ValueError("Log có ID trùng; không tự resume.")
            recorded[row["query_id"]] = row
        if not set(recorded) <= {q.query_id for q in queries}:
            raise ValueError("Log chứa ID ngoài public test.")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError("Output đã có dữ liệu; dùng --resume hoặc output-dir mới.")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "input.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    owned = client is None
    client = client or httpx.Client(timeout=60)
    failures = []
    try:
        with (folder / "attempts.jsonl").open("a", encoding="utf-8") as log:
            for index, query in enumerate(queries, start=1):
                row = recorded.get(query.query_id)
                if row is None:
                    start = time.perf_counter()
                    row = {"query_id": query.query_id}
                    try:
                        response = client.post(endpoint, json=query.model_dump())
                        response.raise_for_status()
                        result = validate_response(response.json(), query.query_id)
                        elapsed = time.perf_counter() - start
                        if elapsed > 60:
                            raise ValueError("Response vượt 60 giây.")
                        row.update(status="ok", prediction=result)
                    except (httpx.HTTPError, ValueError) as error:
                        row.update(status="error", error=str(error))
                    row["seconds"] = time.perf_counter() - start
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                if row["status"] == "ok":
                    results.append(validate_response([row["prediction"]], query.query_id))
                else:
                    failures.append(row)
                print(f"Public test {index}/{len(queries)} | {query.query_id} | {row['status']}", flush=True)
    finally:
        if owned:
            client.close()
        # Xuất cả khi Ctrl+C để giữ những prediction đã nhận.
        (folder / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {**manifest, "successful": len(results), "failed": len(failures),
               "complete": len(results) == len(queries), "failures": failures}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Xuất prediction qua API đang serve")
    parser.add_argument("--input", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/predict")
    parser.add_argument("--output-dir", default="outputs/public_test")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_predictions(args.input, args.endpoint, args.output_dir, args.resume)
    print(f"Kết quả: {result['successful']}/{result['total']} | complete={result['complete']}")
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
