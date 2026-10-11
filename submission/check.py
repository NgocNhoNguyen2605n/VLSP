"""Kiểm tra endpoint công khai bằng câu probe; ghi thời gian từng request."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .predict import load_queries
from .schema import validate_response
from .config import DEFAULT_MODEL_NAME


def check(base_url, model_name, output_file, probe_file=None, client=None):
    base_url = base_url.rstrip("/")
    if urlsplit(base_url).scheme not in ("http", "https"):
        raise ValueError("base-url cần HTTP/HTTPS.")
    probes = load_queries(probe_file or Path(__file__).parent / "examples" / "public_test_sample.csv")
    owned = client is None
    client = client or httpx.Client(timeout=60)
    report = {"base_url": base_url, "model_name": model_name, "time_unix": time.time(),
              "models_ok": False, "requests": []}
    try:
        response = client.get(base_url + "/v1/models")
        response.raise_for_status()
        info = response.json()
        if info.get("object") != "list" or model_name not in {m["id"] for m in info["data"]}:
            raise ValueError("/v1/models không có đúng model đã khai báo.")
        report.update(models_ok=True, models=info)
        for probe in probes:
            start = time.perf_counter()
            record = {"query_id": probe.query_id}
            try:
                response = client.post(base_url + "/predict", json=probe.model_dump())
                response.raise_for_status()
                validate_response(response.json(), probe.query_id)
                record.update(ok=True, prediction=response.json()[0])
            except (httpx.HTTPError, ValueError) as error:
                record.update(ok=False, error=str(error))
            record["seconds"] = time.perf_counter() - start
            record["within_60_seconds"] = record["seconds"] < 60
            report["requests"].append(record)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
    finally:
        if owned:
            client.close()
        report["passed"] = (report["models_ok"] and len(report["requests"]) == len(probes)
                            and all(r["ok"] and r["within_60_seconds"] for r in report["requests"]))
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Probe API từ máy khác trước khi nộp")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--probe-file", help="CSV/JSON/JSONL chỉ lấy query_id/query; có thể dùng dev để thử")
    parser.add_argument("--output", default="outputs/preflight.json")
    args = parser.parse_args()
    report = check(args.base_url, args.model_name, args.output, args.probe_file)
    print(f"Preflight passed={report['passed']} | report={args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
