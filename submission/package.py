"""Đóng gói sau khi có checkpoint, URL công khai và bảng notation chính thức."""
import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from .config import ServingConfig
from .notation import read_mapping

ROOT = Path(__file__).resolve().parents[1]


def source_zip(deployment=None):
    buffer = io.BytesIO()
    paths = []
    for name in ("baseline", "submission", "configs", "deploy", "notebooks", "data_preparation"):
        directory = ROOT / name
        if directory.is_dir():
            paths.extend(directory.rglob("*"))
    paths.extend(ROOT.glob("requirements*.txt"))
    paths.extend(ROOT.glob("*.ipynb"))
    paths.extend(ROOT / name for name in ("README.md", "SUBMISSION_GUIDE_VI.md", "MAC_SERVING_VI.md", ".gitignore", ".dockerignore"))
    allowed = {".py", ".json", ".ipynb", ".md", ".txt", ".yaml", ".yml", ".sh"}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if not path.is_file() or "__pycache__" in path.parts or ".ipynb_checkpoints" in path.parts:
                continue
            if path.suffix not in allowed and path.name not in ("Dockerfile", "Caddyfile", ".env.example", ".gitignore", ".dockerignore", "public_test_sample.csv"):
                continue
            relative = path.relative_to(ROOT)
            content = path.read_bytes()
            if path.suffix == ".ipynb":
                notebook = json.loads(content)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        cell.update(outputs=[], execution_count=None)
                content = json.dumps(notebook, ensure_ascii=False, indent=1).encode()
            archive.writestr(str(relative), content)
        if deployment:
            deployment = Path(deployment)
            serving_path = deployment / "serving-mac.json"
            if not serving_path.is_file():
                serving_path = deployment / "serving.json"
            served = json.loads(serving_path.read_text())
            served.update(adapter_path="../deployment/adapter", log_dir="../deployment/logs-mac")
            if served.get("backbone_path"):
                served["backbone_path"] = "../deployment/" + Path(served["backbone_path"]).name
            archive.writestr("configs/served_checkpoint.json", json.dumps(served, ensure_ascii=False, indent=2))
            if serving_path.is_file() and json.loads(serving_path.read_text()).get("backbone_path"):
                local = Path(json.loads(serving_path.read_text())["backbone_path"])
                local = local if local.is_absolute() else deployment / local
                archive.writestr("configs/backbone_conversion.json", (local / "conversion.json").read_bytes())
            provenance = json.loads((deployment / "provenance.json").read_text())
            archive.writestr("configs/served_training.json", json.dumps({
                "base_model": provenance["base_model"], "revision": provenance["revision"],
                "base_parameters": provenance["base_parameters"],
                "adapter_parameters": provenance["adapter_parameters"],
                "adapter_sha256": provenance.get("adapter_sha256"),
                "adapter_epoch": provenance.get("adapter_epoch"),
                "training_config": provenance.get("training_config", {})}, ensure_ascii=False, indent=2))
    return buffer.getvalue()


def package(deployment, solution, notation, official_notation, base_url, preflight, output):
    from pypdf import PdfReader
    folder = Path(deployment)
    cfg = ServingConfig.load(folder / "serving.json")
    cfg.require_prepared()
    provenance = json.loads((folder / "provenance.json").read_text())
    if provenance.get("training_status") != "complete" or provenance.get("base_only"):
        raise ValueError("Hồ sơ Viper cần adapter train xong, không dùng deployment base-only.")
    if provenance["total_parameters"] > 8_000_000_000:
        raise ValueError("Tổng model vượt 8B.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.team_name):
        raise ValueError("Tên đội chỉ dùng chữ/số/_/- để đặt tên ZIP.")
    base_url = base_url.rstrip("/")
    url = urlsplit(base_url)
    if url.scheme != "https" or not url.hostname or url.path or url.query or url.fragment or url.username:
        raise ValueError("Cần URL gốc HTTPS công khai, ví dụ https://api.example.org.")
    if url.hostname in ("localhost", "127.0.0.1", "example.org", "example.com") or "<" in base_url:
        raise ValueError("URL còn là placeholder/local.")
    report = json.loads(Path(preflight).read_text())
    age = time.time() - report.get("time_unix", 0)
    if (not report.get("passed") or report.get("base_url") != base_url
            or report.get("model_name") != cfg.model_name or not 0 <= age <= 6 * 3600):
        raise ValueError("Cần preflight đạt trên đúng URL/model trong 6 giờ vừa qua.")
    rows, official = read_mapping(notation), read_mapping(official_notation)
    immutable = lambda data: [{k: v for k, v in r.items() if k != "your_notation"} for r in data]
    if immutable(rows) != immutable(official):
        raise ValueError("Notation khác danh sách/thứ tự hoặc các cột gốc của BTC.")
    reader = PdfReader(solution)
    if len(reader.pages) != 1:
        raise ValueError("solution.pdf phải đúng một trang.")
    text = reader.pages[0].extract_text() or ""
    if any(word in text for word in ("DRAFT", "PENDING", "CHUA HOAN TAT")):
        raise ValueError("PDF còn là bản nháp; cập nhật theo run thực tế trước khi nộp.")
    target = Path(output) / f"{cfg.team_name}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Đã có {target}; dùng output-dir mới.")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(solution, "solution.pdf")
        archive.writestr("source_code.zip", source_zip(folder))
        archive.writestr("urls.txt", base_url + "/predict\n" + base_url + "/v1/models\n")
        archive.write(notation, "notation_mapping.csv")
    print(target.resolve())
    return target


def main():
    parser = argparse.ArgumentParser(description="Tạo Viper.zip đúng 4 tệp ở thư mục gốc")
    parser.add_argument("--deployment", default="deployment")
    parser.add_argument("--solution", required=True)
    parser.add_argument("--notation", required=True)
    parser.add_argument("--official-notation", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--preflight", default="outputs/preflight.json")
    parser.add_argument("--output-dir", default="outputs/submission_final")
    args = parser.parse_args()
    package(args.deployment, args.solution, args.notation, args.official_notation,
            args.base_url, args.preflight, args.output_dir)


if __name__ == "__main__":
    main()
