"""Chọn adapter đã xuất của experiment/sweep và tạo deployment có provenance."""

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from .config import BASE_MODEL, BASE_PARAMETERS, REVISION, ServingConfig


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run(path):
    root = Path(path).resolve()
    if (root / "sweep.json").is_file():
        sweep = json.loads((root / "sweep.json").read_text())
        if sweep["status"] != "complete" or not sweep.get("best_trial"):
            raise ValueError("Sweep chưa hoàn tất/chưa có best_trial.")
        recorded = Path(sweep["best_trial"]["run_dir"])
        root = recorded if recorded.is_dir() else root / recorded.name
    if (root / "experiment.json").is_file():
        experiment = json.loads((root / "experiment.json").read_text())
        if experiment["status"] != "complete" or experiment.get("workflow") != "train_dev":
            raise ValueError("Cần experiment train_dev đã hoàn tất và đã xuất adapter.")
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if manifest.get("training_status") != "complete":
        raise ValueError("Training chưa hoàn tất.")
    if not manifest.get("adapter_epoch"):
        raise ValueError("Dùng thư mục experiment/trial có adapter đã chọn, không dùng search/epoch-*.")
    return root, manifest


def prepare(output_dir, run_dir=None, base_only=False, team_name="Viper"):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Deployment đã có dữ liệu; chọn output-dir mới.")
    cfg = asdict(ServingConfig(team_name=team_name))
    provenance = {"team_name": team_name, "base_parameters": BASE_PARAMETERS,
                  "base_model": BASE_MODEL, "revision": REVISION, "base_only": base_only,
                  "active_llm_instances": 1}
    source = None
    if base_only:
        if run_dir:
            raise ValueError("Chỉ chọn --run-dir hoặc --base-only.")
        cfg.update(model_name=BASE_MODEL, base_only=True)
        provenance.update(adapter_parameters=0, total_parameters=BASE_PARAMETERS,
                          training_status="not_used", datasets={})
    else:
        if not run_dir:
            raise ValueError("Cần --run-dir đã train; --base-only chỉ dành cho kiểm tra backbone.")
        source, run = resolve_run(run_dir)
        training_cfg = run["config"]
        if training_cfg["model_id"] != BASE_MODEL or training_cfg["revision"] != REVISION:
            raise ValueError("Checkpoint khác backbone/revision của submission.")
        adapter = source / "adapter"
        adapter_cfg = json.loads((adapter / "adapter_config.json").read_text())
        if adapter_cfg.get("base_model_name_or_path") != BASE_MODEL or adapter_cfg.get("peft_type") != "LORA":
            raise ValueError(f"Adapter không phải LoRA của {BASE_MODEL}.")
        weights = adapter / "adapter_model.safetensors"
        from safetensors import safe_open
        with safe_open(str(weights), framework="pt", device="cpu") as tensors:
            names = list(tensors.keys())
            if not names or any("lora_" not in name for name in names):
                raise ValueError("Adapter thiếu LoRA hoặc chứa weights khác chưa khai báo.")
            count = sum(math.prod(tensors.get_slice(name).get_shape()) for name in names)
        if count != run["model"]["trainable_parameters"] or run["model"]["base_parameters"] != BASE_PARAMETERS:
            raise ValueError("Số tham số adapter/backbone không khớp manifest training.")
        cfg.update(adapter_path="adapter", max_new_tokens=training_cfg["max_new_tokens"], seed=training_cfg["seed"])
        provenance.update(adapter_parameters=count, total_parameters=BASE_PARAMETERS + count,
                          lora_rank=adapter_cfg["r"], adapter_epoch=run["adapter_epoch"],
                          training_status="complete", datasets=run["data"],
                          training_config=training_cfg, adapter_sha256=sha256(weights),
                          template_sha256=run["template_sha256"])
    if provenance["total_parameters"] > 8_000_000_000:
        raise ValueError("Tổng backbone + adapter vượt 8B.")
    # Chỉ ghi output sau khi mọi thông tin/weights được kiểm tra.
    output.mkdir(parents=True, exist_ok=True)
    if source:
        shutil.copytree(source / "adapter", output / "adapter")
        shutil.copy2(source / "run.json", output / "training_run.json")
    cfg.update(prepared=True, log_dir="logs")
    (output / "serving.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Deployment: {output} | model={cfg['model_name']} | parameters={provenance['total_parameters']:,}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị checkpoint cho vLLM")
    parser.add_argument("--run-dir", help="Thư mục experiment, trial hoặc sweep đã hoàn tất")
    parser.add_argument("--base-only", action="store_true", help="Chạy backbone gốc để kiểm tra kết nối, chưa dùng LoRA")
    parser.add_argument("--team-name", default="Viper")
    parser.add_argument("--output-dir", default="deployment")
    args = parser.parse_args()
    prepare(args.output_dir, args.run_dir, args.base_only, args.team_name)


if __name__ == "__main__":
    main()
