"""Train toàn bộ train, validation trên dev, xuất checkpoint tốt nhất."""

from .runtime import arguments, fingerprint, read_config, save_json

import json
import subprocess
import shutil
import sys
from pathlib import Path

from .data import load_data, validation_manifest
from .validation import checkpoint_rank


def run_training(config_path):
    """Mỗi run là tiến trình riêng để trả VRAM trước khi nạp lại backbone."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("parallel_training") == "fsdp":
        from .parallel import launch_training
        launch_training(config_path)
        return
    subprocess.run([sys.executable, "-m", "baseline.train", "--config", str(config_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True)


def run_experiment(cfg, runner=None):
    """Một run 5 epoch; dev chọn checkpoint, không có bước retrain."""
    runner = runner or run_training
    cfg = cfg.copy()
    cfg.pop("search_seeds", None)  # Cấu hình cũ không được tạo thêm run theo seed.
    cfg.pop("split_seed", None)
    cfg.pop("validation_fraction", None)
    cfg.setdefault("validation_batch_size", 4)
    seed = cfg["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed phải là số nguyên trong [0, 2**32).")
    if type(cfg["epochs"]) is not int or cfg["epochs"] <= 0:
        raise ValueError("epochs phải là số nguyên dương.")
    output = Path(cfg["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} đã có dữ liệu; hãy chọn --output-dir mới.")
    splits, data_info = load_data(cfg)
    training, validation = splits["train"], splits["dev"]
    split = validation_manifest(splits, data_info)
    split_path = output / "split.json"
    save_json(split_path, split)
    manifest = {"status": "running", "workflow": "train_dev", "config": cfg, "data": data_info, "split": split,
                "search_seed": seed, "search_epochs": cfg["epochs"], "completed_epochs": [],
                "selection": "max validation content_tolerant accuracy, then exact, min loss, earlier epoch",
                "initialization": "fresh backbone + fresh LoRA; no retraining",
                "best_adapter_dir": str(output / "adapter")}
    save_json(output / "experiment.json", manifest)
    candidates, best = [], None
    try:
        run_dir = output / "search"
        run_cfg = {**cfg, "output_dir": str(run_dir),
                   "training_stage": "search", "split_file": str(split_path)}
        config_path = output / "configs" / "search.json"
        save_json(config_path, run_cfg)
        print(f"Train/dev: {cfg['epochs']} epoch | seed={seed} | "
              f"train={len(training)}, validation(dev)={len(validation)}", flush=True)
        runner(config_path)
        run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        recorded_config = run_manifest["config"].copy()
        recorded_config.pop("distributed_preflight", None)
        if recorded_config != run_cfg:
            raise RuntimeError("Cấu hình worker không khớp cấu hình experiment.")
        if (run_manifest.get("training_status") != "complete"
                or run_manifest.get("training_samples") != len(training)
                or run_manifest.get("split") != split
                or any(run_manifest["data"][name]["sha256"] != data_info[name]["sha256"]
                       or fingerprint(data_info[name]["path"]) != data_info[name]["sha256"]
                       for name in ("train", "dev"))):
            raise RuntimeError("Training chưa hoàn tất hoặc dùng khác dữ liệu train/dev.")
        history = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
        if len(history) != cfg["epochs"]:
            raise RuntimeError("Search thiếu kết quả epoch.")
        for epoch, record in enumerate(history, start=1):
            candidate = record["validation"].copy()
            if candidate["seed"] != seed or candidate["epoch"] != epoch or candidate["total"] != len(validation):
                raise RuntimeError("Validation không khớp seed/epoch/split.")
            checkpoint = run_dir / f"epoch-{epoch}"
            if not (checkpoint / "adapter_config.json").is_file():
                raise RuntimeError(f"Thiếu checkpoint {checkpoint}.")
            candidate.update(checkpoint=str(checkpoint), config_file=str(config_path))
            candidates.append(candidate)
            if best is None or checkpoint_rank(candidate) > checkpoint_rank(best):
                best = candidate.copy()
            manifest["completed_epochs"].append(epoch)
        manifest["best_checkpoint"] = best
        save_json(output / "experiment.json", manifest)
        save_json(output / "selection.json", {"best": best, "candidates": candidates,
                                              "status": "complete"})
        # Xuất đúng trọng số của epoch thắng, không nạp model hay train thêm.
        adapter = output / "adapter"
        shutil.copytree(best["checkpoint"], adapter)
        for path in (run_dir / "adapter").iterdir():
            if path.name.startswith("adapter_") or path.name == "README.md":
                continue
            if path.is_dir():
                shutil.copytree(path, adapter / path.name)
            else:
                shutil.copy2(path, adapter / path.name)
        save_json(output / "run.json", {**run_manifest, "adapter_epoch": best["epoch"],
                                       "adapter_checkpoint": best["checkpoint"], "best_checkpoint": best})
        manifest["status"] = "complete"
        save_json(output / "experiment.json", manifest)
        print(f"Best: epoch={best['epoch']}, đúng dev={best['content_tolerant']}/{best['total']}. "
              f"Adapter: {adapter}. Không retrain.", flush=True)
        return manifest
    except Exception as error:
        manifest.update(status="failed", failure=str(error))
        save_json(output / "experiment.json", manifest)
        raise


def main():
    parser = arguments("Train toàn bộ train; validation dev và lưu best checkpoint")
    run_experiment(read_config(parser.parse_args()))


if __name__ == "__main__":
    main()
