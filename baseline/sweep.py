"""Chạy tuần tự các cấu hình LR/weight decay/token, không retrain."""

import csv
import io
import itertools
import json
import math
from pathlib import Path

from .data import load_data, validation_manifest
from .experiment import run_experiment
from .runtime import arguments, read_config, save_json
from .validation import checkpoint_rank


def combinations(grid):
    """Kiểm tra lưới trước khi tạo thư mục hoặc khởi chạy GPU."""
    for key in ("learning_rates", "weight_decays", "max_new_tokens"):
        values = grid.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} phải là danh sách không rỗng.")
        for value in values:
            valid = (type(value) is int and value > 0) if key == "max_new_tokens" else (
                type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if key == "learning_rates" else value >= 0))
            if not valid:
                raise ValueError(f"{key}: giá trị không hợp lệ {value!r}.")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} có giá trị trùng.")
    return [{"learning_rate": lr, "weight_decay": wd, "max_new_tokens": tokens}
            for lr, wd, tokens in itertools.product(grid["learning_rates"], grid["weight_decays"],
                                                    grid["max_new_tokens"])]


def save_results(output, manifest):
    """Lưu kết quả sau mỗi trial, kể cả trial lỗi để giữ kết quả đã hoàn tất."""
    completed = [trial for trial in manifest["trials"] if trial["status"] == "complete"]
    ranked = sorted(completed, key=lambda trial: checkpoint_rank(trial["best"]), reverse=True)
    manifest["best_trial"] = ranked[0] if ranked else None
    save_json(output / "sweep.json", manifest)
    columns = ["trial", "status", "learning_rate", "weight_decay", "max_new_tokens", "best_epoch",
               "accuracy", "content_tolerant", "content_exact", "total", "validation_loss",
               "valid_json", "unfinished_generation", "adapter", "run_dir", "error"]
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()
    for trial in ranked + [t for t in manifest["trials"] if t["status"] != "complete"]:
        best = trial.get("best", {})
        writer.writerow({"trial": trial["name"], "status": trial["status"], **trial["parameters"],
                         "best_epoch": best.get("epoch", ""), "accuracy": best.get("accuracy", ""),
                         **{key: best.get(key, "") for key in columns[7:13]},
                         "adapter": str(Path(trial["run_dir"]) / "adapter") if best else "",
                         "run_dir": trial.get("run_dir", ""), "error": trial.get("error", "")})
    temporary = output / "results.csv.tmp"
    temporary.write_text(table.getvalue(), encoding="utf-8")
    temporary.replace(output / "results.csv")


def run_sweep(cfg, grid, runner=None, resume=False):
    """Mỗi trial là một experiment đầy đủ với backbone/LoRA/optimizer mới."""
    params = combinations(grid)
    cfg = cfg.copy()
    for key in ("training_stage", "split_file", "split_seed", "validation_fraction", "search_seeds"):
        cfg.pop(key, None)
    cfg.setdefault("validation_batch_size", 4)
    output = Path(cfg["output_dir"]).resolve()
    cfg["output_dir"] = str(output)
    splits, info = load_data(cfg)
    data = validation_manifest(splits, info)
    if resume:
        manifest = json.loads((output / "sweep.json").read_text(encoding="utf-8"))
        if (manifest["config"] != cfg or manifest["grid"] != grid or manifest["data"] != data):
            raise ValueError("Cấu hình/lưới/train/dev đã đổi; không resume các trial khác dữ liệu.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"{output} đã có dữ liệu; chọn output mới hoặc dùng --resume.")
        trials = []
        for index, values in enumerate(params, start=1):
            name = (f"trial-{index:02d}_lr-{values['learning_rate']:g}_wd-{values['weight_decay']:g}"
                    f"_tokens-{values['max_new_tokens']}")
            trials.append({"name": name, "parameters": values, "status": "pending", "attempts": []})
        manifest = {"status": "running", "config": cfg, "grid": grid, "data": data, "trials": trials}
    manifest["status"] = "running"
    save_results(output, manifest)
    for index, trial in enumerate(manifest["trials"], start=1):
        if trial["status"] == "complete":
            saved = json.loads((Path(trial["run_dir"]) / "experiment.json").read_text(encoding="utf-8"))
            expected = {**cfg, **trial["parameters"], "output_dir": trial["run_dir"]}
            if (saved["status"] != "complete" or saved["config"] != expected or saved["split"] != data
                    or saved["best_checkpoint"] != trial["best"]
                    or not (Path(trial["run_dir"]) / "adapter/adapter_model.safetensors").is_file()):
                raise ValueError(f"Trial hoàn tất không còn khớp manifest: {trial['name']}.")
            print(f"Sweep {index}/{len(params)}: đã hoàn tất, bỏ qua {trial['name']}", flush=True)
            continue
        attempt = len(trial["attempts"]) + 1
        suffix = "" if attempt == 1 else f"_attempt-{attempt:02d}"
        trial["run_dir"] = str(output / (trial["name"] + suffix))
        trial["attempts"].append(trial["run_dir"])
        trial.update(status="running")
        trial.pop("error", None)
        save_results(output, manifest)
        trial_cfg = {**cfg, **trial["parameters"], "output_dir": trial["run_dir"]}
        print(f"Sweep {index}/{len(params)} | lr={trial_cfg['learning_rate']:g} | "
              f"weight_decay={trial_cfg['weight_decay']:g} | max_new_tokens={trial_cfg['max_new_tokens']}", flush=True)
        try:
            current, current_info = load_data(cfg)
            if validation_manifest(current, current_info) != data:
                raise ValueError("Train/dev CSV đã đổi giữa các trial.")
            result = run_experiment(trial_cfg, runner=runner)
            trial.update(status="complete", best=result["best_checkpoint"])
            save_results(output, manifest)
        except Exception as error:
            trial.update(status="failed", error=str(error))
            manifest["status"] = "failed"
            save_results(output, manifest)
            # Không tự tiếp tục sau NCCL abort trên một CUDA session có thể đã lỗi.
            raise
    manifest["status"] = "complete"
    save_results(output, manifest)
    best = manifest["best_trial"]
    print(f"Sweep hoàn tất: {len(params)} cấu hình. Best: {best['name']} | "
          f"epoch={best['best']['epoch']} | đúng dev={best['best']['content_tolerant']}/{best['best']['total']} | "
          f"adapter={Path(best['run_dir']) / 'adapter'}", flush=True)
    return manifest


def main():
    parser = arguments("Sweep learning rate/weight decay/max_new_tokens trên train/dev")
    parser.add_argument("--grid-config", default="configs/sweep.json")
    parser.add_argument("--resume", action="store_true", help="Bỏ qua trial hoàn tất, chạy lại trial lỗi ở thư mục mới")
    args = parser.parse_args()
    cfg = read_config(args)
    if args.output_dir is None:
        cfg["output_dir"] += "_sweep"
    grid = json.loads(Path(args.grid_config).read_text(encoding="utf-8"))
    run_sweep(cfg, grid, resume=args.resume)


if __name__ == "__main__":
    main()
