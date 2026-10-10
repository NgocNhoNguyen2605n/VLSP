"""90/10 train/validation, 5 seed, chọn checkpoint rồi train mới trên toàn bộ train."""

from .runtime import arguments, fingerprint, read_config, save_json

import json
import subprocess
import sys
from pathlib import Path

from .data import load_data, split_training_rows
from .validation import checkpoint_rank


def run_training(config_path):
    """Mỗi run là tiến trình riêng để trả VRAM trước seed tiếp theo."""
    subprocess.run([sys.executable, "-m", "baseline.train", "--config", str(config_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True)


def run_experiment(cfg, runner=None):
    """Giữ cùng holdout cho mọi seed; dev chỉ dùng để đánh giá cuối bên ngoài."""
    runner = runner or run_training
    seeds = cfg.get("search_seeds", [42, 43, 44, 45, 46])
    if (len(seeds) != 5 or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != 5):
        raise ValueError("search_seeds phải gồm đúng 5 seed nguyên khác nhau trong [0, 2**32).")
    split_seed = cfg.get("split_seed", 42)
    if type(split_seed) is not int or not 0 <= split_seed < 2**32:
        raise ValueError("split_seed phải là số nguyên trong [0, 2**32).")
    output = Path(cfg["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} đã có dữ liệu; hãy chọn --output-dir mới.")
    splits, data_info = load_data(cfg, splits=("train",))
    training, validation = split_training_rows(splits["train"], cfg.get("validation_fraction", 0.1), split_seed)
    split = {"train_csv_sha256": data_info["train"]["sha256"], "split_seed": split_seed,
             "requested_validation_fraction": cfg.get("validation_fraction", 0.1),
             "actual_validation_fraction": len(validation) / len(splits["train"]),
             "grouping": "question: NFC + collapsed whitespace; duplicate questions stay together",
             "train_ids": [row["id"] for row in training],
             "validation_ids": [row["id"] for row in validation]}
    split_path = output / "split.json"
    save_json(split_path, split)
    manifest = {"status": "running", "config": cfg, "data": data_info, "split": split,
                "search_seeds": seeds, "completed_seeds": [],
                "selection": "max validation content_tolerant accuracy, then exact, min loss, earlier epoch, smaller seed",
                "final_initialization": "fresh backbone + fresh LoRA; epochs and seed from selected checkpoint"}
    save_json(output / "experiment.json", manifest)
    candidates, best = [], None
    try:
        for index, seed in enumerate(seeds, start=1):
            run_dir = output / f"seed-{seed}"
            run_cfg = {**cfg, "seed": seed, "output_dir": str(run_dir),
                       "training_stage": "search", "split_file": str(split_path)}
            config_path = output / "configs" / f"seed-{seed}.json"
            save_json(config_path, run_cfg)
            print(f"Search {index}/5: seed={seed} | train={len(training)}, validation={len(validation)}", flush=True)
            runner(config_path)
            run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if (run_manifest.get("training_status") != "complete"
                    or run_manifest["data"]["train"]["sha256"] != split["train_csv_sha256"]):
                raise RuntimeError(f"Seed {seed} chưa hoàn tất hoặc dùng khác dữ liệu.")
            history = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
            if len(history) != cfg["epochs"]:
                raise RuntimeError(f"Seed {seed} thiếu kết quả epoch.")
            for epoch, record in enumerate(history, start=1):
                candidate = record["validation"].copy()
                if candidate["seed"] != seed or candidate["epoch"] != epoch or candidate["total"] != len(validation):
                    raise RuntimeError(f"Seed {seed}: validation không khớp seed/epoch/split.")
                checkpoint = run_dir / f"epoch-{epoch}"
                if not (checkpoint / "adapter_config.json").is_file():
                    raise RuntimeError(f"Thiếu checkpoint {checkpoint}.")
                candidate.update(checkpoint=str(checkpoint), config_file=str(config_path))
                candidates.append(candidate)
                if best is None or checkpoint_rank(candidate) > checkpoint_rank(best):
                    best = candidate.copy()
            manifest["completed_seeds"].append(seed)
            manifest["best_checkpoint"] = best
            save_json(output / "experiment.json", manifest)
            save_json(output / "selection.json", {"best": best, "candidates": candidates,
                                                  "status": "searching"})
        save_json(output / "selection.json", {"best": best, "candidates": candidates,
                                              "status": "complete"})
        if fingerprint(data_info["train"]["path"]) != split["train_csv_sha256"]:
            raise ValueError("Train CSV đã đổi trước bước train cuối.")
        final_dir = output / "final"
        final_cfg = {**cfg, "seed": best["seed"], "epochs": best["epoch"],
                     "output_dir": str(final_dir), "training_stage": "final",
                     "selected_checkpoint": best}
        final_cfg.pop("split_file", None)
        final_config = output / "configs" / "final.json"
        save_json(final_config, final_cfg)
        manifest.update(status="retraining", final_config=str(final_config), final_run_dir=str(final_dir))
        save_json(output / "experiment.json", manifest)
        print(f"Best: seed={best['seed']}, epoch={best['epoch']}, "
              f"đúng={best['content_tolerant']}/{best['total']}. "
              f"Train MỚI từ backbone gốc trên {len(splits['train'])} mẫu.", flush=True)
        runner(final_config)
        final_manifest = json.loads((final_dir / "run.json").read_text(encoding="utf-8"))
        if (final_manifest.get("training_status") != "complete"
                or final_manifest["data"]["train"]["sha256"] != split["train_csv_sha256"]
                or final_manifest.get("training_samples") != len(splits["train"])):
            raise RuntimeError("Train cuối chưa hoàn tất hoặc không dùng toàn bộ train gốc.")
        manifest["status"] = "complete"
        save_json(output / "experiment.json", manifest)
        print(f"Đã train cuối: {final_dir / 'adapter'}. Đánh giá dev bằng --run-dir {final_dir}", flush=True)
        return manifest
    except Exception as error:
        manifest.update(status="failed", failure=str(error))
        save_json(output / "experiment.json", manifest)
        raise


def main():
    run_experiment(read_config(arguments("Search 5 seed và train cuối ViPRA").parse_args()))


if __name__ == "__main__":
    main()
