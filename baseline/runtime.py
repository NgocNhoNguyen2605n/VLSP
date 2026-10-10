"""Cấu hình, seed và thông tin để đối chiếu các lần chạy."""

import argparse  # Đọc tùy chọn dòng lệnh.
import hashlib  # Ghi dấu vân tay dữ liệu và chat template.
import json  # Đọc cấu hình, lưu kết quả.
import os  # Thiết lập CUDA trước khi khởi tạo PyTorch.
import random  # Cố định bộ sinh số ngẫu nhiên Python.
from importlib.metadata import PackageNotFoundError, version  # Ghi phiên bản thư viện thực tế.
from pathlib import Path  # Xử lý đường dẫn local/Kaggle.

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def arguments(description):
    """Các tùy chọn chung; đường dẫn CLI được ưu tiên hơn file JSON."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/baseline.json")
    for name in ("data-root", "train-file", "dev-file", "output-dir"):
        parser.add_argument(f"--{name}")
    return parser


def read_config(args):
    """Nạp cấu hình và kiểm tra các giới hạn quan trọng."""
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for name in ("data_root", "train_file", "dev_file", "output_dir"):
        if getattr(args, name, None) is not None:
            cfg[name] = getattr(args, name)
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed phải là số nguyên trong [0, 2**32).")
    if cfg.get("parallel_training", "model") not in ("model", "fsdp"):
        raise ValueError("parallel_training phải là model hoặc fsdp.")
    if cfg.get("parallel_training") == "fsdp" and cfg["gpu_count"] != 2:
        raise ValueError("FSDP baseline yêu cầu gpu_count=2.")
    if not cfg.get("revision") or cfg["revision"] == "main":
        raise ValueError("Cần revision cố định để tránh model thay đổi giữa các lần chạy.")
    for name in ("max_length", "epochs", "batch_size", "gradient_accumulation_steps",
                 "loss_chunk_size", "log_every", "max_new_tokens", "timeout_seconds"):
        if cfg[name] <= 0:
            raise ValueError(f"{name} phải lớn hơn 0.")
    if cfg["rtol"] < 0 or cfg["atol"] < 0:
        raise ValueError("Dung sai không được âm.")
    if not 0 < cfg.get("fp16_init_scale", 128.0) < float("inf"):
        raise ValueError("fp16_init_scale phải dương và hữu hạn.")
    skips = cfg.get("max_consecutive_skips", 8)
    if not isinstance(skips, int) or skips <= 0:
        raise ValueError("max_consecutive_skips phải là số nguyên dương.")
    return cfg


def print_training_config(cfg, target_modules, world_size=1):
    """In cấu hình của run hiện tại; worker distributed chỉ gọi từ rank 0."""
    stage = cfg.get("training_stage", "single")
    batch = cfg["batch_size"]
    accumulation = cfg["gradient_accumulation_steps"]
    mode = "FSDP2" if world_size > 1 else "model parallel (một tiến trình)"
    batch_label = "batch_size/GPU" if world_size > 1 else "batch_size/tiến trình"
    lines = [
        f"Cấu hình training | stage={stage} | output={cfg['output_dir']}",
        f"Backbone: {cfg['model_id']} | revision={cfg['revision']}",
        f"Chế độ: {mode} | GPU={cfg['gpu_count']} | seed={cfg['seed']}",
        "Precision: backbone FP16, RMSNorm/LoRA FP32 | LoRA không lượng tử hóa | attention=SDPA math",
        f"Train: epochs={cfg['epochs']} | {batch_label}={batch} | "
        f"global batch={batch * world_size} | gradient_accumulation_steps={accumulation} | "
        f"effective batch={batch * world_size * accumulation} (nhóm đầy đủ)",
        f"Optimizer: AdamW | learning_rate={cfg['learning_rate']:g} | weight_decay={cfg['weight_decay']:g}",
        f"Scheduler: cosine | warmup_ratio={cfg['warmup_ratio']:g} | max_grad_norm={cfg['max_grad_norm']:g}",
        f"LoRA: r={cfg['lora_rank']} | alpha={cfg['lora_alpha']} | dropout={cfg['lora_dropout']:g} | "
        f"target_modules={','.join(target_modules)}",
        f"Tokens: max_length={cfg['max_length']} | loss_chunk_size={cfg['loss_chunk_size']} | "
        f"group_by_length={bool(world_size > 1 and cfg.get('group_by_length', False))}",
        f"FP16: init_scale={cfg.get('fp16_init_scale', 128.0):g} | "
        f"max_consecutive_skips={cfg.get('max_consecutive_skips', 8)} | activation_checkpointing=non-reentrant",
    ]
    if world_size > 1:
        lines.append("FSDP: backward_prefetch=disabled | decoder all-gather on demand | LoRA FP32 all-reduce sau backward")
    if stage == "search":
        lines.extend([
            f"Split: validation_fraction={cfg.get('validation_fraction', 0.1):g} | "
            f"split_seed={cfg.get('split_seed', 42)} | chọn best epoch theo validation answer+unit accuracy",
            f"Validation: greedy | max_new_tokens={cfg['max_new_tokens']} | "
            f"timeout_seconds={cfg['timeout_seconds']} | rtol={cfg['rtol']:g} | atol={cfg['atol']:g}",
        ])
    elif stage == "final":
        lines.append("Retrain: toàn bộ train + validation, backbone gốc + LoRA mới, optimizer/scheduler mới.")
    print("\n".join(lines), flush=True)


def setup_seed(seed=42):
    """Cố định RNG; ưu tiên phép toán xác định, vẫn báo kernel chưa hỗ trợ."""
    import numpy as np  # Seed cho NumPy.
    import torch  # Seed và chế độ tính toán PyTorch/CUDA.
    from transformers.utils import logging  # Giảm log tải model.

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Báo kernel chưa hỗ trợ tính xác định thay vì ngắt lần chạy.
    torch.use_deterministic_algorithms(True, warn_only=True)
    logging.set_verbosity_error()
    logging.disable_progress_bar()


def fingerprint(path):
    """Hash file để phát hiện dữ liệu đổi giữa train và evaluate."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    """Ghi JSON UTF-8 bằng file tạm, tránh kết quả bị ghi dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def environment():
    """Lưu môi trường và GPU; seed không bảo đảm giống bit giữa phần cứng khác."""
    import platform  # Phiên bản Python và hệ điều hành.
    import torch  # Phiên bản CUDA và thông tin GPU.

    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "tokenizers",
                 "huggingface-hub", "safetensors", "jinja2", "triton",
                 "flash-linear-attention", "causal-conv1d"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "determinism": "seed42; deterministic algorithms warn_only; SDPA math attention",
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}
