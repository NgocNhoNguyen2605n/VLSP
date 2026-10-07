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
    if cfg["seed"] != 42:
        raise ValueError("Baseline này cố định seed = 42.")
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
    # Một số phép toán DeltaNet trên CUDA chưa có bản xác định tuyệt đối.
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
            "determinism": "seed42; deterministic algorithms warn_only; eager attention"}
