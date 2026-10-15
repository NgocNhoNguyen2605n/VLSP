"""Cấu hình serving; mặc định chưa có adapter thì dừng, không âm thầm chạy base model."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from baseline.consistency import decoding_settings

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE_PARAMETERS = 4_022_468_096
DEFAULT_MODEL_NAME = "Viper-Qwen3-4B-Instruct-2507-LoRA"
SUPPORTED_BACKBONES = {
    BASE_MODEL: (REVISION, BASE_PARAMETERS),
    "Qwen/Qwen3-4B-Thinking-2507": ("768f209d9ea81521153ed38c47d515654e938aea", BASE_PARAMETERS),
    "Qwen/Qwen2.5-Math-7B-Instruct": ("ef9926d75ab1d54532f6a30dd5e760355eb9aa4d", 7_615_616_512),
    "Qwen/Qwen2.5-3B-Instruct": ("aa8e72537993ba99e69dfaafa59ed015b17504d1", 3_085_938_688),
}


@dataclass(frozen=True)
class ServingConfig:
    team_name: str = "Viper"
    base_model: str = BASE_MODEL
    revision: str = REVISION
    model_name: str = DEFAULT_MODEL_NAME
    adapter_path: str | None = None
    backend_url: str = "http://127.0.0.1:8000"
    max_new_tokens: int = 1500
    request_deadline_seconds: float = 55
    seed: int = 42
    self_consistency_samples: int = 1
    beam_width: int = 1
    sampling_temperature: float = 0.7
    sampling_top_p: float = 0.8
    max_model_len: int = 4096
    gpu_memory_utilization: float = .85
    log_dir: str = "outputs/serving"
    prepared: bool = False
    base_only: bool = False
    backbone_path: str | None = None
    backbone_precision: str = "float16"

    def __post_init__(self):
        decoding_settings(self)
        expected = SUPPORTED_BACKBONES.get(self.base_model)
        if expected is None or self.revision != expected[0]:
            raise ValueError("Backbone/revision serving chưa được khai báo hoặc không khớp checkpoint.")
        if not self.model_name.strip() or not self.team_name.strip():
            raise ValueError("team_name/model_name không được rỗng.")
        if type(self.max_new_tokens) is not int or not 0 < self.max_new_tokens < self.max_model_len:
            raise ValueError("max_new_tokens phải là số nguyên dương, nhỏ hơn max_model_len.")
        if not 0 < self.request_deadline_seconds <= 55:
            raise ValueError("Deadline API phải trong (0, 55] để chừa thời gian mạng trong giới hạn 60s.")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization phải trong (0, 1).")
        url = urlsplit(self.backend_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.path not in ("", "/"):
            raise ValueError("backend_url là URL gốc của vLLM, không kèm /v1.")
        if not self.base_only and self.model_name == self.base_model:
            raise ValueError("LoRA phải có model_name riêng để tránh gọi nhầm backbone.")
        if self.base_only and (self.model_name != self.base_model or self.adapter_path):
            raise ValueError("base_only phải dùng tên backbone và không có adapter.")
        if self.backbone_precision not in ("float16", "mlx-4bit", "mlx-8bit"):
            raise ValueError("backbone_precision cần float16, mlx-4bit hoặc mlx-8bit.")
        if self.backbone_precision in ("mlx-4bit", "mlx-8bit") and not self.backbone_path:
            raise ValueError(f"{self.backbone_precision} cần backbone_path đã chuyển đổi.")

    @classmethod
    def load(cls, path=None):
        path = Path(path or os.environ.get("VIPRA_SERVING_CONFIG", "configs/submission.json")).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__
        values = {k: v for k, v in data.items() if k in allowed}
        if values.get("adapter_path"):
            adapter = Path(values["adapter_path"])
            values["adapter_path"] = str((path.parent / adapter).resolve() if not adapter.is_absolute() else adapter)
        if values.get("log_dir"):
            logs = Path(values["log_dir"])
            values["log_dir"] = str((path.parent / logs).resolve() if not logs.is_absolute() else logs)
        if values.get("backbone_path"):
            backbone = Path(values["backbone_path"])
            values["backbone_path"] = str((path.parent / backbone).resolve() if not backbone.is_absolute() else backbone)
        return cls(**values)

    def require_prepared(self):
        if not self.prepared:
            raise ValueError("Chưa có deployment: chạy python -m submission.prepare --run-dir ... trước.")
        if not self.base_only:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                if not self.adapter_path or not (Path(self.adapter_path) / name).is_file():
                    raise FileNotFoundError(f"Thiếu adapter đã chuẩn bị: {name}.")
        if self.backbone_path:
            folder = Path(self.backbone_path)
            conversion = json.loads((folder / "conversion.json").read_text())
            if conversion.get("base_model") != self.base_model or conversion.get("revision") != self.revision:
                raise ValueError("Backbone chuyển đổi khác model/revision của checkpoint.")
            config = json.loads((folder / "config.json").read_text())
            if self.backbone_precision in ("mlx-4bit", "mlx-8bit"):
                expected_bits = int(self.backbone_precision.split("-")[1].removesuffix("bit"))
                actual_bits = config.get("quantization", {}).get("bits")
                if actual_bits != expected_bits:
                    raise ValueError(f"Backbone cần metadata MLX {expected_bits}-bit, nhận {actual_bits}.")
