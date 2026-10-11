"""Cấu hình serving; mặc định chưa có adapter thì dừng, không âm thầm chạy base model."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
BASE_PARAMETERS = 3_085_938_688


@dataclass(frozen=True)
class ServingConfig:
    team_name: str = "Viper"
    base_model: str = BASE_MODEL
    revision: str = REVISION
    model_name: str = "Viper-Qwen2.5-3B-LoRA"
    adapter_path: str | None = None
    backend_url: str = "http://127.0.0.1:8000"
    max_new_tokens: int = 300
    request_deadline_seconds: float = 55
    seed: int = 42
    max_model_len: int = 4096
    gpu_memory_utilization: float = .85
    log_dir: str = "outputs/serving"
    prepared: bool = False
    base_only: bool = False

    def __post_init__(self):
        if self.base_model != BASE_MODEL or self.revision != REVISION:
            raise ValueError("Serving này yêu cầu đúng backbone/revision Qwen2.5-3B của repo.")
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
        return cls(**values)

    def require_prepared(self):
        if not self.prepared:
            raise ValueError("Chưa có deployment: chạy python -m submission.prepare --run-dir ... trước.")
        if not self.base_only:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                if not self.adapter_path or not (Path(self.adapter_path) / name).is_file():
                    raise FileNotFoundError(f"Thiếu adapter đã chuẩn bị: {name}.")
