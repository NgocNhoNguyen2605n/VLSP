"""Chạy một vLLM + API trong cùng máy GPU; shutdown cả hai khi dừng launcher."""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import ServingConfig


def vllm_command(cfg, backend="cuda", executable="vllm"):
    command = [executable, "serve", cfg.base_model,
               "--revision", cfg.revision, "--tokenizer-revision", cfg.revision,
               "--served-model-name", cfg.base_model, "--host", "127.0.0.1", "--port", "8000",
               "--dtype", "float16", "--tensor-parallel-size", "1",
               "--max-model-len", str(cfg.max_model_len), "--max-num-seqs", "1",
               "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
               "--generation-config", "vllm", "--seed", str(cfg.seed)]
    if backend == "metal":
        # Một request tại một thời điểm chỉ cần cache nhỏ; tránh chiếm RAM của macOS.
        command.extend(["--kv-cache-memory-bytes", "536870912", "--enforce-eager"])
    if not cfg.base_only:
        adapter_cfg = json.loads((Path(cfg.adapter_path) / "adapter_config.json").read_text())
        command.extend(["--enable-lora", "--max-loras", "1", "--max-lora-rank", str(adapter_cfg["r"]),
                        "--lora-modules", json.dumps({"name": cfg.model_name, "path": cfg.adapter_path,
                                                      "base_model_name": cfg.base_model})])
    return command


def main():
    parser = argparse.ArgumentParser(description="Chạy vLLM + /predict cho Viper")
    parser.add_argument("--config", default="deployment/serving.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--backend", choices=("auto", "cuda", "metal"), default="auto")
    parser.add_argument("--vllm-bin", help="Đường dẫn CLI vLLM trong môi trường serving riêng")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8001)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    cfg = ServingConfig.load(args.config)
    cfg.require_prepared()
    backend = ("metal" if sys.platform == "darwin" else "cuda") if args.backend == "auto" else args.backend
    executable = args.vllm_bin or shutil.which("vllm")
    if executable is None and backend == "metal":
        candidate = Path.home() / ".venv-vllm-metal" / "bin" / "vllm"
        if candidate.is_file():
            executable = str(candidate)
    executable = executable or "vllm"
    command = vllm_command(cfg, backend, executable)
    if args.print_command:
        print(shlex.join(command))
        return
    if shutil.which(executable) is None:
        raise RuntimeError("Chưa cài vLLM; cung cấp --vllm-bin hoặc kích hoạt môi trường serving.")
    if backend == "cuda" and (sys.platform != "linux" or shutil.which("nvidia-smi") is None):
        raise RuntimeError("CUDA cần Linux NVIDIA GPU; Mac Apple Silicon dùng --backend metal.")
    if backend == "metal":
        import platform
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("Metal cần macOS Apple Silicon và Python arm64.")
    if cfg.backend_url.rstrip("/") != "http://127.0.0.1:8000":
        raise ValueError("Launcher tích hợp dùng backend_url=http://127.0.0.1:8000.")
    env = os.environ.copy()
    env.update(VIPRA_SERVING_CONFIG=str(Path(args.config).resolve()),
               PYTHONHASHSEED=str(cfg.seed), TOKENIZERS_PARALLELISM="false")
    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        env["VLLM_MLX_DEVICE"] = "gpu"
    processes, handles = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    logs = Path(cfg.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    pid_file = logs / "launcher.pid"
    if pid_file.exists():
        previous = int(pid_file.read_text().strip())
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Launcher đang chạy (PID {previous}); dừng trước khi bật lại.")
    pid_file.write_text(str(os.getpid()), encoding="ascii")
    try:
        for name, cmd in [("vllm", command), ("api", [sys.executable, "-m", "submission.app",
                "--config", str(Path(args.config).resolve()), "--host", args.api_host,
                "--port", str(args.api_port)])]:
            handle = (logs / f"{name}.log").open("a", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            if name == "vllm":
                deadline = time.monotonic() + args.startup_timeout
                with httpx.Client(timeout=3, trust_env=False) as client:
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(f"vLLM đã dừng; xem {logs / 'vllm.log'}")
                        try:
                            response = client.get(cfg.backend_url + "/v1/models")
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(2)
                    else:
                        raise TimeoutError(f"vLLM chưa sẵn sàng; xem {logs / 'vllm.log'}")
        print(f"API: http://127.0.0.1:{args.api_port}/predict | logs={logs}", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError(f"Một tiến trình đã dừng; xem logs ở {logs}")
    except KeyboardInterrupt:
        print("Đang dừng API và vLLM...", flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()


if __name__ == "__main__":
    main()
