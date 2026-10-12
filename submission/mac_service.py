"""Quản lý serving/tunnel bằng LaunchAgent, không phụ thuộc phiên chat/Terminal."""
import argparse
import json
import os
import plistlib
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "Library" / "Application Support" / "ViperService"
LABELS = {"api": "io.viper.vipra", "tunnel": "io.viper.vipra.tunnel"}


def domain(label):
    return f"gui/{os.getuid()}/{label}"


def install():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if ROOT != RUNTIME:
        for name in ("baseline", "submission", "configs"):
            shutil.copytree(ROOT / name, RUNTIME / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (RUNTIME / "deploy").mkdir(exist_ok=True)
        for name in ("start_mac.sh", "stop_mac.sh"):
            shutil.copy2(ROOT / "deploy" / name, RUNTIME / "deploy" / name)
        checkpoint = RUNTIME / "deployment"
        checkpoint.mkdir(exist_ok=True)
        shutil.copytree(ROOT / "deployment" / "adapter", checkpoint / "adapter", dirs_exist_ok=True)
        serving = json.loads((ROOT / "deployment" / "serving-mac.json").read_text())
        if serving.get("backbone_path"):
            relative = Path(serving["backbone_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Runtime cần backbone_path tương đối nằm trong deployment.")
            shutil.copytree(ROOT / "deployment" / relative, checkpoint / relative, dirs_exist_ok=True)
        for name in ("serving-mac.json", "serving.json", "provenance.json", "training_run.json", "metal-environment.txt"):
            shutil.copy2(ROOT / "deployment" / name, checkpoint / name)
    python = RUNTIME / ".venv-api" / "bin" / "python"
    if not python.is_file():
        subprocess.run([str(Path.home() / ".venv-vllm-metal" / "bin" / "python"),
                        "-m", "venv", str(RUNTIME / ".venv-api")], check=True)
        subprocess.run([str(Path.home() / ".local" / "bin" / "uv"), "pip", "install",
                        "--python", str(python), "-r", str(ROOT / "requirements-serving.txt")], check=True)
    logs = RUNTIME / "deployment" / "logs-mac"
    logs.mkdir(parents=True, exist_ok=True)
    destination = Path.home() / "Library" / "LaunchAgents"
    destination.mkdir(parents=True, exist_ok=True)
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        raise RuntimeError("Cần brew install cloudflared trước.")
    commands = {
        "api": ["/bin/bash", str(RUNTIME / "deploy" / "start_mac.sh")],
        "tunnel": [cloudflared, "tunnel", "--url", "http://127.0.0.1:8001",
                   "--protocol", "http2", "--no-autoupdate", "--logfile", str(logs / "tunnel-service.log")],
    }
    for key, label in LABELS.items():
        reloaded = False
        file = destination / f"{label}.plist"
        config = {"Label": label, "ProgramArguments": commands[key], "WorkingDirectory": str(RUNTIME),
                  "RunAtLoad": False, "KeepAlive": False,
                  "StandardOutPath": str(logs / f"service-{key}.log"),
                  "StandardErrorPath": str(logs / f"service-{key}.log")}
        if file.exists() and plistlib.loads(file.read_bytes()) != config:
            old = plistlib.loads(file.read_bytes())
            if old.get("Label") != label or old.get("WorkingDirectory") not in (str(ROOT), str(RUNTIME)):
                raise FileExistsError(f"LaunchAgent đã có cấu hình khác: {file}")
            subprocess.run(["launchctl", "bootout", domain(label)], check=False, capture_output=True)
            reloaded = True
        file.write_bytes(plistlib.dumps(config))
        loaded = not reloaded and subprocess.run(["launchctl", "print", domain(label)], capture_output=True).returncode == 0
        if not loaded:
            subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(file)], check=True)
    print("Đã cài LaunchAgents. Chỉ bật khi gọi start; không tự chạy model khi đăng nhập.")


def control(action, component="all"):
    labels = list(LABELS.values()) if component == "all" else [LABELS[component]]
    if action == "start":
        for label in labels:
            subprocess.run(["launchctl", "kickstart", domain(label)], check=True)
    elif action == "stop":
        deadline = time.monotonic() + 30
        for label in reversed(labels):
            subprocess.run(["launchctl", "kill", "SIGTERM", domain(label)], check=False)
            while time.monotonic() < deadline:
                result = subprocess.run(["launchctl", "print", domain(label)], capture_output=True, text=True)
                if not re.search(r"^\s*pid = \d+", result.stdout, re.MULTILINE):
                    break
                time.sleep(.3)
            else:
                raise TimeoutError(f"Dịch vụ chưa dừng: {label}")
    elif action == "status":
        for key, label in LABELS.items():
            result = subprocess.run(["launchctl", "print", domain(label)], capture_output=True, text=True)
            lines = [line.strip() for line in result.stdout.splitlines()
                     if any(word in line for word in ("state =", "pid =", "last exit code ="))]
            print(key, "; ".join(lines) or "not loaded")
        path = RUNTIME / "deployment" / "logs-mac" / "tunnel-service.log"
        if path.is_file():
            urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", path.read_text())
            if urls:
                print("URL tạm:", urls[-1], "(đổi khi tạo tunnel mới)")


def main():
    parser = argparse.ArgumentParser(description="Bật/tắt Viper trên macOS bằng LaunchAgent")
    parser.add_argument("action", choices=("install", "start", "stop", "status"))
    parser.add_argument("--component", choices=("all", "api", "tunnel"), default="all")
    args = parser.parse_args()
    if args.action == "install":
        install()
    else:
        control(args.action, args.component)


if __name__ == "__main__":
    main()
