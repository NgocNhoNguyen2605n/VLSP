#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$TASK_ROOT/deployment/logs-mac/launcher.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "Không có launcher PID. Nếu đang chạy trong Terminal, dùng Ctrl+C."
  exit 0
fi
TASK_PID="$(cat "$PID_FILE")"
if [[ ! "$TASK_PID" =~ ^[0-9]+$ ]]; then
  echo "PID không hợp lệ."
  exit 1
fi
TASK_COMMAND="$(ps -p "$TASK_PID" -o command= || true)"
if [[ "$TASK_COMMAND" != *"submission.launch"* ]]; then
  echo "PID không trỏ tới submission.launch; không gửi tín hiệu."
  exit 1
fi
kill -TERM "$TASK_PID"
echo "Đã yêu cầu dừng API/model (PID $TASK_PID)."
