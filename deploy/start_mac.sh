#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
API_PYTHON="${VIPRA_API_PYTHON:-$TASK_ROOT/.venv-api/bin/python}"
METAL_VLLM="${VIPRA_METAL_VLLM:-$HOME/.venv-vllm-metal/bin/vllm}"
if [[ ! -x "$API_PYTHON" || ! -x "$METAL_VLLM" ]]; then
  echo "Chưa đủ môi trường. Xem MAC_SERVING_VI.md để cài .venv-api và vLLM Metal."
  exit 1
fi
exec /usr/bin/caffeinate -i "$API_PYTHON" -m submission.launch \
  --backend metal --vllm-bin "$METAL_VLLM" \
  --config deployment/serving-mac.json --startup-timeout 1800
