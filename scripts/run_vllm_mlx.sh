#!/usr/bin/env bash
# Launch a vLLM-mlx server for local inference on Apple Silicon.
#
# Usage:
#   ./scripts/run_vllm_mlx.sh
#   MODEL_NAME=Qwen/Qwen3-4B-MLX-4bit PORT=8001 ./scripts/run_vllm_mlx.sh
#
# Then set environment variables for RDE:
#   export LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
#   export LOCAL_OPENAI_MODEL=default
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B-MLX-8bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONTINUOUS_BATCHING="${CONTINUOUS_BATCHING:-1}"
API_KEY="${API_KEY:-}"

if ! command -v vllm-mlx >/dev/null 2>&1; then
  cat <<'MSG'
[vLLM-MLX] Not found.

Install vLLM-MLX first, then re-run this script:

  pip install git+https://github.com/waybarrios/vllm-mlx.git
MSG
  exit 1
fi

args=("$MODEL_NAME" "--host" "$HOST" "--port" "$PORT")
if [[ "$CONTINUOUS_BATCHING" == "1" ]]; then
  args+=("--continuous-batching")
fi
if [[ -n "$API_KEY" ]]; then
  args+=("--api-key" "$API_KEY")
fi

echo "[vLLM-MLX] Starting server: model=$MODEL_NAME host=$HOST port=$PORT"
exec vllm-mlx serve "${args[@]}"
