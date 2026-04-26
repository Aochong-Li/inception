#!/bin/bash
# Launch a vLLM OpenAI-compatible server hosting the architect model
# (OpenThinker3-7B) on localhost:8001. Inception jobs hit this server via
# the local_architect provider (see core/openaiapi.py:PROVIDERS), so many
# inception cells can run in parallel sharing one warm GPU process.
#
# Usage:
#   bash scripts/start_architect_server.sh           # blocking foreground
#   nohup bash scripts/start_architect_server.sh \
#       > logs/architect_server.log 2>&1 &           # background
#
# Health check (returns model list when ready):
#   curl http://localhost:8001/v1/models
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"

ARCHITECT_MODEL="${ARCHITECT_MODEL:-open-thoughts/OpenThinker3-7B}"
PORT="${ARCHITECT_PORT:-8001}"
GPU_ID="${ARCHITECT_GPU:-0}"
GPU_MEM="${ARCHITECT_GPU_MEM:-0.85}"
MAX_LEN="${ARCHITECT_MAX_LEN:-32768}"

mkdir -p logs

echo "[$(date +%H:%M:%S)] starting vLLM serve: ${ARCHITECT_MODEL} on :${PORT} (GPU ${GPU_ID})"

CUDA_VISIBLE_DEVICES="${GPU_ID}" vllm serve "${ARCHITECT_MODEL}" \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_LEN}" \
    --served-model-name "${ARCHITECT_MODEL}"
