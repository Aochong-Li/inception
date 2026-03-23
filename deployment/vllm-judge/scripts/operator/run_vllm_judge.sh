#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_vllm_judge.sh — Launch vLLM judge server via Apptainer
# ─────────────────────────────────────────────────────────────
# Requires: Apptainer, NVIDIA driver, 2x B200 GPUs, model weights
# at /share/goyal/md2292/huggingface (policy — non-negotiable).
#
# Usage (interactive):
#   bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh
#
# Usage (Slurm):
#   sbatch --gres=gpu:2 --wrap="bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh"
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source .env if present
if [[ -f "${DEPLOY_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${DEPLOY_ROOT}/.env"
fi

# ── Configuration (from .env or defaults) ──────────────────
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer-1.4.5}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:cu130-nightly}"
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-122B-A10B-FP8}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
HF_TOKEN="${HF_TOKEN:-}"
VLLM_FLASH_ATTN_VERSION="${VLLM_FLASH_ATTN_VERSION:-2}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# ── POLICY: model weights path (hardwired, non-negotiable) ──
HF_HOME_HOST="/share/goyal/md2292/huggingface"
HF_HOME_CONTAINER="/root/.cache/huggingface"

echo "=== vLLM Judge Server ==="
echo "  Model:       ${MODEL}"
echo "  TP size:     ${TP_SIZE}"
echo "  Max len:     ${MAX_MODEL_LEN}"
echo "  GPU mem:     ${GPU_MEMORY_UTILIZATION}"
echo "  Port:        ${VLLM_PORT}"
echo "  HF weights:  ${HF_HOME_HOST} -> ${HF_HOME_CONTAINER}"
echo "  Image:       ${VLLM_IMAGE}"
echo ""

# ── Step 1: Load Apptainer ─────────────────────────────────
set +u
if command -v module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE}" 2>&1 || true
fi
set -u

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not on PATH. Run check_apptainer_prerequisite.sh first." >&2
    exit 1
fi

# ── Step 2: Verify weights exist ───────────────────────────
if [[ ! -d "${HF_HOME_HOST}" ]]; then
    echo "ERROR: HF_HOME_HOST=${HF_HOME_HOST} does not exist." >&2
    echo "Run prefetch_hf_model.sh first, or check /share/goyal/md2292 mount." >&2
    exit 1
fi

# ── Step 3: Build environment vars for container ───────────
CONTAINER_ENV=(
    --env "HF_HOME=${HF_HOME_CONTAINER}"
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
    --env "HF_TOKEN=${HF_TOKEN}"
    --env "VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION}"
    --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN}"
)

# ── Step 4: Build vllm serve command ───────────────────────
VLLM_SERVE_CMD=(
    vllm serve "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --reasoning-parser "${REASONING_PARSER}"
    --enable-prefix-caching
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
)

# Append extra args if provided
if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
    # shellcheck disable=SC2206
    VLLM_SERVE_CMD+=(${EXTRA_VLLM_ARGS})
fi

echo "Command:"
echo "  apptainer exec --nv \\"
echo "    -B ${HF_HOME_HOST}:${HF_HOME_CONTAINER} \\"
echo "    ${CONTAINER_ENV[*]} \\"
echo "    ${VLLM_IMAGE} \\"
echo "    ${VLLM_SERVE_CMD[*]}"
echo ""
echo "Starting vLLM judge server..."
echo ""

# ── Step 5: Launch ─────────────────────────────────────────
exec apptainer exec \
    --nv \
    -B "${HF_HOME_HOST}:${HF_HOME_CONTAINER}" \
    "${CONTAINER_ENV[@]}" \
    "${VLLM_IMAGE}" \
    "${VLLM_SERVE_CMD[@]}"
