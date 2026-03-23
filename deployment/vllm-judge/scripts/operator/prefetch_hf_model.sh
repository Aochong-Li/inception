#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# prefetch_hf_model.sh — Download Qwen model weights to shared storage
# ─────────────────────────────────────────────────────────────
# Downloads model weights using `uv run` + huggingface-cli.
# No GPU required. Idempotent — safe to re-run (resumes partial downloads).
#
# POLICY: weights MUST go to /share/goyal/md2292/huggingface (hardwired).
#
# Usage:
#   bash deployment/vllm-judge/scripts/operator/prefetch_hf_model.sh
#
# With HF token (for gated models):
#   HF_TOKEN=hf_xxx bash deployment/vllm-judge/scripts/operator/prefetch_hf_model.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source .env if present
if [[ -f "${DEPLOY_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${DEPLOY_ROOT}/.env"
fi

# ── Configuration ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
HF_TOKEN="${HF_TOKEN:-}"

# POLICY: hardwired weights path (non-negotiable)
export HF_HOME="/share/goyal/md2292/huggingface"

echo "=== Model Weight Prefetch ==="
echo "  Model:   ${MODEL}"
echo "  HF_HOME: ${HF_HOME}"
echo ""

# ── Validate target directory exists ───────────────────────
if [[ ! -d "/share/goyal/md2292" ]]; then
    echo "ERROR: /share/goyal/md2292 does not exist or is not mounted." >&2
    echo "This is the policy-required shared storage location." >&2
    exit 1
fi

# Create HF_HOME if it doesn't exist yet
mkdir -p "${HF_HOME}"

# ── Check uv is available ─────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

# ── Set up token if provided ──────────────────────────────
if [[ -n "${HF_TOKEN}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    echo "  HF token: provided (${#HF_TOKEN} chars)"
else
    echo "  HF token: not set (may fail for gated models)"
fi
echo ""

# ── Download model weights ─────────────────────────────────
echo "Downloading ${MODEL} to ${HF_HOME}..."
echo "This will download ~125-140 GB. It is idempotent and resumable."
echo ""

# Use uv run to execute huggingface-cli in an ephemeral environment
# This avoids polluting the project venv with huggingface_hub CLI deps
uv run --with "huggingface_hub[cli]" -- \
    huggingface-cli download "${MODEL}" \
    --local-dir-use-symlinks False

echo ""
echo "=== Prefetch complete ==="
echo ""
echo "Disk usage:"
du -sh "${HF_HOME}" 2>/dev/null || echo "  (could not measure)"
echo ""
echo "Next steps:"
echo "  1. Verify GPU prerequisites: bash ${SCRIPT_DIR}/install_gpu_prerequisites.sh"
echo "  2. Start the judge server:   bash ${SCRIPT_DIR}/run_vllm_judge.sh"
