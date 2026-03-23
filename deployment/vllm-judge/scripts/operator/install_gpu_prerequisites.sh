#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# install_gpu_prerequisites.sh — Pull vLLM OCI image via Apptainer
# ─────────────────────────────────────────────────────────────
# Run on the GPU node (or any node with Apptainer + network).
# Does NOT require GPUs for the pull itself.
#
# Usage:
#   bash deployment/vllm-judge/scripts/operator/install_gpu_prerequisites.sh
#
# Optional GPU sanity check (requires --nv + NVIDIA driver):
#   VERIFY_GPU=1 bash deployment/vllm-judge/scripts/operator/install_gpu_prerequisites.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source .env if present
if [[ -f "${DEPLOY_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${DEPLOY_ROOT}/.env"
fi

APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer-1.4.5}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:cu130-nightly}"
VERIFY_GPU="${VERIFY_GPU:-0}"

echo "=== vLLM Judge — Install GPU Prerequisites ==="
echo "  Apptainer module: ${APPTAINER_MODULE}"
echo "  OCI image:        ${VLLM_IMAGE}"
echo ""

# ── Step 0: Ensure Apptainer is available ───────────────────
echo "[0/3] Checking Apptainer prerequisite..."
bash "${SCRIPT_DIR}/check_apptainer_prerequisite.sh"
echo ""

# ── Step 1: Load module ────────────────────────────────────
echo "[1/3] Loading Apptainer module..."
set +u
# shellcheck disable=SC1091
if command -v module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE}" 2>&1 || true
fi
set -u

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not on PATH after module load. Aborting." >&2
    exit 1
fi
echo "  apptainer: $(command -v apptainer)"
apptainer --version
echo ""

# ── Step 2: Pull OCI image ─────────────────────────────────
echo "[2/3] Pulling OCI image (this may take 10-30+ minutes on first run)..."
echo "  Image: ${VLLM_IMAGE}"

# apptainer pull converts docker:// to local SIF
# The SIF lands in $APPTAINER_CACHEDIR or current directory
apptainer pull "${VLLM_IMAGE}"

echo ""
echo "  Pull complete."
echo "  SIF cache location: ${APPTAINER_CACHEDIR:-~/.apptainer}"
echo ""

# ── Step 3: Optional GPU sanity check ──────────────────────
if [[ "${VERIFY_GPU}" == "1" ]]; then
    echo "[3/3] GPU sanity check (--nv + nvidia-smi)..."

    # Extract the local SIF name from the image URI
    # docker://vllm/vllm-openai:cu130-nightly -> vllm-openai_cu130-nightly.sif
    SIF_NAME="$(basename "${VLLM_IMAGE#docker://}" | tr ':' '_').sif"
    if [[ ! -f "${SIF_NAME}" ]]; then
        # Try without the org prefix
        SIF_NAME="$(echo "${VLLM_IMAGE}" | sed 's|docker://||' | tr '/:' '_-').sif"
    fi

    if [[ -f "${SIF_NAME}" ]]; then
        apptainer exec --nv "${SIF_NAME}" nvidia-smi
    else
        echo "  WARNING: Could not locate SIF file for GPU check."
        echo "  Run manually: apptainer exec --nv <sif_path> nvidia-smi"
    fi
else
    echo "[3/3] GPU sanity check skipped (set VERIFY_GPU=1 to enable)."
fi

echo ""
echo "=== Prerequisites installed successfully ==="
echo ""
echo "Next steps:"
echo "  1. Download model weights:  bash ${SCRIPT_DIR}/prefetch_hf_model.sh"
echo "  2. Start the judge server:  bash ${SCRIPT_DIR}/run_vllm_judge.sh"
