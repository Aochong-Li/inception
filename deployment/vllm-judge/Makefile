# vLLM Judge Service (Apptainer) — Makefile
# Convenience targets for the Apptainer-based judge service.
# NOTE: GPU targets (serve) must run on the GPU node.

.PHONY: check-apptainer install prefetch serve smoke health help

# ── Prerequisites (no GPU) ──────────────────────────────────

check-apptainer:   ## Gate: verify Apptainer is available (no GPU)
	bash scripts/operator/check_apptainer_prerequisite.sh

install:           ## Pull vLLM OCI image via Apptainer (no GPU for pull)
	bash scripts/operator/install_gpu_prerequisites.sh

prefetch:          ## Download model weights to /share/goyal/md2292/huggingface (no GPU)
	bash scripts/operator/prefetch_hf_model.sh

# ── Serving (GPU node only) ─────────────────────────────────

serve:             ## Start vLLM judge server (requires 2x B200 GPUs)
	bash scripts/operator/run_vllm_judge.sh

# ── Testing (any machine with network to server) ────────────

smoke:             ## Run smoke test against running server
	python scripts/smoke_openai.py

health:            ## Quick /health endpoint check
	@curl -sf http://localhost:$${VLLM_PORT:-8000}/health && echo "OK" || echo "UNHEALTHY"

# ── Help ────────────────────────────────────────────────────

help:              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
