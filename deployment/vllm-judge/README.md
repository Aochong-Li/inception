# vLLM Judge Service (Apptainer)

Serves **Qwen/Qwen3.5-122B-A10B-FP8** as an OpenAI-compatible judge model on NVIDIA Blackwell (B200) GPUs via **Apptainer**.

**Runtime:** Apptainer only (`docker://vllm/vllm-openai:cu130-nightly` pulled as SIF). **No Docker, no Docker Compose, no Podman, no bare `uv pip install vllm`** for serving.

**`uv`:** Host-side only -- prefetch weights, smoke tests, eval client. The vLLM container image ships its own PyTorch + CUDA stack.

**Weights:** All model weights **must** reside at **`/share/goyal/md2292/huggingface`** (site policy, non-negotiable).

See **`agent/plans/vllm-judge-docker-deployment.md`** for the full handoff plan.

## Roles

| Role | Responsibilities | GPU access? |
|------|-----------------|-------------|
| **Builder** | Scripts, docs, client code | No |
| **GPU Operator** | Pull image, prefetch weights, start server | Yes (2x B200) |
| **Eval User** | Run judge pipeline against `http://<host>:8000/v1` | No |

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Apptainer | `module load apptainer-1.4.5` (or site equivalent) |
| GPU | 2x NVIDIA B200 with NVLink/PCIe for NCCL |
| NVIDIA driver | Compatible with CUDA 13.0+ |
| Shared storage | Read/write on `/share/goyal/md2292` |
| HF access | `huggingface-cli login` or `HF_TOKEN` (if model is gated) |
| uv | For host-side Python (prefetch, smoke tests) |

## Quickstart

### Step 0: Check Apptainer (no GPU)

```bash
cd deployment/vllm-judge
bash scripts/operator/check_apptainer_prerequisite.sh
```

If this fails, **stop**. Do not substitute Docker. See [Failure Report](#failure-report).

Optional: save a failure report:

```bash
export APPTAINER_PREREQ_REPORT_PATH=deployment/vllm-judge/APPTAINER_PREREQ_FAILURE_REPORT.txt
bash deployment/vllm-judge/scripts/operator/check_apptainer_prerequisite.sh
```

Override module name if needed:

```bash
export APPTAINER_MODULE=apptainer-1.4.0
bash deployment/vllm-judge/scripts/operator/check_apptainer_prerequisite.sh
```

### Step 1: Configure

```bash
cp .env.example .env
# Edit .env -- at minimum set HF_TOKEN if the model is gated
```

### Step 2: Pull image + prefetch weights (can run in parallel)

```bash
# Track A: Pull vLLM OCI image (10-30+ min first time)
bash scripts/operator/install_gpu_prerequisites.sh

# Track B: Download model weights (~125-140 GB)
bash scripts/operator/prefetch_hf_model.sh
```

### Step 3: Start the server (GPU node, 2x B200)

```bash
bash scripts/operator/run_vllm_judge.sh
```

Or via Slurm:
```bash
sbatch --gres=gpu:2 --wrap="bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh"
```

### Step 4: Verify

```bash
# Quick health check
curl http://localhost:8000/health

# Full smoke test
python scripts/smoke_openai.py
# or: python scripts/smoke_openai.py --base-url http://<gpu-node>:8000/v1
```

## Scripts

| Script | Role | GPU? |
|--------|------|------|
| `scripts/operator/check_apptainer_prerequisite.sh` | Gate: verify Apptainer available | No |
| `scripts/operator/install_gpu_prerequisites.sh` | Pull vLLM OCI image | No (pull) |
| `scripts/operator/prefetch_hf_model.sh` | Download weights to `/share/goyal/md2292/huggingface` | No |
| `scripts/operator/run_vllm_judge.sh` | Launch vLLM server via `apptainer exec --nv` | **Yes** |
| `scripts/smoke_openai.py` | HTTP health + completion test | No |

## Serve Arguments

The `run_vllm_judge.sh` script runs `vllm serve` inside the Apptainer container with:

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `Qwen/Qwen3.5-122B-A10B-FP8` | Pre-quantized FP8 MoE checkpoint |
| `--tensor-parallel-size` | `2` | Shard across 2 GPUs |
| `--max-model-len` | `65536` | Max sequence length; covers >99% of judge inputs |
| `--gpu-memory-utilization` | `0.92` | Fraction of GPU HBM to use |
| `--reasoning-parser` | `qwen3` | Parse Qwen3.5 thinking tokens |
| `--enable-prefix-caching` | (flag) | Share KV cache for identical 10-shot prefixes |
| `--served-model-name` | `Qwen3.5-122B-A10B-FP8` | Name in `/v1/models` |

All values come from `.env`. Extra flags via `EXTRA_VLLM_ARGS`.

### Notes

- **FP8**: Pre-quantized. Do **not** add `--quantization fp8`.
- **`--language-model-only`**: The 122B is text-only MoE; may be a no-op. Safe to add via `EXTRA_VLLM_ARGS`.
- **Prefix caching**: Bio/chem prompts share ~6k-token prefixes across all 400 samples per domain.
- **`--max-model-len`**: Default 65536 handles even Kimi-K2 tails (~52k tokens). Lower to 32768 if OOM.

## Apptainer Details

### Image pull

```bash
module load apptainer-1.4.5
apptainer pull docker://vllm/vllm-openai:cu130-nightly
```

### Container launch (manual)

```bash
apptainer exec --nv \
  -B /share/goyal/md2292/huggingface:/root/.cache/huggingface \
  --env HF_HOME=/root/.cache/huggingface \
  --env VLLM_FLASH_ATTN_VERSION=2 \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  docker://vllm/vllm-openai:cu130-nightly \
  vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --served-model-name Qwen3.5-122B-A10B-FP8 \
    --host 0.0.0.0 \
    --port 8000
```

### Bind mount (policy-required)

```
-B /share/goyal/md2292/huggingface:/root/.cache/huggingface
```

This is the **only** allowed weights path.

## Integration with inception-eval

The eval code connects via OpenAI-compatible HTTP. No vLLM in the eval venv.

### Option A: Add provider to `core/openaiapi.py`

```python
PROVIDERS["vllm_local"] = {
    "env": "VLLM_API_KEY",
    "base_url": "http://<gpu-node>:8000/v1",
}
```

Set `VLLM_API_KEY=EMPTY` in your environment.

### Option B: Environment override

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://<gpu-node>:8000/v1
```

Then: `--client_name openai --model Qwen3.5-122B-A10B-FP8`

### Model name

Must match `--served-model-name`:
```bash
curl http://<gpu-node>:8000/v1/models | jq '.data[].id'
```

## Troubleshooting

### OOM

1. Lower `MAX_MODEL_LEN` (try 32768, then 16384)
2. Lower `GPU_MEMORY_UTILIZATION` (try 0.88)
3. Verify 2 GPUs allocated (`nvidia-smi`)
4. Check for competing GPU processes

### Slow startup

First run downloads ~60GB+ weights (if not prefetched). Even cached, model loading takes 3-10 minutes.

### Flash Attention errors

FA3 on Blackwell is unstable. Default `.env` sets `VLLM_FLASH_ATTN_VERSION=2`. Verify this is set.

### NFS performance

Weights on `/share/...` may be NFS. First load is slower than local SSD. Avoid concurrent writers on the same HF cache directory.

### Networking (GPU node != eval node)

If vLLM runs on a GPU node and eval runs on a login node, `localhost:8000` won't work. Options:
- SSH tunnel: `ssh -L 8000:<gpu-node>:8000 <gpu-node>`
- Direct: `http://<gpu-node-hostname>:8000/v1`
- Server binds `0.0.0.0` by default; restrict via firewall as needed.

### Failure Report

If `check_apptainer_prerequisite.sh` exits non-zero:

```bash
APPTAINER_PREREQ_REPORT_PATH=deployment/vllm-judge/APPTAINER_PREREQ_FAILURE_REPORT.txt \
  bash deployment/vllm-judge/scripts/operator/check_apptainer_prerequisite.sh
```

**Do not** substitute Docker or bare vLLM. Pause until Apptainer is available.

Add **`APPTAINER_PREREQ_FAILURE_REPORT.txt`** to `.gitignore` if generated locally.

## Security

Red-team / safety evaluation content in prompts. Server binds to `0.0.0.0` by default for cross-node access. Restrict via site firewall rules or SSH tunnels.

## Reproducibility

Pin the image digest after a successful deployment:

```bash
# After apptainer pull, inspect the SIF
apptainer inspect vllm-openai_cu130-nightly.sif

# Pin in .env
VLLM_IMAGE=docker://vllm/vllm-openai@sha256:<digest>
```

## Workload Profile

| Metric | Value |
|--------|-------|
| Total samples | ~8,800 (11 models x 800) |
| Total input tokens | ~144.6M |
| Total output tokens | ~10M |
| Mean input/sample | ~9.1k tokens (P95: ~12.3k) |
| Shared prefix/domain | ~6k tokens |
| Prompt families | 2 (biosecurity + chemical) |

## Storage Budget

| Component | Estimated Size |
|-----------|---------------|
| Model weights | ~125-140 GB |
| Apptainer SIF/cache | ~15-35 GB |
| uv cache | ~0.5-5 GB |

All weights at `/share/goyal/md2292/huggingface`. SIF cache at `~/.apptainer` or `$APPTAINER_CACHEDIR`.

## Make Targets

```
make help              Show all targets
make check-apptainer   Verify Apptainer available
make install           Pull vLLM OCI image
make prefetch          Download model weights
make serve             Start vLLM judge server (GPU)
make smoke             Run smoke test
make health            Quick health check
```

## References

- [Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM FP8 quantization](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- [vLLM prefix caching](https://docs.vllm.ai/en/latest/examples/offline_inference/prefix_caching.html)
- Full plan: [`agent/plans/vllm-judge-docker-deployment.md`](../../agent/plans/vllm-judge-docker-deployment.md)
