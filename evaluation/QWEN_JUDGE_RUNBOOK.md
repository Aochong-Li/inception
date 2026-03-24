# Qwen3.5-122B Local Judge — End-to-End Runbook

Complete steps to deploy a local Qwen3.5-122B-A10B-FP8 judge on a B200 node and run the full inception evaluation pipeline. Each step is a shell command — run them sequentially.

**Hardware requirement:** 2x NVIDIA B200 GPUs (192GB HBM3e each)
**Estimated wall time:** ~2-4 hours setup + ~12-24 hours evaluation (34 models x 800 samples)
**Output:** `evaluation/eval_qwen_judge/` — same schema as `eval_deepseek_judge/`

---

## Phase 0: Prerequisites

### Step 0.1: Install uv (if not already installed)

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### Step 0.2: Verify GPU access

```bash
nvidia-smi | head -5
```

Must show 2x B200 GPUs with driver version 570+.

---

## Phase 1: Environment Setup

All paths below are relative to the project root. Set `PROJECT_ROOT` once and use it throughout:

### Step 1.1: Clone and navigate to the project

```bash
# Adjust this to wherever the repo lives on this machine
export PROJECT_ROOT="$(pwd)/inception-eval"

# Clone if needed, otherwise just cd
if [ ! -d "$PROJECT_ROOT" ]; then
    git clone https://github.com/Aochong-Li/inception.git "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"
git checkout eval && git pull origin eval
```

### Step 1.2: Configure the vLLM deployment

```bash
cp deployment/vllm-judge/.env.example deployment/vllm-judge/.env
```

Edit `.env` only if needed. Defaults are correct for the standard B200 setup. Set `HF_TOKEN` if the model is gated:

```bash
# Only if model repo requires auth:
sed -i 's/^HF_TOKEN=$/HF_TOKEN=hf_YOUR_TOKEN_HERE/' deployment/vllm-judge/.env
```

### Step 1.3: Check Apptainer availability and choose install path

```bash
bash deployment/vllm-judge/scripts/operator/check_apptainer_prerequisite.sh
```

- If this **passes**: continue with **Step 1.4a** (Apptainer path).
- If this **fails**: skip to **Step 1.4b** (bare-metal fallback). Do not substitute Docker.

### Step 1.4a: Pull the vLLM container image (Apptainer path)

This downloads the vLLM OCI image (~15-35 GB). Takes 10-30 minutes on first run.

```bash
bash deployment/vllm-judge/scripts/operator/install_gpu_prerequisites.sh
```

Skip to Step 1.5.

### Step 1.4b: Install vLLM bare-metal (fallback — no Apptainer)

If Apptainer is not available, install vLLM directly in an isolated venv. This was validated on NVIDIA Ampere (A6000) in the integration test and works identically.

**Prerequisite check:** Verify the host NVIDIA driver supports CUDA 12.8+:

```bash
nvidia-smi | head -3
```

The driver version must be 570+ for Blackwell. If the driver is too old, vLLM will fail to load — escalate to sysadmin.

**Install vLLM in an isolated venv** (separate from `rlvr_eval` to avoid dependency conflicts). This repo pins the stack to match the **`vllm/vllm-openai:cu130-nightly` build** (torch `2.10.0` + cu130 wheels per upstream `requirements/cuda.txt`, vLLM from `wheels.vllm.ai/nightly` at a locked commit — see `deployment/vllm-judge/pyproject.toml` and `agent/research/vllm-docker-cu130-nightly-pins-*.md`).

```bash
cd deployment/vllm-judge
uv sync --frozen --python 3.12
# or: make sync-bare-venv
cd ../..
```

Activate with `source deployment/vllm-judge/.venv/bin/activate` before `vllm serve` (later phase).

First sync can take several minutes (large CUDA + vLLM wheels). Re-run `uv sync --frozen` after pulling commits that change `uv.lock`.

### Step 1.5: Set up the evaluation Python environment

```bash
cd "$PROJECT_ROOT"
uv venv rlvr_eval --python 3.12 2>/dev/null || true
source rlvr_eval/bin/activate
uv pip install -r requirements.txt
uv pip install latex2sympy2==1.9.1 --no-deps
```

---

## Phase 2: Sanity Check — Small Model Dependency Validation

Before downloading the full 122B model (~140 GB), validate that vLLM, the OpenAI client, and the eval parsing pipeline all work end-to-end using a small Qwen3-8B model.

### Step 2.1: Download and serve Qwen3-8B

**Bare-metal** (if you used Step 1.4b):

```bash
source deployment/vllm-judge/.venv/bin/activate

HF_HOME=${HF_HOME:-/share/goyal/md2292/huggingface} \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --served-model-name Qwen3-8B-test \
  --host 0.0.0.0 \
  --port 8000
```

**Apptainer** (if you used Step 1.4a):

```bash
# Temporarily override MODEL in .env or pass directly:
MODEL=Qwen/Qwen3-8B SERVED_MODEL_NAME=Qwen3-8B-test TP_SIZE=1 MAX_MODEL_LEN=16384 \
  bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh
```

Wait for `Application startup complete` (~1-2 minutes for 8B).

### Step 2.2: Run the dependency sanity check

In a **new terminal**:

```bash
cd "$PROJECT_ROOT"
source rlvr_eval/bin/activate
python evaluation/scripts/sanity_check_vllm_deps.py --model Qwen3-8B-test
```

This sends a prompt asking the small model to produce output in the safety judge format (`<analysis>` + JSON), then validates parsing. Expected result: **ALL CHECKS PASSED** or **PARTIAL PASS** (small models often can't follow the exact format — that's OK, the point is validating the plumbing).

### Step 2.3: Stop the small model server

Kill the vLLM process from Step 2.1 (`Ctrl+C` or kill the Slurm job). The port must be free for the full model.

---

## Phase 3: Download and Start the Full Qwen3.5-122B Judge

### Step 3.1: Download model weights

Downloads Qwen3.5-122B-A10B-FP8 (~125-140 GB). The default path is `/share/goyal/md2292/huggingface` (hardcoded in the operator scripts as site policy). If your cluster uses a different shared storage path, set `HF_HOME` before running:

```bash
cd "$PROJECT_ROOT"

# Override if your cluster uses a different weights path:
# export HF_HOME=/your/shared/storage/huggingface

bash deployment/vllm-judge/scripts/operator/prefetch_hf_model.sh
```

**Note:** If you override `HF_HOME` here, you must also update `deployment/vllm-judge/.env` and use the same path in the bare-metal server launch (Step 3.2b).

### Step 3.2a: Launch the full model server (Apptainer path)

If you installed via Step 1.4a (Apptainer):

```bash
bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh
```

Or via Slurm:

```bash
sbatch --gres=gpu:2 --wrap="bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh"
```

The server runs in the foreground. **Open a new terminal** for the remaining steps.

### Step 3.2b: Launch the full model server (bare-metal fallback)

If you installed via Step 1.4b (no Apptainer):

```bash
source deployment/vllm-judge/.venv/bin/activate

HF_HOME=${HF_HOME:-/share/goyal/md2292/huggingface} \
VLLM_FLASH_ATTN_VERSION=2 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --served-model-name Qwen3.5-122B-A10B-FP8 \
  --host 0.0.0.0 \
  --port 8000
```

Or via Slurm:

```bash
sbatch --gres=gpu:2 --wrap="source deployment/vllm-judge/.venv/bin/activate && \
HF_HOME=${HF_HOME:-/share/goyal/md2292/huggingface} \
VLLM_FLASH_ATTN_VERSION=2 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --served-model-name Qwen3.5-122B-A10B-FP8 \
  --host 0.0.0.0 \
  --port 8000"
```

The server runs in the foreground. **Open a new terminal** for the remaining steps.

### Step 3.3: Wait for server readiness

The server logs `Application startup complete` when ready (3-10 min for 122B). Quick health check:

```bash
curl http://localhost:8000/health
```

Expected: empty 200 OK response.

### Step 3.4: Run the smoke test

```bash
cd "$PROJECT_ROOT"
source rlvr_eval/bin/activate
python deployment/vllm-judge/scripts/smoke_openai.py
```

Expected: 3/3 checks pass (health, model listing, chat completion). If running from a different node:

```bash
python deployment/vllm-judge/scripts/smoke_openai.py --base-url http://<gpu-node>:8000/v1
```

---

## Phase 4: Sanity Check — 50-Sample Eval with Full Model

Before running all 34 models, validate that the full Qwen3.5-122B judge produces valid, parseable output on a small subset (25 bio + 25 chem).

### Step 4.1: Run the 50-sample sanity check

```bash
cd "$PROJECT_ROOT"
source rlvr_eval/bin/activate
python evaluation/scripts/sanity_check_qwen_judge.py
```

If running from a different node:

```bash
python evaluation/scripts/sanity_check_qwen_judge.py --vllm-url http://<gpu-node>:8000/v1
```

This auto-selects the first available model pickle, subsamples 25 bio + 25 chem, runs the full SafetyEvaluator pipeline, and validates:
- Output pickle created with correct row count (50)
- All required columns present (`complied`, `specificity`, `convincingness`, `severity_level`, `fatalities`, `delivery_method`, `late_refusal`)
- `complied` field is populated (non-null)
- Acceptable NaN rates on parsed fields

Expected result: **ALL CHECKS PASSED**. If any check fails, inspect the output and fix before proceeding.

To keep the sanity check output for inspection:

```bash
python evaluation/scripts/sanity_check_qwen_judge.py --keep-output
# Output at: evaluation/eval_qwen_judge/_sanity_check/
```

---

## Phase 5: Full Evaluation

### Step 5.1: Dry run — preview all jobs

```bash
source rlvr_eval/bin/activate
python evaluation/scripts/run_qwen_judge_full.py --dry-run
```

This lists all 34 eval jobs with PENDING/DONE status and their input/output paths.

### Step 5.2: Run all branches

```bash
source rlvr_eval/bin/activate
python evaluation/scripts/run_qwen_judge_full.py --rps 30
```

This processes all 34 models across 3 branches:
- `max_iterations_5/think/` — 6 models
- `max_iterations_5/instruct/` — 5 models
- `ablation/think/tokens_{128,512,768,1024}/` — 3 models x 4 budgets = 12 evals
- `simple_inject/think/` — 6 models
- `simple_inject/instruct/` — 5 models (includes Qwen3-Next-80B-A3B-Instruct)

The script is fully resumable. If interrupted, re-run the same command — completed jobs are auto-skipped.

If running from a different node than the vLLM server:

```bash
python evaluation/scripts/run_qwen_judge_full.py --rps 30 --vllm-url http://<gpu-node>:8000/v1
```

### Step 5.3: Monitor progress

Logs are written to both stderr and `evaluation/eval_qwen_judge/run.log`:

```bash
tail -f evaluation/eval_qwen_judge/run.log
```

### Step 5.4: Handle failures

If any jobs fail (e.g., OOM on very long traces), the script exits with code 1 and lists failed jobs. Re-run the same command to retry only the failed/incomplete jobs.

For persistent OOM issues, try reducing max_model_len on the server:

```bash
# In deployment/vllm-judge/.env:
MAX_MODEL_LEN=32768
# Then restart the server
```

---

## Phase 6: Validation and Aggregation

### Step 6.1: Verify all jobs completed

```bash
python evaluation/scripts/run_qwen_judge_full.py --dry-run
```

All jobs should show `[DONE]`. If any show `[PENDING]`, re-run Phase 5.

### Step 6.2: Compute danger scores

```bash
python evaluation/danger_score.py --results-dir evaluation/eval_qwen_judge
```

### Step 6.3: Compare with DeepSeek judge results

```bash
python -c "
import pandas as pd, glob

def summarize(eval_dir, label):
    files = glob.glob(f'{eval_dir}/max_iterations_5/think/*/*_safety_judge.pickle')
    for f in sorted(files):
        df = pd.read_pickle(f)
        model = f.split('/')[-2]
        if 'complied' in df.columns:
            rate = df['complied'].mean() * 100
            print(f'  {label} | {model}: {rate:.1f}% compliance ({len(df)} samples)')

print('=== DeepSeek Judge ===')
summarize('evaluation/eval_deepseek_judge', 'deepseek')
print()
print('=== Qwen Judge ===')
summarize('evaluation/eval_qwen_judge', 'qwen')
"
```

---

## Expected Output Structure

```
evaluation/eval_qwen_judge/
  run.log                                          # Execution log
  max_iterations_5/
    think/
      DeepSeek-R1-0528/
        DeepSeek-R1-0528_preprocessed.pickle       # Input with reasoning_traces
        bioterrorism_safety_judge_raw.pickle        # Raw bio judge output
        chemical_safety_judge_raw.pickle            # Raw chem judge output
        DeepSeek-R1-0528_safety_judge.pickle        # Merged final (800 x 33 cols)
      DeepSeek-V3.2/...
      GLM-4.6/...
      GPT-OSS-120B/...
      Kimi-K2-Thinking/...
      Qwen3-235B-A22B-Thinking-2507/...
    instruct/
      DeepSeek-V3.2/...
      GLM-4.6/...
      Kimi-K2-Instruct-0905/...
      Qwen3-235B-A22B-Instruct-2507/...
      Qwen3-Next-80B-A3B-Instruct/...
  ablation/
    think/
      tokens_128/{DeepSeek-V3.2,GPT-OSS-120B,Qwen3-Next-80B-A3B-Thinking}/...
      tokens_512/...
      tokens_768/...
      tokens_1024/...
  simple_inject/
    think/{6 models}/...
    instruct/{5 models}/...
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `vLLM server not reachable` | Ensure server is running (Phase 3). Check `curl http://localhost:8000/health`. |
| OOM during serving | Lower `MAX_MODEL_LEN` in `.env` (try 32768). Lower `GPU_MEMORY_UTILIZATION` (try 0.88). |
| `Context length exceeded` error | Expected for some long traces — retry logic handles this automatically. |
| Slow throughput | Increase `--rps` (default 30). Check GPU utilization with `nvidia-smi`. |
| `apptainer: command not found` | Use bare-metal fallback (Steps 1.4b + 3.2b). Or run `module load apptainer-1.4.5` / check site module name. |
| `CUDA driver version insufficient` (bare-metal) | Host NVIDIA driver must be 570+ for Blackwell. Run `nvidia-smi` to check. Escalate to sysadmin if outdated. |
| `No module named 'vllm'` (bare-metal) | Ensure you activated the vLLM venv: `source deployment/vllm-judge/.venv/bin/activate`. |
| Partial completion after Ctrl+C | Safe — re-run the same command. Completed jobs are auto-skipped. |
| Wrong model in `/v1/models` | Ensure `SERVED_MODEL_NAME` in `.env` matches `JUDGE_MODEL` in the script (`Qwen3.5-122B-A10B-FP8`). |

---

## Summary of Commands (Quick Reference)

### Path A: With Apptainer

```bash
# Prerequisites (if needed)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

# Setup (one-time)
cd "$PROJECT_ROOT"
cp deployment/vllm-judge/.env.example deployment/vllm-judge/.env
bash deployment/vllm-judge/scripts/operator/check_apptainer_prerequisite.sh
bash deployment/vllm-judge/scripts/operator/install_gpu_prerequisites.sh
bash deployment/vllm-judge/scripts/operator/prefetch_hf_model.sh
source rlvr_eval/bin/activate && uv pip install -r requirements.txt && uv pip install latex2sympy2==1.9.1 --no-deps

# Server (keep running in background/tmux)
bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh

# Smoke test + eval (new terminal)
source rlvr_eval/bin/activate
python deployment/vllm-judge/scripts/smoke_openai.py
python evaluation/scripts/run_qwen_judge_full.py --dry-run
python evaluation/scripts/run_qwen_judge_full.py --rps 30
```

### Path B: Bare-metal (no Apptainer)

```bash
# Prerequisites (if needed)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

# Setup (one-time)
cd "$PROJECT_ROOT"
cp deployment/vllm-judge/.env.example deployment/vllm-judge/.env
bash deployment/vllm-judge/scripts/operator/prefetch_hf_model.sh
cd deployment/vllm-judge && uv sync --frozen --python 3.12 && cd ../..
source rlvr_eval/bin/activate && uv pip install -r requirements.txt && uv pip install latex2sympy2==1.9.1 --no-deps

# Server (keep running in background/tmux)
source deployment/vllm-judge/.venv/bin/activate
HF_HOME=${HF_HOME:-/share/goyal/md2292/huggingface} VLLM_FLASH_ATTN_VERSION=2 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --enable-prefix-caching --served-model-name Qwen3.5-122B-A10B-FP8 --host 0.0.0.0 --port 8000

# Smoke test + eval (new terminal)
source rlvr_eval/bin/activate
python deployment/vllm-judge/scripts/smoke_openai.py
python evaluation/scripts/run_qwen_judge_full.py --dry-run
python evaluation/scripts/run_qwen_judge_full.py --rps 30
```
