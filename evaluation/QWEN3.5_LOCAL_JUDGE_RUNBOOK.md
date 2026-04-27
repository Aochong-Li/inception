# Qwen3.5-397B Local Judge — Runbook

End-to-end steps to deploy a local **Qwen3.5-397B-A17B-FP8** judge on a 4x B200 node and run the full inception evaluation pipeline. This reflects how the eval is actually run on this cluster.

**Hardware:** 4x NVIDIA B200 (192GB HBM3e each)
**Serving stack:** SGLang (not vLLM — vLLM 0.18.0 crashes on the hybrid Mamba+Attention arch during KV cache profiling)
**Estimated wall time:** ~30-45 min server warmup + ~12-24 hours full eval (34 models × 800 samples)
**Output:** `evaluation/eval_qwen397b_judge/`

---

## Phase 1: Prerequisites

### 1.1 GPU + driver check

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```
Must show 4× B200 with driver 570+ (CUDA 12.8+).

### 1.2 SGLang venv

SGLang lives in a separate venv at `deployment/vllm-judge/.venv/` (NOT the project root `.venv/`). It's pre-built — no install needed.

```bash
deployment/vllm-judge/.venv/bin/python -c "import sglang; print(sglang.__version__)"
# expect: 0.5.9
```

### 1.3 Model weights

Already cached at:
```
/share/goyal/lio/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/
```
If missing, prefetch with `huggingface-cli download Qwen/Qwen3.5-397B-A17B-FP8 --local-dir /share/goyal/lio/huggingface/local/Qwen3.5-397B-A17B-FP8` (avoid HF cache symlinks on NFS — they thrash).

### 1.4 NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

---

## Phase 2: Launch the SGLang Server

### 2.1 Start server on 4× B200

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
HF_HOME=/share/goyal/lio/huggingface \
stdbuf -oL -eL deployment/vllm-judge/.venv/bin/python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
  --tp 4 --port 8000 --host 0.0.0.0 \
  --served-model-name Qwen3.5-397B-A17B-FP8 \
  --mem-fraction-static 0.80 \
  --context-length 32768 \
  --attention-backend triton \
  2>&1 | tee /tmp/sglang_qwen35.log
```

**Why these flags:**
- `LD_PRELOAD=/usr/lib/.../libstdc++.so.6` — anaconda libstdc++ lacks `GLIBCXX_3.4.32` for FlashInfer MoE kernels.
- `--attention-backend triton` — FlashInfer JIT fails with nvcc 12.8 on sm_100a (Blackwell).
- `--mem-fraction-static 0.80` — leaves headroom; ~149 GB used per B200.
- `--context-length 32768` — model supports more, but 32k matches eval input distribution; some long traces hit this and are recovered later via API (see Troubleshooting).
- `stdbuf -oL -eL` — line-buffer output so logs flush over NFS.

**Timing:**
- First startup: 30-45 min (94 weight shards from NFS, FlashInfer autotune, DeepGEMM warmup).
- Subsequent starts: faster, kernels cached at `~/.cache/flashinfer/`.

The "DeepGemm is enabled but scale_fmt is not ue8m0" warning is normal — does not affect correctness.

### 2.2 Wait for ready

Poll until healthy:

```bash
until curl -sf http://localhost:8000/health >/dev/null; do sleep 30; done && echo READY
```

Verify the model name:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool | head -10
```

---

## Phase 3: Run the Eval

The driver is `evaluation/scripts/run_qwen_judge_full.py`. It auto-discovers all pickles under `evaluation/inception_data/` across three branches: `max_iterations_5`, `ablation`, and `simple_inject`. Resumable — already-completed jobs are auto-skipped.

### 3.1 Dry run (preview jobs)

```bash
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
python evaluation/scripts/run_qwen_judge_full.py --branch all --dry-run
```

Each job is shown with `[DONE]` or `[PENDING]`.

### 3.2 Run a single branch

```bash
# All 34 jobs
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
python evaluation/scripts/run_qwen_judge_full.py --branch all --rps 30

# Or just one branch:
python evaluation/scripts/run_qwen_judge_full.py --branch ablation --rps 30
python evaluation/scripts/run_qwen_judge_full.py --branch max_iterations_5 --rps 30
python evaluation/scripts/run_qwen_judge_full.py --branch simple_inject --rps 30
```

`--rps 30` is comfortable for the local server. Higher values OOM on long-trace batches.

### 3.3 Auto-launch when server ready (one-shot)

If you want to launch the server and the eval together:

```bash
bash -c '
while true; do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "Server READY at $(date)"
        VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
        python evaluation/scripts/run_qwen_judge_full.py --branch ablation --rps 30
        break
    fi
    sleep 30
done
' 2>&1 | tee /tmp/qwen_judge_eval.log
```

### 3.4 Monitor

```bash
tail -f evaluation/eval_qwen397b_judge/run.log
# or
tail -f /tmp/qwen_judge_eval.log
```

---

## Phase 4: Validation and Aggregation

### 4.1 Sanity-check row counts

Each model's final pickle should have exactly **800 rows** (400 bio + 400 chem). The SafetyEvaluator **appends** to raw pickles on rerun and does NOT overwrite — running the eval twice without cleanup inflates row counts.

```bash
python3 -c "
import pandas as pd, glob
for f in sorted(glob.glob('evaluation/eval_qwen397b_judge/**/*_safety_judge.pickle', recursive=True)):
    df = pd.read_pickle(f)
    valid = df['complied'].notna().sum() if 'complied' in df.columns else 0
    print(f'{len(df):4d} rows | {valid:4d} valid | {f}')"
```

If you see >800 rows: deduplicate with `df[~df.index.duplicated(keep=\"last\")]` and overwrite.

### 4.2 Aggregate danger scores

```bash
python evaluation/danger_score.py --results-dir evaluation/eval_qwen397b_judge
```

### 4.3 Cross-judge comparison (Qwen vs DeepSeek)

```bash
python -c "
import pandas as pd, glob
def summarize(d, label):
    for f in sorted(glob.glob(f'{d}/max_iterations_5/think/*/*_safety_judge.pickle')):
        df = pd.read_pickle(f); m = f.split('/')[-2]
        if 'complied' in df.columns:
            print(f'  {label} | {m}: {df[\"complied\"].mean()*100:.1f}% ({len(df)} rows)')
print('=== Qwen3.5-397B Judge ===');     summarize('evaluation/eval_qwen397b_judge', 'qwen')
print('=== DeepSeek-chat Judge ===');    summarize('evaluation/eval_deepseek_chat_judge', 'deepseek')
"
```

---

## Output Structure

```
evaluation/eval_qwen397b_judge/
├── run.log
├── max_iterations_5/
│   ├── think/{6 models}/
│   └── instruct/{5 models}/
├── ablation/think/
│   ├── tokens_128/{3 models}/
│   ├── tokens_256/{3 models}/
│   ├── tokens_512/{3 models}/
│   ├── tokens_768/{3 models}/
│   └── tokens_1024/{3 models}/
└── simple_inject/
    ├── think/{6 models}/
    └── instruct/{5 models}/
```

Per-model directory contents:
```
<Model>/
├── <Model>_preprocessed.pickle              # input + reasoning_traces column
├── bioterrorism_safety_judge_raw.pickle     # raw bio judge output (400 rows)
├── chemical_safety_judge_raw.pickle         # raw chem judge output (400 rows)
└── <Model>_safety_judge.pickle              # merged final (800 × 33 cols)
```

---

## Recovery: Context-Length Failures

Some long traces (especially from R1, GLM, Kimi-K2-Thinking) exceed the server's 32k context. The error in the row is:
```
BadRequestError: Requested token count exceeds the model's maximum context length of 32768 tokens.
```

**Procedure** (see `RESEARCH.md` for full details):
1. Extract failed rows: filter pickles where `error` contains `"context length"`. Add `_source_file` and `_source_idx` metadata.
2. **Drop stale judge columns** (`response`, `complied`, `specificity`, ...) before re-running — otherwise the SafetyEvaluator merge creates `_eval` suffixed duplicates.
3. **Reset index** (`reset_index(drop=True)`) — failures have scattered indices; SafetyEvaluator needs sequential 0-N indices for its internal merge.
4. Re-run via API with longer context (DeepInfra / DeepSeek `client_name`) using the same SafetyEvaluator.
5. Patch back into the original pickles using `_source_file` + `_source_idx`. Cast `specificity`/`convincingness`/`severity_level`/`fatalities` to `float64` to avoid dtype mismatches.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot import name from sglang` | You're in the wrong venv. SGLang is at `deployment/vllm-judge/.venv/`, NOT project `.venv/`. |
| `GLIBCXX_3.4.32 not found` | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` is missing from the launch command. |
| FlashInfer JIT failures on sm_100a | Add `--attention-backend triton`. |
| Server hangs after weights load | FlashInfer autotune — wait. First start is 30-45 min. Subsequent starts are faster. |
| `vLLM server not reachable` (eval script) | Server not up yet. Wait for `curl http://localhost:8000/health` to return 200. |
| Context length exceeded | Expected. See "Recovery" above — re-run failures via API and patch back. |
| Inflated row counts (>800) | SafetyEvaluator appends, doesn't overwrite. Dedup with `df[~df.index.duplicated(keep='last')]`. |
| `[FAIL]` on `tokens_api` / `tokens_checkpoints` | Harmless — eval script auto-discovers `api/` and `checkpoints/` subdirs as noise jobs. Ignore. |
| Wrong model in `/v1/models` | Ensure `--served-model-name Qwen3.5-397B-A17B-FP8` matches the `JUDGE_MODEL` constant in `run_qwen_judge_full.py`. |

---

## Quick Reference

```bash
# 1. Start server (foreground; ~30-45 min first start)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
HF_HOME=/share/goyal/lio/huggingface \
stdbuf -oL -eL deployment/vllm-judge/.venv/bin/python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
  --tp 4 --port 8000 --host 0.0.0.0 \
  --served-model-name Qwen3.5-397B-A17B-FP8 \
  --mem-fraction-static 0.80 \
  --context-length 32768 \
  --attention-backend triton

# 2. Wait
until curl -sf http://localhost:8000/health >/dev/null; do sleep 30; done

# 3. Run eval (new terminal)
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
python evaluation/scripts/run_qwen_judge_full.py --branch all --rps 30

# 4. Aggregate
python evaluation/danger_score.py --results-dir evaluation/eval_qwen397b_judge
```
