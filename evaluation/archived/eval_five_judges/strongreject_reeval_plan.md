# Plan: Rerun Missing DeepSeek Evals with Higher max_tokens

## Objective
Target missing/failed DeepSeek evaluations (comprehensive bio/chem pipeline), rerun with higher token limit, and merge results into `deepseek_deepseek-chat/`.

## Root Cause
Parse failures (NaN in complied/specificity/convincingness) were due to truncated JSON. DeepSeek API caps `max_tokens` at **8192** (not 16000).

## Executed (2026-03-04)
- Removed StrongREJECT max_tokens override in safety-judge.py
- Ran reeval for GLM-4.6 and Qwen3 with max_tokens=8192
- All 26 previously failed rows now complete; diagnose_parse_failures reports 0 failures

## Steps

### Step 1: Remove max_tokens override for StrongREJECT
**File:** `evaluation/safety-judge.py`
**Change:** Remove or relax the override so `max_tokens=16000` is honored when passed explicitly.
- Current: `if self.strongreject_only and max_tokens > 512: self.max_tokens = 512`
- New: Remove this block entirely, or only apply when max_tokens not explicitly set.

### Step 2: Identify missing rows
**Finding:** StrongReject standalone data is complete (0 missing). The 26 truncation failures are in the **comprehensive** (bio/chem) pipeline per `diagnose_parse_failures.py`:
- GLM-4.6: 16 rows (bioterrorism)
- Qwen3-235B-A22B-Thinking-2507: 10 rows (bioterrorism)

**Models:** DeepSeek-V3.2, GLM-4.6, Qwen3-235B-A22B-Thinking-2507

### Step 3: Run reeval for comprehensive (bio/chem) pipeline
The comprehensive pipeline produces `{Model}_safety_judge.pickle` (not strongreject). Run reeval without --strongreject_only:
```bash
cd /home/md2292/inception-eval
source rlvr_eval/bin/activate
for Model in DeepSeek-V3.2 GLM-4.6 Qwen3-235B-A22B-Thinking-2507; do
  python evaluation/safety-judge.py \
    --input_filepath evaluation/eval_five_judges/deepseek_deepseek-chat/$Model/${Model}_preprocessed.pickle \
    --output_dir evaluation/eval_five_judges/deepseek_deepseek-chat/$Model \
    --nick_name $Model \
    --eval_model deepseek-chat \
    --client_name deepseek \
    --inquiry_col inquiry \
    --reasoning_trace_col reasoning_traces \
    --category_col category \
    --chemical_label wmdp-chem \
    --bioterrorism_label wmdp-bio \
    --rate_limit 5.0 \
    --max_validation_retries 3 \
    --max_tokens 8192 \
    --reeval_only
done
```

### Step 4: Verify merge
The `run_reeval` flow in safety-judge.py:
- Identifies failed rows via `_identify_failed_rows`
- Re-evaluates only those rows
- Merges via `_apply_reeval_results` into `strongreject_safety_judge_raw.pickle`
- Regenerates `{Model}_strongreject_safety_judge.pickle` via `_merge_results`

### Step 5: Recompute cumulative strongreject_danger_scores
**Command:**
```bash
python evaluation/danger_score.py --results-dir evaluation/eval_five_judges/deepseek_deepseek-chat
```
Or use the aggregation logic from `run_strongreject_three_models.py` if danger_score expects a different structure.

### Step 6: Validate
- All 50 rows per model have non-NaN complied, specificity, convincingness
- strongreject_danger_scores.pickle updated
