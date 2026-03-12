#!/usr/bin/env bash
# Run truncated-only reeval for DeepSeek judge at max_tokens=16000.
# Requires DEEPSEEK_API_KEY. Run from repo root.
set -e
cd "$(dirname "$0")/../.."
BASE="evaluation/archived/eval_five_judges/deepseek_deepseek-chat"
TARGETS="DeepSeek-V3.2 GLM-4.6 Qwen3-235B-A22B-Thinking-2507"
for target in $TARGETS; do
  prep="$BASE/$target/${target}_preprocessed.pickle"
  out="$BASE/$target"
  if [[ ! -f "$prep" ]]; then echo "Skip $target: no preprocessed pickle"; continue; fi
  echo "=== Reeval truncated-only: $target ==="
  uv run python evaluation/safety-judge.py \
    --input_filepath "$prep" \
    --output_dir "$out" \
    --nick_name "$target" \
    --eval_model deepseek-chat \
    --client_name deepseek \
    --inquiry_col inquiry \
    --reasoning_trace_col reasoning_traces \
    --category_col category \
    --chemical_label wmdp-chem \
    --bioterrorism_label wmdp-bio \
    --rate_limit 5.0 \
    --max_validation_retries 3 \
    --max_tokens 16000 \
    --reeval_only \
    --reeval_truncated_only
done
echo "=== Done ==="
