#!/bin/bash
# Pool A — Simple-inject baseline for 4 new think-mode models.
# 10 trials × 4 models = 40 trials, fully parallel (no GPU, API-only).
# Output: results/simple_inject/think/trial_{0..9}/{nick}.pickle
# Overrides (mode / client / extra_body) applied via TARGET_MODEL_OVERRIDES in main.py.
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
N_TRIALS=10

MODELS=(
    "deepseek-ai/DeepSeek-V4-Pro|DeepSeek-V4-Pro|deepseek_beta"
    "deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash|deepseek_beta"
    "moonshotai/Kimi-K2.6|Kimi-K2.6|deepinfra"
    "zai-org/GLM-5.1|GLM-5.1|deepinfra"
)

run_model() {
    local model_name="$1" nick="$2" client="$3"
    echo "[$(date +%H:%M:%S)] simple_inject START $nick"
    for trial in $(seq 0 $((N_TRIALS - 1))); do
        python src/simple_inject.py \
            --target_model_name "${model_name}" \
            --target_nick_name "${nick}" \
            --dataset_name "${DATASET_NAME}" \
            --split_name "test" \
            --results_dir "${RESULTS_DIR}" \
            --max_tokens 32768 \
            --temperature 0.6 \
            --top_p 1.0 \
            --sample_size 10 \
            --client_name "${client}" \
            --trial_idx "${trial}"
    done
    echo "[$(date +%H:%M:%S)] simple_inject DONE $nick"
}

export -f run_model
export RESULTS_DIR DATASET_NAME N_TRIALS

PIDS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_model "$model_name" "$nick" "$client" &
    PIDS+=($!)
    sleep 5
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${MODELS[$i]%%|*}"
        FAILED=$((FAILED + 1))
    fi
done

echo "=== simple_inject 4 models done — failed: $FAILED ==="
exit $FAILED
