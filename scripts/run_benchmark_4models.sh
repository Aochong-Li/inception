#!/bin/bash
# Pool A — Benchmark baseline for 4 new think-mode models.
# 10 trials × 4 models = 40 trials, fully parallel (no GPU, API-only).
# Output: results/benchmark/think/trial_{0..9}/{nick}.pickle
# Stagger launches by 5s to spread initial provider load.
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
N_TRIALS=10

MODELS=(
    "deepseek-ai/DeepSeek-V4-Pro|DeepSeek-V4-Pro|deepseek"
    "deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash|deepseek_beta"
    "moonshotai/Kimi-K2.6|Kimi-K2.6|deepinfra"
    "zai-org/GLM-5.1|GLM-5.1|deepinfra"
)

run_model() {
    local model_name="$1" nick="$2" client="$3"
    echo "[$(date +%H:%M:%S)] benchmark START $nick"
    for trial in $(seq 0 $((N_TRIALS - 1))); do
        python src/benchmark.py \
            --model_name "${model_name}" \
            --nick_name "${nick}" \
            --tokenizer_name "${model_name}" \
            --dataset_name_or_path "${DATASET_NAME}" \
            --split_name "test" \
            --sample_size 10 \
            --output_dir "${RESULTS_DIR}/benchmark/think" \
            --max_tokens 32768 \
            --temperature 0.6 \
            --top_p 1.0 \
            --client_name "${client}" \
            --trial_idx "${trial}" \
            --overwrite False
    done
    echo "[$(date +%H:%M:%S)] benchmark DONE $nick"
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

# Wait for all and report
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${MODELS[$i]%%|*}"
        FAILED=$((FAILED + 1))
    fi
done

echo "=== benchmark 4 models done — failed: $FAILED ==="
exit $FAILED
