#!/bin/bash
# Benchmark baseline for 4 new think-mode models, single trial each.
# API-only (no GPU); runs the 4 models in parallel.
# Per-model max_tokens is sourced from config/target_models.yaml.
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
MODELS_YAML="config/target_models.yaml"
DEFAULT_MAX_TOKENS=32768

MODELS=(
    "deepseek-ai/DeepSeek-V4-Pro|DeepSeek-V4-Pro|deepseek_beta"
    "deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash|deepseek_beta"
    "moonshotai/Kimi-K2.6|Kimi-K2.6|deepinfra"
    "zai-org/GLM-5.1|GLM-5.1|deepinfra"
)

get_max_tokens() {
    local nick="$1"
    python -c "
import yaml
with open('${MODELS_YAML}') as f:
    data = yaml.safe_load(f)
for m in data.get('think_models', []) + data.get('instruct_models', []):
    if m.get('nick_name') == '${nick}':
        print(m.get('max_tokens', ${DEFAULT_MAX_TOKENS}))
        break
else:
    print(${DEFAULT_MAX_TOKENS})
"
}

run_model() {
    local model_name="$1" nick="$2" client="$3"
    local max_tokens
    max_tokens=$(get_max_tokens "$nick")
    echo "[$(date +%H:%M:%S)] benchmark START $nick (max_tokens=${max_tokens})"
    python src/benchmark.py \
        --model_name "${model_name}" \
        --nick_name "${nick}" \
        --tokenizer_name "${model_name}" \
        --dataset_name_or_path "${DATASET_NAME}" \
        --split_name "test" \
        --output_dir "${RESULTS_DIR}/benchmark/think" \
        --max_tokens "${max_tokens}" \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k -1 \
        --sample_k 1 \
        --client_name "${client}" \
        --overwrite False
    echo "[$(date +%H:%M:%S)] benchmark DONE $nick"
}

export -f get_max_tokens run_model
export MODELS_YAML DEFAULT_MAX_TOKENS RESULTS_DIR DATASET_NAME

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

echo "=== benchmark 4 models done — failed: $FAILED ==="
exit $FAILED
