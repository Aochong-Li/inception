#!/bin/bash
# Ablation: max_iterations=1 across 5 architect_initial_max_tokens values
# (128, 256, 512, 768, 1024) for the 4 new think-mode models.
# GPU-serialized cheapest-first (smallest arc_tokens first).
# Outputs nest under results/max_iterations_1/think/architect_initial_max_tokens_{X}/{nick}.pickle
# Per-model max_tokens is sourced from config/target_models.yaml.
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
ARCHITECT_MODEL="open-thoughts/OpenThinker3-7B"
MAX_ITERATIONS=1
MODELS_YAML="config/target_models.yaml"
DEFAULT_MAX_TOKENS=32768

ARC_TOKEN_SWEEP=(128 256 512 768 1024)

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

for arc_tokens in "${ARC_TOKEN_SWEEP[@]}"; do
    echo "=== Phase: architect_initial_max_tokens=${arc_tokens} ==="
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r model_name nick client <<< "$entry"
        max_tokens=$(get_max_tokens "$nick")
        echo "[$(date +%H:%M:%S)] ablation START $nick (arc_tokens=${arc_tokens}, max_tokens=${max_tokens})"
        CUDA_VISIBLE_DEVICES=0 python src/main.py \
            --target_model_name "${model_name}" \
            --target_nick_name "${nick}" \
            --client_name "${client}" \
            --architect_model_name "${ARCHITECT_MODEL}" \
            --dataset_name "${DATASET_NAME}" \
            --split_name "test" \
            --results_dir "${RESULTS_DIR}" \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.85 \
            --dtype bfloat16 \
            --max_tokens "${max_tokens}" \
            --min_reasoning_tokens 2048 \
            --architect_initial_max_tokens "${arc_tokens}" \
            --architect_reiterate_max_tokens 64 \
            --max_iterations "${MAX_ITERATIONS}" \
            --temperature 0.6 \
            --top_p 1.0
        echo "[$(date +%H:%M:%S)] ablation DONE $nick (arc_tokens=${arc_tokens})"
    done
done

echo "=== ablation 4 models × ${#ARC_TOKEN_SWEEP[@]} arc-token cells done ==="
