#!/bin/bash
# Pool B — Architect-using inception for 4 new models, 6 variants, 10 trials each.
# Total: 24 cells × 10 trials = 240 architect-using trials, GPU-serialized cheapest-first.
# Each cell loads vLLM once, runs 10 trials, exits; GPU is freed between cells.
#
# Ordering (§5.2 cheapest-first):
#   ablation_128 × 4  → ablation_256 × 4 → ablation_512 × 4
#   ablation_768 × 4  → ablation_1024 × 4 → max_iter=5 × 4
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
ARCHITECT_MODEL="open-thoughts/OpenThinker3-7B"
N_TRIALS=10

MODELS=(
    "deepseek-ai/DeepSeek-V4-Pro|DeepSeek-V4-Pro|deepseek_beta"
    "deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash|deepseek_beta"
    "moonshotai/Kimi-K2.6|Kimi-K2.6|deepinfra"
    "zai-org/GLM-5.1|GLM-5.1|deepinfra"
)

COMMON_ARGS=(
    --architect_model_name "${ARCHITECT_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --split_name "test"
    --results_dir "${RESULTS_DIR}"
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.85
    --dtype bfloat16
    --max_tokens 32768
    --min_reasoning_tokens 2048
    --architect_reiterate_max_tokens 64
    --temperature 0.6
    --top_p 1.0
    --sample_size 10
)

run_cell() {
    local model_name="$1" nick="$2" client="$3" max_iter="$4" arc_tokens="$5"
    local label="${nick}|max_iter=${max_iter}|arc_tokens=${arc_tokens}"
    echo "[$(date +%H:%M:%S)] START $label"
    for trial in $(seq 0 $((N_TRIALS - 1))); do
        CUDA_VISIBLE_DEVICES=0 python src/main.py \
            --target_model_name "${model_name}" \
            --target_nick_name "${nick}" \
            --client_name "${client}" \
            --max_iterations "${max_iter}" \
            --architect_initial_max_tokens "${arc_tokens}" \
            --trial_idx "${trial}" \
            "${COMMON_ARGS[@]}"
    done
    echo "[$(date +%H:%M:%S)] DONE $label"
}

# ── Cheapest-first sweep ────────────────────────────────────────────────────────

echo "=== Phase: ablation_128 ==="
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_cell "$model_name" "$nick" "$client" 1 128
done

echo "=== Phase: ablation_256 ==="
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_cell "$model_name" "$nick" "$client" 1 256
done

echo "=== Phase: ablation_512 ==="
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_cell "$model_name" "$nick" "$client" 1 512
done

echo "=== Phase: ablation_768 ==="
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_cell "$model_name" "$nick" "$client" 1 768
done

echo "=== Phase: ablation_1024 ==="
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_cell "$model_name" "$nick" "$client" 1 1024
done

echo "=== Phase: max_iterations_5 (most expensive) ==="
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_cell "$model_name" "$nick" "$client" 5 256
done

echo "=== ALL POOL B DONE ==="
