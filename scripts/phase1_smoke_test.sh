#!/bin/bash
# Phase 1 smoke test for the 4 new target models (DeepSeek-V4-Pro,
# DeepSeek-V4-Flash, Kimi-K2.6, GLM-5.1) per agent/parallel_run_plan.md §4.
# Each model runs `src/main.py` with --sample_size 2 --max_iterations 1
# --architect_initial_max_tokens 128, writing under ./results_smoke/.
#
# Pass criteria (validated by inspection after each run):
#   - Output pickle exists with 2 rows.
#   - target_iteration_0 contains a complete <think>...</think> envelope
#     (especially critical for GLM-5.1 — confirms enable_thinking=true is
#     forwarded — and DeepSeek-V4-Pro — confirms chat_completions_prefill
#     synthesizes the envelope).
#   - architect_iteration_0 / reasoning columns populated.
set -ex

# HF cache: shared /share path is read-only for this user; route to ~/.cache.
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results_smoke"
ARCHITECT_MODEL_NAME="open-thoughts/OpenThinker3-7B"

COMMON_ARGS=(
    --architect_model_name "${ARCHITECT_MODEL_NAME}"
    --dataset_name "${DATASET_NAME}"
    --split_name "test"
    --results_dir "${RESULTS_DIR}"
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.85
    --dtype bfloat16
    --max_tokens 8192
    --min_reasoning_tokens 256
    --architect_initial_max_tokens 128
    --architect_reiterate_max_tokens 64
    --max_iterations 1
    --temperature 0.6
    --top_p 1.0
    --sample_size 2
    --trial_idx 0
    --overwrite
)

# DeepSeek-V4-Pro — chat_completions_prefill mode (overrides set client=deepseek)
echo "=== smoke: DeepSeek-V4-Pro ==="
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --target_model_name "deepseek-ai/DeepSeek-V4-Pro" \
    --target_nick_name "DeepSeek-V4-Pro" \
    --client_name "deepseek" \
    "${COMMON_ARGS[@]}"

# DeepSeek-V4-Flash — completions on /beta endpoint
echo "=== smoke: DeepSeek-V4-Flash ==="
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --target_model_name "deepseek-ai/DeepSeek-V4-Flash" \
    --target_nick_name "DeepSeek-V4-Flash" \
    --client_name "deepseek_beta" \
    "${COMMON_ARGS[@]}"

# Kimi-K2.6 — completions, deepinfra
echo "=== smoke: Kimi-K2.6 ==="
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --target_model_name "moonshotai/Kimi-K2.6" \
    --target_nick_name "Kimi-K2.6" \
    --client_name "deepinfra" \
    "${COMMON_ARGS[@]}"

# GLM-5.1 — completions, deepinfra, enable_thinking via extra_body
echo "=== smoke: GLM-5.1 ==="
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --target_model_name "zai-org/GLM-5.1" \
    --target_nick_name "GLM-5.1" \
    --client_name "deepinfra" \
    "${COMMON_ARGS[@]}"

echo "=== smoke ALL DONE ==="
