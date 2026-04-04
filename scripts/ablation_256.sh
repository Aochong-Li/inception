#!/bin/bash
set -ex

# 256-token architect ablation for 3 target models
# Architect (OpenThinker3-7B) loads locally on GPU via CUDA_VISIBLE_DEVICES

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
ARCHITECT_MODEL_NAME="open-thoughts/OpenThinker3-7B"
MAX_ITERATIONS=1
ARCHITECT_INITIAL_MAX_TOKENS=256

COMMON_ARGS=(
    --architect_model_name "${ARCHITECT_MODEL_NAME}"
    --dataset_name "${DATASET_NAME}"
    --split_name "test"
    --results_dir "${RESULTS_DIR}"
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.85
    --dtype bfloat16
    --max_tokens 32768
    --min_reasoning_tokens 2048
    --architect_initial_max_tokens ${ARCHITECT_INITIAL_MAX_TOKENS}
    --architect_reiterate_max_tokens 64
    --max_iterations ${MAX_ITERATIONS}
    --temperature 0.6
    --top_p 1.0
)

# --- API models (DeepSeek-V3.2 + GPT-OSS-120B) ---
# Architect runs on GPU 0, targets go through API

echo "=== DeepSeek-V3.2 (deepinfra API) ==="
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --target_model_name "deepseek-ai/DeepSeek-V3.2" \
    --target_nick_name "DeepSeek-V3.2" \
    --client_name "deepinfra" \
    "${COMMON_ARGS[@]}"

echo "=== GPT-OSS-120B (deepinfra API) ==="
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --target_model_name "openai/gpt-oss-120b" \
    --target_nick_name "GPT-OSS-120B" \
    --client_name "deepinfra" \
    "${COMMON_ARGS[@]}"

# --- Qwen3-Next-80B (local SGLang server) ---
# Requires SGLang serving on port 8000 before running this section.
# Launch separately:
#   CUDA_VISIBLE_DEVICES=0,1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
#   python -m sglang.launch_server --model-path Qwen/Qwen3-Next-80B-A3B-Thinking \
#   --tp 2 --mem-fraction-static 0.90 --port 8000 --attention-backend triton \
#   --served-model-name Qwen3-Next-80B-A3B-Thinking

echo "=== Qwen3-Next-80B-A3B-Thinking (vllm_local) ==="
CUDA_VISIBLE_DEVICES=2 VLLM_API_KEY=dummy python src/main.py \
    --target_model_name "Qwen3-Next-80B-A3B-Thinking" \
    --target_nick_name "Qwen3-Next-80B-A3B-Thinking" \
    --client_name "vllm_local" \
    "${COMMON_ARGS[@]}"

echo "=== ALL DONE ==="
