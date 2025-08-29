#!/bin/bash
set -ex

# -------- static bits you rarely touch --------
DATASET_NAME="allmath"
SAMPLE_K=1
DATASET_PATH="./data/${DATASET_NAME}"
OUTPUT_DIR="./results/${DATASET_NAME}/benchmark"
# ----------------------------------------------

# Define models as array of "model_name,nick_name" pairs
MODELS_NICK=(
    "openai/gpt-oss-120b,GPT-OSS-120B"
    # "deepseek-reasoner,DeepSeek-R1-0528"
    # "Qwen/Qwen3-235B-A22B,Qwen3-235B-A22B"
)
# Loop through each model
for model_info in "${MODELS_NICK[@]}"; do
    IFS=, read -r model_name nick_name <<< "$model_info"
    echo "Running model: $nick_name (model_name: $model_name)"

    python benchmark_eval.py \
    --model_name "$model_name" \
    --nick_name "$nick_name" \
    --tokenizer_name "$model_name" \
    --dataset_name_or_path $DATASET_PATH \
    --split_name "test" \
    --output_dir $OUTPUT_DIR \
    --max_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --sample_k $SAMPLE_K \
    --client_name "deepseek" \
    --overwrite True
done 