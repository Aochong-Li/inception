#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Hyperparameters
DATASET_NAME="countdown_ood_question"
PASS_AT_K=1

# Paths
MODELS_YAML="config/local_models.yaml"
DATASET_PATH="./data/$DATASET_NAME"
OUTPUT_DIR="./results/$DATASET_NAME/benchmark"

# Use Python to extract model information from YAML
MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    python benchmark_eval.py \
    --model_name "$model_name" \
    --nick_name "$nick_name" \
    --tokenizer_name "$model_name" \
    --dataset_name_or_path $DATASET_PATH \
    --split_name "test" \
    --output_dir $OUTPUT_DIR \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.75 \
    --dtype bfloat16 \
    --max_tokens 8192 \
    --temperature 0.6 \
    --top_p 1.0 \
    --top_k -1 \
    --pass_at_k $PASS_AT_K \
    --overwrite True
done 