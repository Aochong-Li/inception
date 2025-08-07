#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Path to the models YAML file
MODELS_YAML="config/market_models.yaml"
DATASET_PATH="./data/aime2425"
EVAL_DIR="./results/aime2425"
PASS_AT_K=8
OVERWRITE=false
TP=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

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
    echo "Skip Thinking Stress Test: $nick_name (model_name: $model_name)"

    python stress-test/skip_thinking.py \
        --model_name "$model_name" \
        --nick_name "$nick_name" \
        --tokenizer_name "$model_name" \
        --results_dir $EVAL_DIR \
        --tensor_parallel_size $TP \
        --gpu_memory_utilization 0.85 \
        --dtype bfloat16 \
        --max_tokens 8192 \
        --temperature 0.6 \
        --top_p 1.0 \
        --top_k -1 \
        --pass_at_k $PASS_AT_K \
        --overwrite $OVERWRITE
done 