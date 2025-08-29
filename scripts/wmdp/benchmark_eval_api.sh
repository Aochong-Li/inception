#!/bin/bash
set -ex

# -------- static bits you rarely touch --------
MODELS_YAML="config/target_models.yaml"
DATASET_NAME="wmdp_inquiries_300"
SPLIT="test"
SAMPLE_K=1
DATASET_PATH="aochongoliverli/wmdp_inquiries_300"
OUTPUT_DIR="./results/${DATASET_NAME}/benchmark"
# ----------------------------------------------

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    echo "Running model: $nick_name (model_name: $model_name)"

    python benchmark_eval.py \
    --model_name "$model_name" \
    --nick_name "$nick_name" \
    --tokenizer_name "$model_name" \
    --dataset_name_or_path $DATASET_PATH \
    --split_name $SPLIT \
    --output_dir $OUTPUT_DIR \
    --max_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --sample_k $SAMPLE_K \
    --client_name "deepinfra" \
    --overwrite False
done 