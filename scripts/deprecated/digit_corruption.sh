set -ex

export CUDA_VISIBLE_DEVICES=0,1
MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/math-500"
OUTPUT_DIR="$EVAL_DIR/digit_corruption"

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    DATASET_PATH="$EVAL_DIR/reasoning/${nick_name}_correct.pickle"
    MAX_TOKENS=16384

    echo "Running Digit Corruption Experiment:"
    echo "  Model: ${model_name}, ${nick_name}"
    echo "  Dataset: ${DATASET_PATH}"
    echo "  Output Directory: ${OUTPUT_DIR}"
    echo "-------------------------------------"

    python digit_corruption.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.85 \
        --dtype bfloat16 \
        --max_tokens $MAX_TOKENS \
        --temperature 0.6 \
        --top_p 1.0 \
        --top_k -1 \
        --corrupt_type "answer_digit" \
    
    python digit_corruption.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.85 \
        --dtype bfloat16 \
        --max_tokens $MAX_TOKENS \
        --temperature 0.6 \
        --top_p 1.0 \
        --top_k -1 \
        --corrupt_type "midway" \
        
done