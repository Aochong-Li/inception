set -ex

# TODO: before running this script, run the following command to filter the questions
# bash scripts/filter_questions.sh
export CUDA_VISIBLE_DEVICES=0,1
MODELS_YAML="config/market_models.yaml"
DATASET_NAME="allmath"

RESULTS_DIR="./results/${DATASET_NAME}"

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    python stress-test/teacher_guide.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --results_dir "${RESULTS_DIR}" \
        --sample_size 100 \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9 \
        --dtype bfloat16 \
        --max_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k -1 \
        --granularity 30 \
        --max_num_batched_tokens 32768 \
        --num_responses_per_problem 8 \
        --overwrite
done