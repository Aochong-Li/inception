set -ex

# TODO: before running this script, run the following command to filter the questions
# bash scripts/filter_questions.sh

export CUDA_VISIBLE_DEVICES=0
MODELS_YAML="config/local_models.yaml"
EVAL_DIR="./results/countdown"
QUESTION_IDS_FNAME="stress_test_problems.json"

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    MAX_TOKENS=12288

    echo "Running Corrupt Numbers Stress Test:"
    echo "  Model: ${model_name}, ${nick_name}"
    echo "  Evaluation Directory: ${EVAL_DIR}"
    echo "-------------------------------------"

    python stress-test/corrupt_numbers.py \
        --task "countdown" \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --results_dir "${EVAL_DIR}" \
        --question_ids_fname "${QUESTION_IDS_FNAME}" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.75 \
        --dtype bfloat16 \
        --max_tokens $MAX_TOKENS \
        --temperature 0.6 \
        --top_p 1.0 \
        --top_k -1 \
        --granularity 30 \
        --unit 0.25 \
done