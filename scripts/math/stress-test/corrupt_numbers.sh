set -ex

# TODO: before running this script, run the following command to filter the questions
# bash scripts/filter_questions.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
MODELS_YAML="config/control_study.yaml"
DATASET_NAME="math500amc23"

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
    python stress-test/corrupt_numbers.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --results_dir "${RESULTS_DIR}" \
        --sample_size 250 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.9 \
        --dtype bfloat16 \
        --max_tokens 32768 \
        --mini_batch_size 640 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k -1 \
        --granularity 30 \
        --how fixed \
        --max_num_batched_tokens 8192 \
        --unit 0.2
done

    