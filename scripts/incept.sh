set -ex
ARCHITECT_MODELS_YAML="config/architect_models.yaml"
TARGET_MODELS_YAML="config/target_models.yaml"

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results/"

ARCHITECT_MODEL_NAME=$(python -c "
import yaml
with open('$ARCHITECT_MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
print(f\"{data['model_name']}\")
")

TARGET_MODELS_INFO=$(python -c "
import yaml
with open('$TARGET_MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']},{model['client_name']}\")
")
MAX_ITERATIONS=1

echo "$TARGET_MODELS_INFO" | while IFS=, read -r TARGET_MODEL_NAME TARGET_NICK_NAME CLIENT_NAME; do
    python src/main.py \
    --target_model_name "${TARGET_MODEL_NAME}" \
    --target_nick_name "${TARGET_NICK_NAME}" \
    --architect_model_name "${ARCHITECT_MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --split_name "test" \
    --results_dir "${RESULTS_DIR}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 32768 \
    --architect_initial_max_tokens 128 \
    --architect_reiterate_max_tokens 64 \
    --max_iterations ${MAX_ITERATIONS} \
    --temperature 0.6 \
    --top_p 1.0 \
    --sample_size 25 \
    --client_name "${CLIENT_NAME}"
    
done
