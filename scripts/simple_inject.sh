set -ex
TARGET_MODELS_YAML="config/target_models.yaml"

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"

TARGET_MODELS_INFO=$(python -c "
import yaml
with open('$TARGET_MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['think_models']:
    print(f\"{model['model_name']},{model['nick_name']},{model['client_name']}\")
")

echo "$TARGET_MODELS_INFO" | while IFS=, read -r TARGET_MODEL_NAME TARGET_NICK_NAME CLIENT_NAME; do
    python src/simple_inject.py \
    --target_model_name "${TARGET_MODEL_NAME}" \
    --target_nick_name "${TARGET_NICK_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --split_name "test" \
    --results_dir "${RESULTS_DIR}" \
    --max_tokens 32768 \
    --temperature 0.6 \
    --top_p 1.0 \
    --client_name "${CLIENT_NAME}"
done
