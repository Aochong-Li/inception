set -ex
# Use Python to extract model information from YAML
MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/deepmath_7to9"

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    echo "Annotating reasoning: $nick_name (model_name: $model_name)"

    python utils/annotate_r.py \
        --nick_name "$nick_name" \
        --results_dir $EVAL_DIR \
        --granuality 40
done 