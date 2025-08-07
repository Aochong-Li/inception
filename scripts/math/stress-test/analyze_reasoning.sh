set -ex
# Use Python to extract model information from YAML
MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/math-500"
ANALYSIS_TYPE="first_answer_pos"

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

    python stress-test/analysis_tools.py \
        --model_name "$model_name" \
        --nick_name "$nick_name" \
        --results_dir $EVAL_DIR \
        --analysis_type $ANALYSIS_TYPE
done 