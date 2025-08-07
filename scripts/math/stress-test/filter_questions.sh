set -ex

MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/math-500"
QUESTION_ID="unique_id"
MIN_CORRECT=1
TEST_SIZE=200
OUTPUT_PATH="./results/math-500/questions_stress_test_min_correct=${MIN_CORRECT}_test_size=${TEST_SIZE}.json"

MODEL_LIST=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
import json
model_list = [f\"{model['model_name']},{model['nick_name']}\" for model in data['models']]
print(json.dumps(model_list))
")

python filter_questions.py \
    --model_list "$MODEL_LIST" \
    --eval_dir "$EVAL_DIR" \
    --question_id "$QUESTION_ID" \
    --min_correct "$MIN_CORRECT" \
    --test_size "$TEST_SIZE" \
    --output_path "$OUTPUT_PATH"