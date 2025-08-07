set -ex

MODELS_YAML="config/local_models.yaml"
EVAL_DIR="./results/countdown"

MODEL_LIST=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
import json
model_nick_name_list = [model['nick_name'] for model in data['models']]
print(json.dumps(model_nick_name_list))
")

python utils/filter_questions.py \
    --model_list "$MODEL_LIST" \
    --eval_dir "$EVAL_DIR" \
    --task "countdown" \
    --levels 4 5 6 \
    --test_size 250