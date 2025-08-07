set -ex

export CUDA_VISIBLE_DEVICES=0,1
MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/hendrycks_math/sample200"
OUTPUT_DIR="$EVAL_DIR/distract_thinking"

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    for midway in False; do # TODO: add False now only distract from the first thinking
        for inject_user_prompt in True; do # TODO: add False Now only injects in user prompt
            DATASET_PATH="$EVAL_DIR/reasoning/${nick_name}_correct.pickle"
            MAX_TOKENS=16384

            echo "Running Distraction Experiment:"
            echo "  Model: ${model_name}, ${nick_name}"
            echo "  Dataset: ${DATASET_PATH}"
            echo "  Output Directory: ${OUTPUT_DIR}"
            echo "-------------------------------------"
            
            python distract_thinking.py \
                --model_name "${model_name}" \
                --nick_name "${nick_name}" \
                --tokenizer_name "${model_name}" \
                --dataset_path "${DATASET_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --tensor_parallel_size 1 \
                --gpu_memory_utilization 0.85 \
                --dtype bfloat16 \
                --max_tokens $MAX_TOKENS \
                --temperature 0.6 \
                --top_p 1.0 \
                --top_k -1 \
                --num_distract_candidates 10 \
                --granularity 20 \
                --midway $midway \
                --inject_user_prompt $inject_user_prompt
            done
    done
done

