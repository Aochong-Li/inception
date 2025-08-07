set -ex

# TODO: before running this script, run the following command to filter the questions
# bash scripts/filter_questions.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EVAL_DIR="./results/countdown_train_stage2_level5_100K"

# Hyperparameters
DATASET_NAME="countdown_train_stage2_level5_100K"
PASS_AT_K=1
MAX_TOKENS=8192

# Paths
DATASET_PATH="./data/$DATASET_NAME"
OUTPUT_DIR="./results/$DATASET_NAME/stage2_rl_gen_seq"

MODEL_NAME="aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch"
NICK_NAME="Qwen2.5-3B-countdown-level4-5-stage1_rl"

# Loop through each model

python generate_train_data/corrupt_sequences.py \
    --task "countdown" \
    --model_name "${MODEL_NAME}" \
    --nick_name "${NICK_NAME}" \
    --tokenizer_name "${MODEL_NAME}" \
    --results_dir "${EVAL_DIR}" \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens $MAX_TOKENS \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --granularity 30 \
    --unit 0.25 \
    --max_start_pos 0.3 \
    --sample_size 5 \
    --max_num_batched_tokens 16384 \
    --overwrite \

