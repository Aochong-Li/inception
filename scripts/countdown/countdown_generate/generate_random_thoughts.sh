#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Hyperparameters
DATASET_SIZE=10000
# Paths
OUTPUT_DIR="./results/countdown_stage2"

MODEL_NAME="aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch"
NICK_NAME="Qwen2.5-3B-countdown-level4-5-stage1_rl"

python generate_train_data/generate_random_thoughts.py \
    --model_name $MODEL_NAME \
    --nick_name $NICK_NAME \
    --tokenizer_name $MODEL_NAME \
    --results_dir $OUTPUT_DIR \
    --dataset_size $DATASET_SIZE \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --max_num_batched_tokens 16384