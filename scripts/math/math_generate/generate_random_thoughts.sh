#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Hyperparameters
DATASET_SIZE=10000
# Paths
OUTPUT_DIR="./results/deepmath_generate_data"

MODEL_NAME="aochongoliverli/R1-Distill-Qwen-1.5B-DeepMath-level3-4-stage1-grpo-5epochs-4rollouts-8192max-length"
NICK_NAME="Qwen2.5-1.5B-DeepMath-stage1-grpo-level3-4-5epochs-4rollouts-8192max-length-5epochs"

python generate_train_data/generate_random_thoughts.py \
    --task math \
    --model_name $MODEL_NAME \
    --nick_name $NICK_NAME \
    --tokenizer_name $MODEL_NAME \
    --results_dir $OUTPUT_DIR \
    --dataset_size $DATASET_SIZE \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --max_num_batched_tokens 32768