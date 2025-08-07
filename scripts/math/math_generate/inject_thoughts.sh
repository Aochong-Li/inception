#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Paths
OUTPUT_DIR="./results/deepmath_generate_data"

DATASET_PATH="../rlvr/data/train/deepmath_level3-4-stage2-post-step285-reward-lt-1.0/train.parquet"
MODEL_NAME="aochongoliverli/R1-Distill-Qwen-1.5B-DeepMath-level3-4-stage1-grpo-5epochs-4rollouts-8192max-length"
NICK_NAME="Qwen2.5-1.5B-DeepMath-stage1-grpo-level3-4-5epochs-4rollouts-8192max-length-5epochs"
DISTRACTOR_DATASET_PATH="./results/deepmath_generate_data/random_thoughts/Qwen2.5-1.5B-DeepMath-stage1-grpo-level3-4-5epochs-4rollouts-8192max-length-5epochs.pickle"

python generate_train_data/deepmath/inject_random_thoughts.py \
    --model_name $MODEL_NAME \
    --nick_name $NICK_NAME \
    --tokenizer_name $MODEL_NAME \
    --results_dir $OUTPUT_DIR \
    --dataset_path $DATASET_PATH \
    --distractor_dataset_path $DISTRACTOR_DATASET_PATH \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --max_cot_tokens 2048 \
    --max_num_batched_tokens 131072