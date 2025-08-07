#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Hyperparameters
DATASET_NAME="countdown_train_stage2_level5_100K"
PASS_AT_K=1

# Paths
DATASET_PATH="./data/$DATASET_NAME"
OUTPUT_DIR="./results/$DATASET_NAME/stage1_rft"

MODEL_NAME="aochongoliverli/Qwen2.5-3B-sft-distill-countdown-level3-4-150"
NICK_NAME="Qwen2.5-3B-sft-countdown-level3-4-stage0"

python benchmark_eval.py \
    --model_name $MODEL_NAME \
    --nick_name $NICK_NAME \
    --tokenizer_name $MODEL_NAME \
    --dataset_name_or_path $DATASET_PATH \
    --split_name "test" \
    --output_dir $OUTPUT_DIR \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 4096 \
    --temperature 0.6 \
    --top_p 1.0 \
    --top_k -1 \
    --pass_at_k $PASS_AT_K \
    --overwrite True