#!/usr/bin/env bash
set -euo pipefail
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

# -------- static bits you rarely touch --------
DATASET_NAME="countdown"
PASS_AT_K=1
DATASET_PATH="./data/${DATASET_NAME}"
OUTPUT_DIR="./results/${DATASET_NAME}/benchmark"
# ----------------------------------------------

export CUDA_VISIBLE_DEVICES=$GPU_ID

python benchmark_eval.py \
  --model_name            "$MODEL" \
  --nick_name             "$NICK" \
  --tokenizer_name        "$MODEL" \
  --dataset_name_or_path  "$DATASET_PATH" \
  --split_name            test \
  --output_dir            "$OUTPUT_DIR" \
  --tensor_parallel_size  1 \
  --gpu_memory_utilization 0.75 \
  --dtype                 bfloat16 \
  --max_tokens            12288 \
  --temperature           0.6 \
  --top_p                 1.0 \
  --top_k                -1 \
  --pass_at_k             "$PASS_AT_K" \
  --max_num_batched_tokens 32768