#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

# -------- static bits you rarely touch --------
DATASET_NAME="wmdp_inquiries_300"
SAMPLE_K=5
DATASET_PATH="aochongoliverli/wmdp_inquiries_300"
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
  --tensor_parallel_size  $(echo "$GPU_ID" | awk -F',' '{print NF}') \
  --gpu_memory_utilization 0.9 \
  --dtype                 bfloat16 \
  --max_tokens            32768 \
  --max_num_batched_tokens 8192 \
  --temperature           0.6 \
  --top_p                 0.95 \
  --top_k                -1 \
  --sample_k             "$SAMPLE_K"