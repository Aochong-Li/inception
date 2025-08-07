#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID

DATASET_NAME="allmath"
RESULTS_DIR="./results/${DATASET_NAME}"

python stress-test/teacher_guide.py \
    --model_name "${MODEL}" \
    --nick_name "${NICK}" \
    --tokenizer_name "${MODEL}" \
    --results_dir "${RESULTS_DIR}" \
    --sample_size 50 \
    --tensor_parallel_size $(echo "$GPU_ID" | awk -F',' '{print NF}') \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --granularity 30 \
    --max_num_batched_tokens 32768 \
    --num_responses_per_problem 8 \
    --overwrite

