#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID
DATASET_NAME="math500amc23"
RESULTS_DIR="./results/${DATASET_NAME}"


python stress-test/inject_distractor.py \
    --model_name "${MODEL}" \
    --nick_name "${NICK}" \
    --tokenizer_name "${MODEL}" \
    --results_dir "${RESULTS_DIR}" \
    --sample_size 250 \
    --tensor_parallel_size $(echo "$GPU_ID" | awk -F',' '{print NF}') \
    --gpu_memory_utilization 0.9 \
    --dtype bfloat16 \
    --max_tokens 8192 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --granularity 30 \
    --num_distract_candidates 20 \
    --unit 0.2 \
    --overwrite