#!/usr/bin/env bash
set -euo pipefail
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID

EVAL_DIR="./results/countdown"
QUESTION_IDS_FNAME="stress_test_problems.json"

python stress-test/inject_distractor.py \
    --task "countdown" \
    --model_name "${MODEL}" \
    --nick_name "${NICK}" \
    --tokenizer_name "${MODEL}" \
    --results_dir "${EVAL_DIR}" \
    --question_ids_fname "${QUESTION_IDS_FNAME}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 12288 \
    --temperature 0.6 \
    --top_p 1.0 \
    --top_k -1 \
    --granularity 30 \
    --num_distract_candidates 20 \
    --unit 0.25 \
    --overwrite