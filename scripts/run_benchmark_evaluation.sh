#!/bin/bash
# Run safety evaluation on all benchmark results
# Usage: bash scripts/run_benchmark_evaluation.sh [--overwrite]

set -e

OVERWRITE_FLAG=""
if [[ "$1" == "--overwrite" ]]; then
    OVERWRITE_FLAG="--overwrite"
fi

EVAL_MODEL="deepseek-ai/DeepSeek-V3"
CLIENT="deepinfra"
RATE_LIMIT="5.0"

# Common arguments
COMMON_ARGS="--eval_model $EVAL_MODEL --client_name $CLIENT --inquiry_col inquiry --reasoning_trace_col response --category_col category --bioterrorism_label wmdp-bio --chemical_label wmdp-chem --rate_limit $RATE_LIMIT $OVERWRITE_FLAG"

echo "=================================================="
echo "Benchmark Safety Evaluation Pipeline"
echo "=================================================="

# Create output directories
mkdir -p think-vs-instruct-benchmark/evaluation-results/benchmark/think
mkdir -p think-vs-instruct-benchmark/evaluation-results/benchmark/instruct

# Process think models
echo -e "\n[THINK MODELS]"
for pickle in think-vs-instruct-benchmark/single-inject-results/think/*_benchmark*.pickle; do
    if [[ -f "$pickle" ]]; then
        basename=$(basename "$pickle" .pickle)
        model_name=${basename%_benchmark*}
        # Handle togetherai suffix
        if [[ "$basename" == *"_togetherai"* ]]; then
            model_name="${model_name}-TogetherAI"
        fi

        echo "Processing: $model_name"
        python evaluation/safety-judge.py \
            --input_filepath "$pickle" \
            --output_dir "./think-vs-instruct-benchmark/evaluation-results/benchmark/think/$model_name" \
            --nick_name "$model_name" \
            $COMMON_ARGS
    fi
done

# Process instruct models
echo -e "\n[INSTRUCT MODELS]"
for pickle in think-vs-instruct-benchmark/single-inject-results/instruct/*_benchmark*.pickle; do
    if [[ -f "$pickle" ]]; then
        basename=$(basename "$pickle" .pickle)
        model_name=${basename%_benchmark*}
        # Handle togetherai suffix
        if [[ "$basename" == *"_togetherai"* ]]; then
            model_name="${model_name}-TogetherAI"
        fi

        echo "Processing: $model_name"
        python evaluation/safety-judge.py \
            --input_filepath "$pickle" \
            --output_dir "./think-vs-instruct-benchmark/evaluation-results/benchmark/instruct/$model_name" \
            --nick_name "$model_name" \
            $COMMON_ARGS
    fi
done

# Run aggregation
echo -e "\n[AGGREGATION]"
python evaluation/aggregate_benchmark_results.py --output_dir ./think-vs-instruct-benchmark/evaluation-results/benchmark

echo -e "\n=================================================="
echo "Evaluation complete!"
echo "Results: think-vs-instruct-benchmark/evaluation-results/benchmark/"
echo "=================================================="
