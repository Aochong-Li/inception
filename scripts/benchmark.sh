#!/bin/bash
set -ex

# Benchmark: direct query evaluation (no jailbreak).
# Sends the same 800 inquiries used in inception directly to each target model.
# Supports both local vLLM (client_name="") and API (client_name="deepinfra", etc.)
#
# Usage:
#   bash scripts/benchmark.sh                      # run all models from YAML
#   bash scripts/benchmark.sh --overwrite True      # re-run even if results exist

# -------- config --------
MODELS_YAML="config/target_models.yaml"
DATASET_PATH="aochongoliverli/wmdp_biochem_inquiries_800"
SPLIT="test"
OUTPUT_BASE="./results/benchmark"
OVERWRITE="${1:-False}"
DEFAULT_MAX_TOKENS=32768
# -------------------------

# Extract model info with group (think/instruct) for subdirectory separation
# Includes optional max_tokens from YAML config
MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for group, subdir in [('think_models', 'think'), ('instruct_models', 'instruct')]:
    for model in data.get(group, []):
        client = model.get('client_name', '')
        max_tokens = model.get('max_tokens', '$DEFAULT_MAX_TOKENS')
        print(f\"{model['model_name']},{model['nick_name']},{client},{subdir},{max_tokens}\")
")

echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name client_name subdir max_tokens; do
    OUTPUT_DIR="${OUTPUT_BASE}/${subdir}"
    mkdir -p "$OUTPUT_DIR"

    echo "========================================"
    echo "Benchmark: $nick_name ($subdir) via ${client_name:-local} [max_tokens=$max_tokens]"
    echo "========================================"

    python src/benchmark.py \
        --model_name "$model_name" \
        --nick_name "$nick_name" \
        --tokenizer_name "$model_name" \
        --dataset_name_or_path "$DATASET_PATH" \
        --split_name "$SPLIT" \
        --output_dir "$OUTPUT_DIR" \
        --client_name "$client_name" \
        --max_tokens "$max_tokens" \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k -1 \
        --sample_k 1 \
        --overwrite "$OVERWRITE"
done
