#!/bin/bash
# Parallel inception for 4 frontier models, single trial each.
# All 4 jobs share one warm architect server (vLLM serve OpenThinker3-7B)
# and queue against the same GPU process — each cell becomes API-only and
# can run concurrently.
#
# Prerequisite: start the architect server first
#   bash scripts/start_architect_server.sh    # foreground in another tmux pane
# or nohup-it as a background daemon.
#
# Override the architect URL with: export ARCHITECT_BASE_URL=http://...:8001/v1
set -e

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
ARCHITECT_MODEL="open-thoughts/OpenThinker3-7B"
ARCHITECT_CLIENT="local_architect"   # see core/openaiapi.py:PROVIDERS
MAX_ITERATIONS=5
ARCHITECT_INITIAL_MAX_TOKENS=256
MODELS_YAML="config/target_models.yaml"
DEFAULT_MAX_TOKENS=32768

MODELS=(
    "deepseek-ai/DeepSeek-V4-Pro|DeepSeek-V4-Pro|deepseek_beta"
    "deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash|deepseek_beta"
    "moonshotai/Kimi-K2.6|Kimi-K2.6|deepinfra"
    "zai-org/GLM-5.1|GLM-5.1|deepinfra"
)

get_max_tokens() {
    local nick="$1"
    python -c "
import yaml
with open('${MODELS_YAML}') as f:
    data = yaml.safe_load(f)
for m in data.get('think_models', []) + data.get('instruct_models', []):
    if m.get('nick_name') == '${nick}':
        print(m.get('max_tokens', ${DEFAULT_MAX_TOKENS}))
        break
else:
    print(${DEFAULT_MAX_TOKENS})
"
}

# Sanity check: architect server reachable?
ARCHITECT_HEALTH_URL="${ARCHITECT_BASE_URL:-http://localhost:8001/v1}/models"
if ! curl -sf "${ARCHITECT_HEALTH_URL}" > /dev/null 2>&1; then
    echo "ERROR: architect server not reachable at ${ARCHITECT_HEALTH_URL}"
    echo "Start it first: bash scripts/start_architect_server.sh"
    exit 1
fi
echo "[$(date +%H:%M:%S)] architect server reachable at ${ARCHITECT_HEALTH_URL}"

run_one() {
    local model_name="$1" nick="$2" client="$3"
    local max_tokens
    max_tokens=$(get_max_tokens "$nick")
    echo "[$(date +%H:%M:%S)] inception START $nick (max_tokens=${max_tokens})"
    python src/main.py \
        --target_model_name "${model_name}" \
        --target_nick_name "${nick}" \
        --client_name "${client}" \
        --architect_model_name "${ARCHITECT_MODEL}" \
        --architect_client_name "${ARCHITECT_CLIENT}" \
        --dataset_name "${DATASET_NAME}" \
        --split_name "test" \
        --results_dir "${RESULTS_DIR}" \
        --max_tokens "${max_tokens}" \
        --min_reasoning_tokens 2048 \
        --architect_initial_max_tokens "${ARCHITECT_INITIAL_MAX_TOKENS}" \
        --architect_reiterate_max_tokens 64 \
        --max_iterations "${MAX_ITERATIONS}" \
        --temperature 0.6 \
        --top_p 1.0
    echo "[$(date +%H:%M:%S)] inception DONE $nick"
}

export -f get_max_tokens run_one
export MODELS_YAML DEFAULT_MAX_TOKENS RESULTS_DIR DATASET_NAME ARCHITECT_MODEL \
       ARCHITECT_CLIENT MAX_ITERATIONS ARCHITECT_INITIAL_MAX_TOKENS

PIDS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name nick client <<< "$entry"
    run_one "$model_name" "$nick" "$client" &
    PIDS+=($!)
    sleep 5
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${MODELS[$i]%%|*}"
        FAILED=$((FAILED + 1))
    fi
done

echo "=== inception 4 models (parallel) done — failed: $FAILED ==="
exit $FAILED
