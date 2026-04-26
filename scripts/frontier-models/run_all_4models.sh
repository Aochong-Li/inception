#!/bin/bash
# Full experimental sweep for the 4 frontier models (V4-Pro, V4-Flash,
# Kimi-K2.6, GLM-5.1) — runs simple_inject, benchmark, inception, and the
# 5-cell architect-token ablation.
#
# Concurrency policy: AT MOST 2 jobs in flight at any time, paired so that
# every wave has one DeepSeek model + one DeepInfra model. This balances
# load across providers (no single provider sees >1 concurrent request from
# this orchestrator) and lets the warm shared architect server batch two
# inception/ablation cells together.
#
# Architect server prerequisite: must already be running at $ARCHITECT_BASE_URL
# (defaults to http://localhost:8001/v1). Start with
#   nohup bash scripts/start_architect_server.sh > logs/architect_server.log 2>&1 &
#
# Output: ./results/ (production layout). All scripts skip-if-exists, so
# this orchestrator is safely re-runnable to resume after partial failures.
set -e
cd "$(dirname "$0")/../.."

export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HOME/.cache/huggingface}"
unset HF_DATASETS_CACHE

DATASET_NAME="aochongoliverli/wmdp_biochem_inquiries_800"
RESULTS_DIR="./results"
ARCHITECT_MODEL="open-thoughts/OpenThinker3-7B"
ARCHITECT_CLIENT="local_architect"
LOG_DIR="logs/full_sweep"
mkdir -p "${LOG_DIR}"

# Models — Kimi-K2.6 dropped (DeepInfra was throttling it specifically with
# ~15-37% per-row failures across cells; V4-Pro / V4-Flash / GLM-5.1 only).
# Wave layout: V4-Pro pairs with GLM-5.1 (deepseek + deepinfra in parallel),
# V4-Flash runs alone in a second wave.
PAIR_DS="deepseek-ai/DeepSeek-V4-Pro|DeepSeek-V4-Pro|deepseek_beta|380000"
PAIR_DI="zai-org/GLM-5.1|GLM-5.1|deepinfra|65535"
SOLO="deepseek-ai/DeepSeek-V4-Flash|DeepSeek-V4-Flash|deepseek_beta|380000"

# Pre-flight: architect server reachable?
ARCHITECT_HEALTH_URL="${ARCHITECT_BASE_URL:-http://localhost:8001/v1}/models"
if ! curl -sf "${ARCHITECT_HEALTH_URL}" > /dev/null 2>&1; then
    echo "ERROR: architect server not reachable at ${ARCHITECT_HEALTH_URL}"
    echo "Start it first: nohup bash scripts/start_architect_server.sh > logs/architect_server.log 2>&1 &"
    exit 1
fi
echo "[$(date +%H:%M:%S)] ✓ architect server reachable at ${ARCHITECT_HEALTH_URL}"

# ─── Per-cell runners ──────────────────────────────────────────────────────
run_simple_inject() {
    local entry="$1"
    IFS='|' read -r model_name nick client max_tokens <<< "$entry"
    local log="${LOG_DIR}/simple_inject_${nick}.log"
    echo "[$(date +%H:%M:%S)] simple_inject START $nick"
    python src/simple_inject.py \
        --target_model_name "${model_name}" \
        --target_nick_name "${nick}" \
        --dataset_name "${DATASET_NAME}" \
        --split_name "test" \
        --results_dir "${RESULTS_DIR}" \
        --max_tokens "${max_tokens}" \
        --temperature 0.6 \
        --top_p 1.0 \
        --client_name "${client}" \
        > "${log}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] simple_inject DONE  $nick (rc=${rc})"
    return $rc
}

run_benchmark() {
    local entry="$1"
    IFS='|' read -r model_name nick client max_tokens <<< "$entry"
    local log="${LOG_DIR}/benchmark_${nick}.log"
    echo "[$(date +%H:%M:%S)] benchmark START $nick"
    python src/benchmark.py \
        --model_name "${model_name}" \
        --nick_name "${nick}" \
        --tokenizer_name "${model_name}" \
        --dataset_name_or_path "${DATASET_NAME}" \
        --split_name "test" \
        --output_dir "${RESULTS_DIR}/benchmark/think" \
        --max_tokens "${max_tokens}" \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k -1 \
        --sample_k 1 \
        --client_name "${client}" \
        --overwrite False \
        > "${log}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] benchmark DONE  $nick (rc=${rc})"
    return $rc
}

run_architect_cell() {
    local entry="$1" max_iter="$2" arc_tokens="$3" tag="$4"
    IFS='|' read -r model_name nick client max_tokens <<< "$entry"
    local log="${LOG_DIR}/${tag}_${nick}_arc${arc_tokens}_iter${max_iter}.log"
    echo "[$(date +%H:%M:%S)] ${tag} START $nick (arc=${arc_tokens}, iter=${max_iter})"
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
        --architect_initial_max_tokens "${arc_tokens}" \
        --architect_reiterate_max_tokens 64 \
        --max_iterations "${max_iter}" \
        --temperature 0.6 \
        --top_p 1.0 \
        > "${log}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] ${tag} DONE  $nick (rc=${rc}, arc=${arc_tokens}, iter=${max_iter})"
    return $rc
}

# ─── Wave runners ─────────────────────────────────────────────────────────
# pair_*: run DS + DI in parallel (one of each provider)
# solo_*: run a single cell alone (used for V4-Flash, since GLM is paired with V4-Pro)
pair_simple_inject() {
    run_simple_inject "$PAIR_DS" &
    local p1=$!
    run_simple_inject "$PAIR_DI" &
    local p2=$!
    wait "$p1" || WAVE_FAILED=$((WAVE_FAILED + 1))
    wait "$p2" || WAVE_FAILED=$((WAVE_FAILED + 1))
}
solo_simple_inject() {
    run_simple_inject "$SOLO" || WAVE_FAILED=$((WAVE_FAILED + 1))
}

pair_benchmark() {
    run_benchmark "$PAIR_DS" &
    local p1=$!
    run_benchmark "$PAIR_DI" &
    local p2=$!
    wait "$p1" || WAVE_FAILED=$((WAVE_FAILED + 1))
    wait "$p2" || WAVE_FAILED=$((WAVE_FAILED + 1))
}
solo_benchmark() {
    run_benchmark "$SOLO" || WAVE_FAILED=$((WAVE_FAILED + 1))
}

pair_architect() {
    local max_iter="$1" arc_tokens="$2" tag="$3"
    run_architect_cell "$PAIR_DS" "$max_iter" "$arc_tokens" "$tag" &
    local p1=$!
    run_architect_cell "$PAIR_DI" "$max_iter" "$arc_tokens" "$tag" &
    local p2=$!
    wait "$p1" || WAVE_FAILED=$((WAVE_FAILED + 1))
    wait "$p2" || WAVE_FAILED=$((WAVE_FAILED + 1))
}
solo_architect() {
    local max_iter="$1" arc_tokens="$2" tag="$3"
    run_architect_cell "$SOLO" "$max_iter" "$arc_tokens" "$tag" || WAVE_FAILED=$((WAVE_FAILED + 1))
}

WAVE_FAILED=0

# ═══════════════════════════════════════════════════════════════════════════
# PHASE B — API-only sweep (simple_inject + benchmark)
# 3 simple_inject cells + 3 benchmark cells = 6 cells (Kimi dropped)
# Per stage: pair (V4-Pro + GLM-5.1) then solo (V4-Flash)
# ═══════════════════════════════════════════════════════════════════════════
echo
echo "════════════════════════════════════════════════════════════════════"
echo "[$(date +%H:%M:%S)] PHASE B: simple_inject + benchmark (6 cells; skip-if-exists handles done cells)"
echo "════════════════════════════════════════════════════════════════════"
pair_simple_inject
solo_simple_inject
pair_benchmark
solo_benchmark
echo "[$(date +%H:%M:%S)] PHASE B done — failed so far: $WAVE_FAILED"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE A — Architect-bound sweep (ablation cheapest-first → inception)
# 5 ablation arc-cells × 3 models + 3 inception cells = 18 cells (Kimi dropped)
# Per config: pair wave (V4-Pro + GLM-5.1) then solo wave (V4-Flash)
# Skip-if-exists handles already-completed cells.
# ═══════════════════════════════════════════════════════════════════════════
echo
echo "════════════════════════════════════════════════════════════════════"
echo "[$(date +%H:%M:%S)] PHASE A: architect-bound sweep (18 cells)"
echo "════════════════════════════════════════════════════════════════════"

# Ablation: max_iter=1, arc ∈ {128, 256, 512, 768, 1024}, cheapest first
for arc in 128 256 512 768 1024; do
    echo
    echo "─── ablation arc=${arc} (max_iter=1) ───"
    pair_architect 1 "$arc" "ablation"
    solo_architect 1 "$arc" "ablation"
done

# Inception: max_iter=5, arc=256 (the canonical full-loop run)
echo
echo "─── inception max_iter=5, arc=256 ───"
pair_architect 5 256 "inception"
solo_architect 5 256 "inception"

echo "[$(date +%H:%M:%S)] PHASE A done"

echo
echo "════════════════════════════════════════════════════════════════════"
echo "[$(date +%H:%M:%S)] FULL SWEEP DONE — total failed cells: $WAVE_FAILED"
echo "════════════════════════════════════════════════════════════════════"
exit $WAVE_FAILED
