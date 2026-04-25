#!/bin/bash
# Sequential overnight runner: inception → simple_inject → benchmark.
# Each underlying script is idempotent (skips existing pickles), so retries resume.
cd /home/al2644/research/codebase/reasoning/inception

run_with_retry() {
    local script="$1"
    local max_retries=4
    local backoff=30
    local attempt=0
    while (( attempt < max_retries )); do
        attempt=$((attempt + 1))
        echo "==================================================================="
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt $attempt/$max_retries: $script"
        echo "==================================================================="
        if bash "$script"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $script (attempt $attempt)"
            return 0
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $script (attempt $attempt, exit=$?)"
        if (( attempt < max_retries )); then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] sleeping ${backoff}s before retry..."
            sleep "$backoff"
            backoff=$((backoff * 2))
        fi
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GIVING UP: $script after $max_retries attempts"
    return 1
}

echo "=== Overnight 4-model run start: $(date) ==="
echo "Order: inception (Pool B, GPU) → simple_inject (Pool A, API) → benchmark (Pool A, API)"

run_with_retry scripts/run_inception_4models.sh
INC_RC=$?

run_with_retry scripts/run_simple_inject_4models.sh
SI_RC=$?

run_with_retry scripts/run_benchmark_4models.sh
BM_RC=$?

echo
echo "=== Overnight done: $(date) ==="
echo "  inception:     $([[ $INC_RC -eq 0 ]] && echo OK || echo FAILED)"
echo "  simple_inject: $([[ $SI_RC -eq 0 ]] && echo OK || echo FAILED)"
echo "  benchmark:     $([[ $BM_RC -eq 0 ]] && echo OK || echo FAILED)"

exit $((INC_RC + SI_RC + BM_RC))
