#!/usr/bin/env bash
# Apptainer is a strict prerequisite (see agent/plans/vllm-judge-docker-deployment.md §2.12).
# No GPUs. Exit 0 if apptainer is on PATH after module load; exit 1 with a report otherwise.
set -euo pipefail

APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer-1.4.5}"
REPORT_PATH="${APPTAINER_PREREQ_REPORT_PATH:-}"

emit_report() {
  cat <<EOF
================================================================
APPTAINER PREREQUISITE CHECK FAILED
================================================================
hostname: $(hostname 2>/dev/null || echo unknown)
date: $(date -Iseconds 2>/dev/null || date)
APPTAINER_MODULE tried: ${APPTAINER_MODULE}

--- command -v module ---
$(command -v module 2>&1 || echo "(module not on PATH)")

--- module avail (filtered: apptainer | singularity) ---
$(if command -v module >/dev/null 2>&1; then module avail 2>&1 | grep -Ei 'apptainer|singularity' || echo "(no matching lines)"; else echo "(skipped: module not found)"; fi)

--- PATH ---
${PATH:-}

--- result ---
${FAIL_REASON:-unknown}

Apptainer is REQUIRED. Do not substitute Docker or bare uv+vLLM for this deployment.
Fix: load a site module (e.g. module load ${APPTAINER_MODULE}) or install Apptainer, then re-run:
  bash deployment/vllm-judge/scripts/operator/check_apptainer_prerequisite.sh
================================================================
EOF
}

exit_fail() {
  if [[ -n "${REPORT_PATH}" ]]; then
    emit_report | tee "${REPORT_PATH}"
  else
    emit_report 1>&2
  fi
  exit 1
}

if ! command -v module >/dev/null 2>&1; then
  FAIL_REASON="module: not found; cannot load ${APPTAINER_MODULE}"
  exit_fail
fi

set +u
module load "${APPTAINER_MODULE}" 2>&1 || true
set -u

if command -v apptainer >/dev/null 2>&1; then
  echo "OK: $(command -v apptainer)"
  apptainer --version
  exit 0
fi

FAIL_REASON="apptainer not on PATH after: module load ${APPTAINER_MODULE}"
exit_fail
