#!/usr/bin/env python3
"""
Full-scale Qwen3.5-122B local judge evaluation master script.

Evaluates all inception data pickles under evaluation/inception_data/ using
Qwen3.5-122B-A10B-FP8 served locally via vLLM as the judge.

Prerequisites:
  1. vLLM server running:  bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh
  2. Verify server:        curl http://localhost:8000/health
  3. Eval venv active:     source rlvr_eval/bin/activate

Covers 3 branches:
  - max_iterations_5  (think/ + instruct/)
  - ablation          (max_iterations_1/think/architect_initial_max_tokens_{N}/)
  - simple_inject     (think/ + instruct/)

Output: evaluation/eval_qwen_judge/<branch_subpath>/<model>/

Usage:
  python evaluation/scripts/run_qwen_judge_full.py
  python evaluation/scripts/run_qwen_judge_full.py --branch ablation
  python evaluation/scripts/run_qwen_judge_full.py --dry-run
  python evaluation/scripts/run_qwen_judge_full.py --rps 30
  python evaluation/scripts/run_qwen_judge_full.py --vllm-url http://gpu-node:8000/v1
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
import urllib.request
import urllib.error
from collections import namedtuple
from pathlib import Path

from dotenv import load_dotenv

# -- Path setup ---------------------------------------------------------------

_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent

load_dotenv(_project_root / "core" / ".env")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(_project_root))

from evaluation.preprocess_results import get_full_trace  # noqa: E402

import importlib.util  # noqa: E402


def _import_module_directly(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_safety_judge_mod = _import_module_directly(
    "safety_judge", _eval_dir / "safety-judge.py"
)
SafetyEvaluator = _safety_judge_mod.SafetyEvaluator

# -- Judge constants -----------------------------------------------------------

JUDGE_MODEL = "Qwen3.5-397B-A17B-FP8"        # Must match --served-model-name
JUDGE_PROVIDER = "vllm_local"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 16000                       # Local: no API cap, avoid truncation
JUDGE_MAX_VALIDATION_RETRIES = 3
JUDGE_STRONGREJECT_ONLY = False

DEFAULT_RPS = 30.0                             # Local vLLM: no API quota
DEFAULT_VLLM_URL = "http://localhost:8000/v1"

INCEPTION_DATA = _eval_dir / "inception_data"
OUTPUT_BASE = _eval_dir / "eval_qwen397b_judge"

# -- EvalJob -------------------------------------------------------------------

EvalJob = namedtuple(
    "EvalJob",
    ["branch", "label", "input_pickle", "output_dir", "model_name"],
)

# -- Logging -------------------------------------------------------------------


def setup_logging(output_base: Path) -> logging.Logger:
    """Configure dual logging to stderr and eval_qwen_judge/run.log."""
    output_base.mkdir(parents=True, exist_ok=True)
    log_file = output_base / "run.log"

    fmt = "[%(asctime)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger("qwen_judge")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    return root


# -- Pre-flight check ---------------------------------------------------------


def check_vllm_server(base_url: str, logger: logging.Logger) -> bool:
    """Verify vLLM server is reachable and serving the expected model."""
    # Health check
    health_url = base_url.replace("/v1", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            if resp.status != 200:
                logger.error(f"vLLM health check failed: HTTP {resp.status}")
                return False
    except (urllib.error.URLError, OSError) as e:
        logger.error(
            f"vLLM server not reachable at {health_url}: {e}\n"
            f"Start it first: bash deployment/vllm-judge/scripts/operator/run_vllm_judge.sh"
        )
        return False

    # Model check
    import json
    models_url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            model_ids = [m["id"] for m in data.get("data", [])]
            if JUDGE_MODEL not in model_ids:
                logger.warning(
                    f"Expected model '{JUDGE_MODEL}' not in /v1/models: {model_ids}. "
                    f"Proceeding anyway -- ensure --eval_model matches."
                )
    except Exception as e:
        logger.warning(f"Could not verify model list: {e}")

    logger.info(f"vLLM server OK at {base_url}")
    return True


# -- Job discovery -------------------------------------------------------------


def _discover_pickles(directory: Path, branch: str, label: str, output_subpath: str) -> list:
    """Scan directory for .pickle files and return EvalJob list."""
    jobs = []
    if not directory.exists():
        return jobs
    for pkl in sorted(directory.glob("*.pickle")):
        model_name = pkl.stem
        output_dir = OUTPUT_BASE / output_subpath / model_name
        jobs.append(
            EvalJob(
                branch=branch,
                label=label,
                input_pickle=pkl,
                output_dir=output_dir,
                model_name=model_name,
            )
        )
    return jobs


def build_eval_jobs(branch_filter: str) -> list:
    """
    Auto-discover all pickles across the 3 branches and return a flat list of EvalJobs.

    Output path mapping:
      max_iterations_5/{think|instruct}/{model}.pickle
          -> eval_qwen_judge/max_iterations_5/{think|instruct}/{model}/

      max_iterations_1/think/architect_initial_max_tokens_{N}/{model}.pickle
          -> eval_qwen_judge/ablation/think/tokens_{N}/{model}/

      simple_inject/{think|instruct}/{model}.pickle
          -> eval_qwen_judge/simple_inject/{think|instruct}/{model}/
    """
    jobs = []

    # Branch 1: max_iterations_5
    if branch_filter in ("all", "max_iterations_5"):
        for subtype in ("think", "instruct"):
            src_dir = INCEPTION_DATA / "max_iterations_5" / subtype
            jobs += _discover_pickles(
                src_dir,
                branch="max_iterations_5",
                label=f"max_iterations_5/{subtype}",
                output_subpath=f"max_iterations_5/{subtype}",
            )

    # Branch 2: ablation (max_iterations_1)
    if branch_filter in ("all", "ablation"):
        ablation_base = INCEPTION_DATA / "max_iterations_1" / "think"
        if ablation_base.exists():
            for token_dir in sorted(ablation_base.iterdir()):
                if not token_dir.is_dir():
                    continue
                dir_name = token_dir.name
                if "max_tokens_" in dir_name:
                    token_level = dir_name.split("max_tokens_")[-1]
                else:
                    token_level = dir_name
                jobs += _discover_pickles(
                    token_dir,
                    branch="ablation",
                    label=f"ablation/think/tokens_{token_level}",
                    output_subpath=f"ablation/think/tokens_{token_level}",
                )

    # Branch 3: simple_inject
    if branch_filter in ("all", "simple_inject"):
        simple_base = INCEPTION_DATA / "simple_inject"
        if simple_base.exists():
            for subtype in ("think", "instruct"):
                src_dir = simple_base / subtype
                jobs += _discover_pickles(
                    src_dir,
                    branch="simple_inject",
                    label=f"simple_inject/{subtype}",
                    output_subpath=f"simple_inject/{subtype}",
                )

    return jobs


# -- Per-job execution ---------------------------------------------------------


def run_single_eval(job: EvalJob, rps: float, logger: logging.Logger) -> None:
    """Execute preprocessing + SafetyEvaluator for a single pickle."""
    # Step 1: Load source pickle
    df = pd.read_pickle(job.input_pickle)
    n_rows = len(df)
    n_bio = int((df.get("category", pd.Series()) == "wmdp-bio").sum()) if "category" in df.columns else 0
    n_chem = int((df.get("category", pd.Series()) == "wmdp-chem").sum()) if "category" in df.columns else 0
    logger.info(f"  Preprocessing... ({n_rows} rows, {n_bio} bio, {n_chem} chem)")

    # Step 2: Preprocess -- add reasoning_traces column
    job.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = job.output_dir / f"{job.model_name}_preprocessed.pickle"

    if preprocessed_path.exists():
        logger.info(f"  Preprocessed pickle exists, reusing: {preprocessed_path}")
        df_preprocessed = pd.read_pickle(preprocessed_path)
    else:
        df["reasoning_traces"] = df.apply(get_full_trace, axis=1)
        df.to_pickle(preprocessed_path)
        df_preprocessed = df
        logger.info(f"  Preprocessed: saved {preprocessed_path}")

    # Step 3: Instantiate SafetyEvaluator
    logger.info(f"  Running SafetyEvaluator...")
    evaluator = SafetyEvaluator(
        input_df=df_preprocessed,
        inquiry_col="inquiry",
        reasoning_trace_col="reasoning_traces",
        category_col="category",
        chemical_label="wmdp-chem",
        bioterrorism_label="wmdp-bio",
        output_dir=str(job.output_dir),
        nick_name=job.model_name,
        eval_model=JUDGE_MODEL,
        client_name=JUDGE_PROVIDER,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        requests_per_second=rps,
        max_validation_retries=JUDGE_MAX_VALIDATION_RETRIES,
        strongreject_only=JUDGE_STRONGREJECT_ONLY,
        max_consecutive_refusals=0,
    )

    # Step 4: Run (async -> sync bridge)
    loop = asyncio.new_event_loop()
    try:
        result_df = loop.run_until_complete(evaluator.run())
    finally:
        loop.close()

    n_result = len(result_df) if result_df is not None else 0
    logger.info(f"  SafetyEvaluator complete: {n_result} rows in result")


# -- Main ----------------------------------------------------------------------


def fmt_duration(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(
        description="Run full Qwen3.5-122B local judge evaluation over all inception data pickles."
    )
    parser.add_argument(
        "--branch",
        choices=["max_iterations_5", "ablation", "simple_inject", "all"],
        default="all",
        help="Which branch(es) to evaluate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List jobs without executing any evaluations",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help=f"Requests per second rate limit (default: {DEFAULT_RPS})",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_URL,
        help=f"vLLM server base URL (default: {DEFAULT_VLLM_URL})",
    )
    args = parser.parse_args()

    # Set VLLM_BASE_URL for the provider registry + set dummy API key
    os.environ["VLLM_BASE_URL"] = args.vllm_url
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")

    logger = setup_logging(OUTPUT_BASE)

    # -- Pre-flight: verify vLLM server is up ---------------------------------
    if not args.dry_run:
        if not check_vllm_server(args.vllm_url, logger):
            sys.exit(1)

    # -- Discover jobs --------------------------------------------------------
    all_jobs = build_eval_jobs(args.branch)
    total = len(all_jobs)
    total_rows = total * 800

    logger.info("===== QWEN LOCAL JUDGE FULL EVALUATION =====")
    logger.info(
        f"Judge: {JUDGE_MODEL} | Provider: {JUDGE_PROVIDER} (local vLLM) | Mode: comprehensive"
    )
    logger.info(f"vLLM URL: {args.vllm_url}")
    logger.info(f"Total jobs: {total} | Estimated rows: {total_rows:,}")
    logger.info(f"Output: {OUTPUT_BASE}")
    logger.info(f"RPS: {args.rps} | max_tokens: {JUDGE_MAX_TOKENS} | temperature: {JUDGE_TEMPERATURE}")
    logger.info("-" * 45)

    if not all_jobs:
        logger.info("No jobs found. Check that inception_data/ directories exist.")
        sys.exit(0)

    # -- Dry-run mode ---------------------------------------------------------
    if args.dry_run:
        logger.info(f"DRY-RUN: listing {total} jobs (no evaluation will run)")
        for i, job in enumerate(all_jobs, 1):
            final_pickle = job.output_dir / f"{job.model_name}_safety_judge.pickle"
            status = "DONE" if final_pickle.exists() else "PENDING"
            logger.info(
                f"  [{i:3d}/{total}] [{status}] {job.label}/{job.model_name}"
                f"\n           src : {job.input_pickle}"
                f"\n           out : {job.output_dir}"
            )
        sys.exit(0)

    # -- Sequential execution -------------------------------------------------
    completed = []
    skipped = []
    failed = []

    wall_start = time.monotonic()

    current_branch = None
    branch_jobs_count = {}
    for job in all_jobs:
        branch_jobs_count[job.branch] = branch_jobs_count.get(job.branch, 0) + 1

    branch_order = []
    seen_branches = set()
    for job in all_jobs:
        if job.branch not in seen_branches:
            branch_order.append(job.branch)
            seen_branches.add(job.branch)

    branch_idx = {b: i + 1 for i, b in enumerate(branch_order)}

    for job_num, job in enumerate(all_jobs, 1):
        if job.branch != current_branch:
            current_branch = job.branch
            bidx = branch_idx[job.branch]
            n_branch = branch_jobs_count[job.branch]
            logger.info(
                f"[BRANCH {bidx}/{len(branch_order)}] {job.branch} ({n_branch} pickles)"
            )

        final_pickle = job.output_dir / f"{job.model_name}_safety_judge.pickle"

        # Resume: skip already-completed jobs
        if final_pickle.exists():
            logger.info(
                f"[{job_num}/{total}] SKIP {job.label}/{job.model_name} -- already complete"
            )
            skipped.append(job)
            continue

        # Missing source pickle
        if not job.input_pickle.exists():
            logger.warning(
                f"[WARNING] {job.label}/{job.model_name} -- source pickle not found, skipping: {job.input_pickle}"
            )
            skipped.append(job)
            continue

        logger.info(f"[{job_num}/{total}] START {job.label}/{job.model_name}")
        job_start = time.monotonic()

        try:
            run_single_eval(job, rps=args.rps, logger=logger)
            elapsed = time.monotonic() - job_start
            logger.info(
                f"[{job_num}/{total}] DONE {job.label}/{job.model_name} ({fmt_duration(elapsed)})"
            )
            completed.append(job)
        except Exception:
            elapsed = time.monotonic() - job_start
            tb = traceback.format_exc()
            logger.error(
                f"[{job_num}/{total}] FAIL {job.label}/{job.model_name} ({fmt_duration(elapsed)})\n{tb}"
            )
            failed.append(job)

    # -- Summary --------------------------------------------------------------
    total_wall = time.monotonic() - wall_start
    logger.info("===== EVALUATION COMPLETE =====")
    logger.info(
        f"Completed: {len(completed)}/{total} | "
        f"Skipped: {len(skipped)} | "
        f"Failed: {len(failed)} | "
        f"Time: {fmt_duration(total_wall)}"
    )

    if failed:
        logger.info("Failed jobs:")
        for job in failed:
            logger.info(f"  - {job.label}/{job.model_name}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
