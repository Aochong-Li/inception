#!/usr/bin/env python3
"""
Sanity Check 2: Run 50-sample eval subset with the full Qwen3.5-122B judge.

Grabs 25 bio + 25 chem samples from one target model's inception data,
runs SafetyEvaluator, and validates the output schema before proceeding
to the full 33-model evaluation.

Prerequisites:
  - vLLM server running with Qwen3.5-122B-A10B-FP8
  - rlvr_eval venv activated
  - inception_data/ populated with at least one model's pickle

Usage:
  python evaluation/scripts/sanity_check_qwen_judge.py
  python evaluation/scripts/sanity_check_qwen_judge.py --vllm-url http://gpu-node:8000/v1
  python evaluation/scripts/sanity_check_qwen_judge.py --input-pickle evaluation/inception_data/max_iterations_5/think/DeepSeek-R1-0528.pickle
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent

sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / "core" / ".env")

import pandas as pd
import importlib.util

from evaluation.preprocess_results import get_full_trace


def _import_module_directly(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_safety_judge_mod = _import_module_directly(
    "safety_judge", _eval_dir / "safety-judge.py"
)
SafetyEvaluator = _safety_judge_mod.SafetyEvaluator

# -- Constants ----------------------------------------------------------------

JUDGE_MODEL = "Qwen3.5-122B-A10B-FP8"
JUDGE_PROVIDER = "vllm_local"
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
N_BIO = 25
N_CHEM = 25
SANITY_OUTPUT_DIR = _eval_dir / "eval_qwen_judge" / "_sanity_check"

# Expected columns in merged output (shared across bio/chem)
REQUIRED_PARSED_COLS = {"complied", "specificity", "convincingness",
                        "severity_level", "fatalities", "delivery_method",
                        "late_refusal"}

INCEPTION_DATA = _eval_dir / "inception_data"


def find_first_pickle() -> Path:
    """Auto-discover the first available inception data pickle."""
    search_dirs = [
        INCEPTION_DATA / "max_iterations_5" / "think",
        INCEPTION_DATA / "max_iterations_5" / "instruct",
    ]
    for d in search_dirs:
        if d.exists():
            pickles = sorted(d.glob("*.pickle"))
            if pickles:
                return pickles[0]
    raise FileNotFoundError(
        f"No inception data pickles found under {INCEPTION_DATA}. "
        f"Ensure evaluation/inception_data/ is populated."
    )


def check_server(base_url: str) -> bool:
    health_url = base_url.replace("/v1", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def subsample(df: pd.DataFrame, n_bio: int, n_chem: int) -> pd.DataFrame:
    """Take n_bio bio + n_chem chem samples from the DataFrame."""
    bio = df[df["category"] == "wmdp-bio"]
    chem = df[df["category"] == "wmdp-chem"]

    bio_sample = bio.head(n_bio) if len(bio) >= n_bio else bio
    chem_sample = chem.head(n_chem) if len(chem) >= n_chem else chem

    subset = pd.concat([bio_sample, chem_sample])
    print(f"  Subsampled: {len(bio_sample)} bio + {len(chem_sample)} chem = {len(subset)} total")
    return subset


def validate_output(output_path: Path) -> dict:
    """Validate the output pickle matches expected schema."""
    results = {
        "file_exists": False,
        "row_count": 0,
        "col_count": 0,
        "has_required_cols": False,
        "missing_cols": [],
        "complied_populated": False,
        "complied_rate": 0.0,
        "nan_rate": {},
    }

    if not output_path.exists():
        print(f"  [FAIL] Output pickle not found: {output_path}")
        return results

    results["file_exists"] = True
    df = pd.read_pickle(output_path)
    results["row_count"] = len(df)
    results["col_count"] = len(df.columns)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Check required columns
    present = set(df.columns) & REQUIRED_PARSED_COLS
    missing = REQUIRED_PARSED_COLS - set(df.columns)
    results["missing_cols"] = list(missing)
    results["has_required_cols"] = len(missing) == 0

    if missing:
        print(f"  [FAIL] Missing required columns: {missing}")
    else:
        print(f"  [PASS] All {len(REQUIRED_PARSED_COLS)} required columns present")

    # Check complied is populated
    if "complied" in df.columns:
        non_null = df["complied"].notna().sum()
        results["complied_populated"] = non_null > 0
        results["complied_rate"] = float(df["complied"].sum()) / len(df) if len(df) > 0 else 0
        print(f"  [{'PASS' if non_null > 0 else 'FAIL'}] 'complied' populated: {non_null}/{len(df)} non-null")
        print(f"  Compliance rate: {results['complied_rate']*100:.1f}%")

    # Check NaN rates for key fields
    for col in ["specificity", "convincingness", "severity_level"]:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            results["nan_rate"][col] = nan_pct
            status = "PASS" if nan_pct < 50 else "WARN"
            print(f"  [{status}] '{col}' NaN rate: {nan_pct:.1f}%")

    # Check raw_response exists
    if "raw_response" in df.columns:
        raw_populated = df["raw_response"].notna().sum()
        print(f"  [{'PASS' if raw_populated > 0 else 'FAIL'}] 'raw_response' populated: {raw_populated}/{len(df)}")
    elif "response" in df.columns:
        raw_populated = df["response"].notna().sum()
        print(f"  [{'PASS' if raw_populated > 0 else 'FAIL'}] 'response' populated: {raw_populated}/{len(df)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: run 50-sample eval with full Qwen judge"
    )
    parser.add_argument(
        "--vllm-url", default=DEFAULT_VLLM_URL,
        help=f"vLLM server base URL (default: {DEFAULT_VLLM_URL})",
    )
    parser.add_argument(
        "--input-pickle", default=None,
        help="Path to a specific inception data pickle (auto-detected if not set)",
    )
    parser.add_argument(
        "--keep-output", action="store_true",
        help="Keep sanity check output (default: delete on success)",
    )
    args = parser.parse_args()

    os.environ["VLLM_BASE_URL"] = args.vllm_url
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")

    print("=" * 60)
    print("SANITY CHECK 2: 50-Sample Eval with Qwen3.5-122B Judge")
    print("=" * 60)
    print()

    # Step 1: Server check
    print("[1/5] Checking vLLM server")
    if not check_server(args.vllm_url):
        print(f"  [FAIL] Server not reachable at {args.vllm_url}")
        sys.exit(1)
    print(f"  [PASS] Server OK at {args.vllm_url}")

    # Verify model
    try:
        models_url = f"{args.vllm_url}/models"
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            model_ids = [m["id"] for m in data.get("data", [])]
            if JUDGE_MODEL in model_ids:
                print(f"  [PASS] Model '{JUDGE_MODEL}' found")
            else:
                print(f"  [WARN] Expected '{JUDGE_MODEL}', found: {model_ids}")
    except Exception:
        print("  [WARN] Could not verify model list")
    print()

    # Step 2: Load and subsample data
    print("[2/5] Loading inception data")
    if args.input_pickle:
        pkl_path = Path(args.input_pickle)
    else:
        try:
            pkl_path = find_first_pickle()
        except FileNotFoundError as e:
            print(f"  [FAIL] {e}")
            sys.exit(1)

    print(f"  Source: {pkl_path}")
    model_name = pkl_path.stem
    df = pd.read_pickle(pkl_path)
    print(f"  Loaded: {len(df)} rows")

    subset = subsample(df, N_BIO, N_CHEM)
    print()

    # Step 3: Preprocess
    print("[3/5] Preprocessing (adding reasoning_traces)")
    subset = subset.copy()
    subset["reasoning_traces"] = subset.apply(get_full_trace, axis=1)

    # Check traces are non-empty
    non_empty = subset["reasoning_traces"].apply(lambda x: bool(x and len(str(x)) > 100)).sum()
    print(f"  Non-empty traces: {non_empty}/{len(subset)}")
    if non_empty == 0:
        print("  [FAIL] All reasoning traces are empty — check preprocess_results.py")
        sys.exit(1)
    print()

    # Step 4: Run SafetyEvaluator
    print("[4/5] Running SafetyEvaluator on 50 samples")
    output_dir = str(SANITY_OUTPUT_DIR / model_name)
    nick_name = f"{model_name}_sanity"

    # Clean previous sanity output
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    evaluator = SafetyEvaluator(
        input_df=subset,
        inquiry_col="inquiry",
        reasoning_trace_col="reasoning_traces",
        category_col="category",
        chemical_label="wmdp-chem",
        bioterrorism_label="wmdp-bio",
        output_dir=output_dir,
        nick_name=nick_name,
        eval_model=JUDGE_MODEL,
        client_name=JUDGE_PROVIDER,
        temperature=0.0,
        max_tokens=16000,
        requests_per_second=10.0,
        max_validation_retries=3,
        strongreject_only=False,
        max_consecutive_refusals=0,
    )

    t0 = time.time()
    loop = asyncio.new_event_loop()
    try:
        result_df = loop.run_until_complete(evaluator.run())
    finally:
        loop.close()
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print()

    # Step 5: Validate output
    print("[5/5] Validating output")
    output_pickle = Path(output_dir) / f"{nick_name}_safety_judge.pickle"
    validation = validate_output(output_pickle)
    print()

    # Also check per-category raw files
    for cat in ["bioterrorism", "chemical"]:
        raw_path = Path(output_dir) / f"{cat}_safety_judge_raw.pickle"
        if raw_path.exists():
            raw_df = pd.read_pickle(raw_path)
            print(f"  {cat} raw: {raw_df.shape}")
        else:
            print(f"  [WARN] {cat} raw pickle not found")
    print()

    # Summary
    print("=" * 60)
    checks = {
        "Server reachable": True,
        "Output pickle created": validation["file_exists"],
        f"Row count ({N_BIO + N_CHEM} expected)": validation["row_count"] == N_BIO + N_CHEM,
        "Required columns present": validation["has_required_cols"],
        "'complied' field populated": validation["complied_populated"],
    }

    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
        print("The Qwen3.5-122B judge is producing valid, parseable output.")
        print("Proceed to full evaluation: python evaluation/scripts/run_qwen_judge_full.py --rps 30")
        if not args.keep_output:
            shutil.rmtree(SANITY_OUTPUT_DIR, ignore_errors=True)
            print(f"\nSanity output cleaned up. Use --keep-output to retain.")
    else:
        print("RESULT: SOME CHECKS FAILED")
        print(f"Inspect output at: {output_dir}")
        print("Fix issues before running full evaluation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
