#!/usr/bin/env python3
"""
Verify that all evaluation results have no missing rows.
Reports which models/indices still have failed (empty) responses.

Supports both single-directory and multi-judge directory layouts:
  python verify_eval_completeness.py                              # default: eval_three_models_results/
  python verify_eval_completeness.py --results-dir eval_five_judges/claude-sonnet-4.6
  python verify_eval_completeness.py --results-dir eval_five_judges --all-judges
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
DEFAULT_RESULTS = _script_dir / "eval_three_models_results"

# Expected evaluation columns produced by SafetyEvaluator._parse_json_responses
EXPECTED_EVAL_COLS = {
    "error", "retries", "raw_response", "analysis",
    "complied", "late_refusal",
    "specificity", "convincingness", "severity_level", "fatalities",
    "delivery_method",
}
BIO_COLS = {"threat", "contagiousness", "route", "scale"}
CHEM_COLS = {"agent_class", "agent_grade", "exposure_environment"}


def _find_sample_file(results_dir: Path) -> Path:
    """Locate sample_indices.json — check results_dir, then fallback to default."""
    candidate = results_dir / "sample_indices.json"
    if candidate.exists():
        return candidate
    fallback = DEFAULT_RESULTS / "sample_indices.json"
    if fallback.exists():
        return fallback
    print(f"ERROR: sample_indices.json not found in {results_dir} or {DEFAULT_RESULTS}")
    sys.exit(1)


def verify_directory(results_dir: Path, expected: set, label: str = "") -> bool:
    """Verify all model results under a single results directory.

    Returns True if all models pass, False otherwise.
    """
    prefix = f"[{label}] " if label else ""
    all_ok = True

    model_dirs = [
        d for d in sorted(results_dir.iterdir())
        if d.is_dir() and d.name not in ("cumulative", "_preprocessed", ".ipynb_checkpoints")
    ]

    if not model_dirs:
        print(f"{prefix}No model directories found in {results_dir}")
        return False

    for model_dir in model_dirs:
        model_name = model_dir.name
        main_pickle = model_dir / f"{model_name}_safety_judge.pickle"
        if not main_pickle.exists():
            pickles = [
                p for p in model_dir.glob("*safety_judge.pickle")
                if "raw" not in p.name and "reeval" not in p.name
            ]
            main_pickle = pickles[0] if pickles else None

        if main_pickle is None or not main_pickle.exists():
            print(f"{prefix}{model_name}: No main safety_judge pickle found")
            all_ok = False
            continue

        df = pd.read_pickle(main_pickle)

        # --- Row completeness ---
        missing_idx = expected - set(df.index)
        if "raw_response" in df.columns:
            failed = df[df["raw_response"].isna()]
        else:
            failed = pd.DataFrame()
        failed_idx = set(failed.index) if len(failed) > 0 else set()

        # --- Column schema ---
        cols = set(df.columns)
        missing_eval_cols = EXPECTED_EVAL_COLS - cols

        # Category-specific column checks
        bio_rows = df[df["category"] == "wmdp-bio"] if "category" in df.columns else pd.DataFrame()
        chem_rows = df[df["category"] == "wmdp-chem"] if "category" in df.columns else pd.DataFrame()
        missing_bio = BIO_COLS - cols if len(bio_rows) > 0 else set()
        missing_chem = CHEM_COLS - cols if len(chem_rows) > 0 else set()

        # --- Value range checks ---
        range_issues = []
        for col, lo, hi in [("specificity", 1, 5), ("convincingness", 1, 5), ("severity_level", 1, 10)]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                oob = vals[(vals < lo) | (vals > hi)]
                if len(oob) > 0:
                    range_issues.append(f"{col}: {len(oob)} values outside [{lo},{hi}]")

        # --- Verdict ---
        issues = []
        if missing_idx:
            issues.append(f"missing {len(missing_idx)} indices")
        if failed_idx:
            issues.append(f"{len(failed_idx)} failed rows")
        if missing_eval_cols:
            issues.append(f"missing cols: {sorted(missing_eval_cols)}")
        if missing_bio:
            issues.append(f"missing bio cols: {sorted(missing_bio)}")
        if missing_chem:
            issues.append(f"missing chem cols: {sorted(missing_chem)}")
        issues.extend(range_issues)

        status = "OK" if not issues else "INCOMPLETE"
        if status != "OK":
            all_ok = False

        print(f"{prefix}{model_name}: {status}  ({len(df)} rows, {len(df.columns)} cols)")
        for issue in issues:
            print(f"  {issue}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify evaluation result completeness and schema"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing per-model result subdirectories (default: eval_three_models_results/)",
    )
    parser.add_argument(
        "--all-judges",
        action="store_true",
        help="Treat --results-dir as the parent of multiple judge directories and verify each",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS

    sample_file = _find_sample_file(results_dir)
    with open(sample_file) as f:
        sample = json.load(f)
    expected = set(sample["chem"] + sample["bio"])
    print(f"Expected: {len(expected)} indices (25 chem + 25 bio)")
    print(f"Sample file: {sample_file}\n")

    all_ok = True

    if args.all_judges:
        judge_dirs = [
            d for d in sorted(results_dir.iterdir())
            if d.is_dir() and d.name not in ("_preprocessed", ".ipynb_checkpoints")
        ]
        if not judge_dirs:
            print(f"No judge directories found in {results_dir}")
            sys.exit(1)
        for judge_dir in judge_dirs:
            print(f"{'='*60}")
            print(f"Judge: {judge_dir.name}")
            print(f"{'='*60}")
            ok = verify_directory(judge_dir, expected, label=judge_dir.name)
            if not ok:
                all_ok = False
            print()
    else:
        all_ok = verify_directory(results_dir, expected)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
