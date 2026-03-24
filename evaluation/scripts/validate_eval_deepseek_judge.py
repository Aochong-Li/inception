#!/usr/bin/env python3
"""
Thorough validation of evaluation/eval_deepseek_judge/ data.

Validates:
- Completeness: all expected model outputs exist with correct row counts
- Missing values: NaN, None, empty strings in critical columns
- Schema consistency: expected columns per branch
- Type checking: dtypes for numeric/boolean columns
- Cross-file consistency: raw category files sum to expected rows
- Index integrity: no gaps, sequential where expected

Usage:
  python evaluation/scripts/validate_eval_deepseek_judge.py
  python evaluation/scripts/validate_eval_deepseek_judge.py --json  # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Path setup
_script_dir = Path(__file__).parent.resolve()
_eval_dir = _script_dir.parent
EVAL_BASE = _eval_dir / "eval_deepseek_judge"
INCEPTION_DATA = _eval_dir / "inception_data"

# Expected row count per model (WMDP benchmark convention)
EXPECTED_ROWS = 800

# Critical columns that must not be entirely null (evaluation outputs)
CRITICAL_COLS = ["complied", "raw_response"]
# Columns that may have nulls but should be checked for empty-string coding
TEXT_COLS_TO_CHECK = ["raw_response", "response", "analysis"]

# Boolean columns (should be True/False/NaN)
BOOL_COLS = {"complied", "late_refusal"}
# Numeric columns (should be float/NaN)
NUMERIC_COLS = {"severity_level", "fatalities", "specificity", "convincingness", "material_access"}


def discover_model_dirs() -> list[tuple[str, Path]]:
    """Return (label, model_dir) for every model output directory."""
    results: list[tuple[str, Path]] = []
    if not EVAL_BASE.exists():
        return results
    for p in EVAL_BASE.rglob("*"):
        if not p.is_dir():
            continue
        # Model dir: contains {model}_safety_judge.pickle
        pickles = list(p.glob("*_safety_judge.pickle"))
        if not pickles:
            continue
        # Exclude raw category pickles (e.g. chemical_safety_judge_raw.pickle)
        model_pickles = [f for f in pickles if not f.name.startswith(("chemical_", "bioterrorism_", "strongreject_"))]
        if model_pickles:
            rel = p.relative_to(EVAL_BASE)
            results.append((str(rel), p))
    return sorted(results, key=lambda x: x[0])


def validate_single_model(label: str, model_dir: Path) -> dict[str, Any]:
    """Validate one model's output directory. Returns dict of checks and issues."""
    model_name = model_dir.name
    report: dict[str, Any] = {
        "label": label,
        "model": model_name,
        "path": str(model_dir),
        "ok": True,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    # 1. Final safety judge pickle
    final_pickle = model_dir / f"{model_name}_safety_judge.pickle"
    if not final_pickle.exists():
        report["ok"] = False
        report["issues"].append(f"Missing final output: {final_pickle.name}")
        return report

    try:
        df = pd.read_pickle(final_pickle)
    except Exception as e:
        report["ok"] = False
        report["issues"].append(f"Failed to load {final_pickle.name}: {e}")
        return report

    # 2. Row count
    n_rows = len(df)
    report["stats"]["rows"] = n_rows
    if n_rows != EXPECTED_ROWS:
        report["ok"] = False
        report["issues"].append(f"Row count {n_rows} != expected {EXPECTED_ROWS}")

    # 3. Index integrity
    if hasattr(df.index, "is_monotonic_increasing"):
        if not df.index.is_monotonic_increasing and len(df) > 1:
            report["warnings"].append("Index not monotonic")
    # Check for duplicate indices
    if df.index.duplicated().any():
        report["ok"] = False
        report["issues"].append(f"Duplicate indices: {df.index.duplicated().sum()}")

    # 4. Schema: required columns
    required = ["category", "inquiry", "reasoning_traces"]
    for col in required:
        if col not in df.columns:
            report["ok"] = False
            report["issues"].append(f"Missing required column: {col}")

    # 5. Missing values in critical columns
    for col in CRITICAL_COLS:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        empty_count = 0
        if df[col].dtype == object or (df[col].dtype == "string" if hasattr(pd, "StringDtype") else False):
            empty_count = (df[col].astype(str).str.strip() == "").sum()
        total_missing = null_count + empty_count
        report["stats"][f"{col}_null"] = int(null_count)
        report["stats"][f"{col}_empty"] = int(empty_count)
        if total_missing == n_rows and col in CRITICAL_COLS:
            report["ok"] = False
            report["issues"].append(f"Column '{col}' is entirely null/empty")
        elif total_missing > 0:
            report["warnings"].append(f"Column '{col}': {null_count} null, {empty_count} empty")

    # 6. Empty-string coding (per data-integrity-warnings.md)
    for col in TEXT_COLS_TO_CHECK:
        if col not in df.columns:
            continue
        if df[col].dtype != object:
            continue
        empty_str = (df[col].astype(str).str.strip() == "").sum()
        if empty_str > 0 and df[col].notna().sum() > 0:
            report["warnings"].append(f"Column '{col}': {empty_str} empty strings (not null-coded)")

    # 7. Type checking for boolean columns
    for col in BOOL_COLS:
        if col not in df.columns:
            continue
        valid = df[col].isin([True, False]) | df[col].isna()
        invalid = (~valid).sum()
        if invalid > 0:
            report["ok"] = False
            report["issues"].append(f"Column '{col}': {invalid} non-boolean values")
        report["stats"][f"{col}_valid"] = int(valid.sum())

    # 8. Type checking for numeric columns
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_numeric = df[col].notna() & numeric.isna()
            if non_numeric.sum() > 0:
                report["warnings"].append(f"Column '{col}': {non_numeric.sum()} non-numeric values")
        except Exception:
            pass

    # 9. Category coverage
    if "category" in df.columns:
        cats = df["category"].value_counts()
        report["stats"]["category_counts"] = cats.to_dict()
        if "wmdp-bio" in cats and "wmdp-chem" in cats:
            total_cat = cats["wmdp-bio"] + cats["wmdp-chem"]
            if total_cat != n_rows:
                report["warnings"].append(f"Category sum {total_cat} != rows {n_rows}")

    # 10. Raw category files consistency
    chem_raw = model_dir / "chemical_safety_judge_raw.pickle"
    bio_raw = model_dir / "bioterrorism_safety_judge_raw.pickle"
    raw_rows = 0
    if chem_raw.exists():
        try:
            cdf = pd.read_pickle(chem_raw)
            raw_rows += len(cdf)
            if "raw_response" in cdf.columns:
                null_resp = cdf["raw_response"].isna().sum()
                if null_resp > 0:
                    report["warnings"].append(f"chemical raw: {null_resp} null raw_response")
        except Exception as e:
            report["warnings"].append(f"chemical raw load error: {e}")
    if bio_raw.exists():
        try:
            bdf = pd.read_pickle(bio_raw)
            raw_rows += len(bdf)
            if "raw_response" in bdf.columns:
                null_resp = bdf["raw_response"].isna().sum()
                if null_resp > 0:
                    report["warnings"].append(f"bioterrorism raw: {null_resp} null raw_response")
        except Exception as e:
            report["warnings"].append(f"bioterrorism raw load error: {e}")
    if chem_raw.exists() or bio_raw.exists():
        report["stats"]["raw_total_rows"] = raw_rows
        if raw_rows != n_rows:
            report["warnings"].append(f"Raw category rows {raw_rows} != final rows {n_rows}")

    # 11. Preprocessed pickle exists and matches
    # simple_inject uses 'response' column, not iterations — reasoning_traces are expected empty
    is_simple_inject = "simple_inject" in label
    preproc = model_dir / f"{model_name}_preprocessed.pickle"
    if preproc.exists() and not is_simple_inject:
        try:
            pdf = pd.read_pickle(preproc)
            if "reasoning_traces" in pdf.columns:
                null_traces = pdf["reasoning_traces"].isna().sum()
                empty_traces = (pdf["reasoning_traces"].astype(str).str.strip() == "").sum()
                if null_traces + empty_traces == len(pdf):
                    report["warnings"].append("Preprocessed: all reasoning_traces null/empty")
        except Exception as e:
            report["warnings"].append(f"Preprocessed load error: {e}")

    return report


def build_expected_jobs() -> set[str]:
    """Build set of expected (label/model) from inception_data structure."""
    expected: set[str] = set()
    if not INCEPTION_DATA.exists():
        return expected
    for pkl in INCEPTION_DATA.rglob("*.pickle"):
        if "api" in pkl.parts or any("checkpoint" in str(p).lower() for p in pkl.parts):
            continue
        stem = pkl.stem
        # Map path to eval_deepseek_judge subpath
        try:
            rel = pkl.relative_to(INCEPTION_DATA)
            parts = list(rel.parts)
            if len(parts) >= 2:
                if "max_iterations_5" in parts[0]:
                    subpath = f"{parts[0]}/{parts[1]}/{stem}"
                elif "max_iterations_1" in parts[0] and "architect_initial_max_tokens" in str(rel):
                    tok = parts[2].replace("architect_initial_max_tokens_", "tokens_")
                    subpath = f"ablation/think/{tok}/{stem}"
                elif "simple_inject" in parts[0]:
                    subpath = f"{parts[0]}/{parts[1]}/{stem}"
                else:
                    continue
                expected.add(subpath)
        except ValueError:
            continue
    return expected


def deep_scan_missing_values() -> dict[str, Any]:
    """
    Load every *_safety_judge.pickle and compute missing-value statistics.
    Returns aggregate report of null/empty counts per column across all models.
    """
    model_dirs = discover_model_dirs()
    all_null_counts: dict[str, dict[str, int]] = {}
    all_empty_counts: dict[str, dict[str, int]] = {}
    columns_seen: set[str] = set()

    for label, model_dir in model_dirs:
        model_name = model_dir.name
        final_pickle = model_dir / f"{model_name}_safety_judge.pickle"
        if not final_pickle.exists():
            continue
        try:
            df = pd.read_pickle(final_pickle)
        except Exception:
            continue
        columns_seen.update(df.columns.tolist())
        for col in df.columns:
            if col not in all_null_counts:
                all_null_counts[col] = {}
                all_empty_counts[col] = {}
            nulls = df[col].isna().sum()
            all_null_counts[col][label] = int(nulls)
            empty = 0
            if df[col].dtype == object:
                # Non-null but empty string
                empty = int((df[col].notna() & (df[col].astype(str).str.strip() == "")).sum())
            all_empty_counts[col][label] = int(empty)
    return {
        "columns": sorted(columns_seen),
        "null_counts": all_null_counts,
        "empty_counts": all_empty_counts,
        "n_models": len(model_dirs),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate eval_deepseek_judge data completeness and integrity")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--deep", action="store_true", help="Run deep missing-value scan across all pickles")
    args = parser.parse_args()

    model_dirs = discover_model_dirs()
    if not model_dirs:
        print("No model directories found in eval_deepseek_judge", file=sys.stderr)
        sys.exit(1)

    if args.deep:
        deep = deep_scan_missing_values()
        if args.json:
            print(json.dumps(deep, indent=2))
        else:
            print("DEEP MISSING-VALUE SCAN")
            print("=" * 60)
            critical = ["complied", "raw_response", "response", "category", "inquiry"]
            for col in critical:
                if col not in deep["null_counts"]:
                    continue
                nc = deep["null_counts"][col]
                ec = deep["empty_counts"][col]
                max_null = max(nc.values()) if nc else 0
                max_empty = max(ec.values()) if ec else 0
                models_with_null = sum(1 for v in nc.values() if v > 0)
                models_with_empty = sum(1 for v in ec.values() if v > 0)
                print(f"  {col}: max_null={max_null} (in {models_with_null} models), max_empty={max_empty} (in {models_with_empty} models)")
            sys.exit(0)

    reports: list[dict[str, Any]] = []
    for label, model_dir in model_dirs:
        r = validate_single_model(label, model_dir)
        if args.strict and r.get("warnings"):
            r["ok"] = False
        reports.append(r)

    failed = [r for r in reports if not r["ok"]]
    n_ok = len(reports) - len(failed)

    if args.json:
        out = {
            "summary": {
                "total": len(reports),
                "ok": n_ok,
                "failed": len(failed),
                "expected_rows_per_model": EXPECTED_ROWS,
            },
            "reports": reports,
        }
        print(json.dumps(out, indent=2))
        sys.exit(1 if failed else 0)

    # Human-readable output
    print("=" * 60)
    print("eval_deepseek_judge Data Validation Report")
    print("=" * 60)
    print(f"Models checked: {len(reports)}")
    print(f"OK: {n_ok} | Failed: {len(failed)}")
    print(f"Expected rows per model: {EXPECTED_ROWS}")
    print()

    if failed:
        print("FAILED MODELS:")
        for r in failed:
            print(f"  [{r['label']}]")
            for issue in r["issues"]:
                print(f"    - {issue}")
        print()

    # Summary of warnings
    warned = [r for r in reports if r.get("warnings")]
    if warned:
        print("WARNINGS:")
        for r in warned:
            print(f"  [{r['label']}]")
            for w in r["warnings"]:
                print(f"    - {w}")
        print()

    # Row count summary
    print("ROW COUNTS:")
    for r in reports:
        rows = r.get("stats", {}).get("rows", "?")
        status = "OK" if r["ok"] else "FAIL"
        print(f"  {r['label']}: {rows} rows [{status}]")

    # Missing value summary
    print()
    print("CRITICAL COLUMN NULL/EMPTY SUMMARY:")
    for r in reports:
        s = r.get("stats", {})
        raw_null = s.get("raw_response_null", 0)
        raw_empty = s.get("raw_response_empty", 0)
        complied_null = s.get("complied_null", 0)
        if raw_null or raw_empty or complied_null:
            print(f"  {r['label']}: raw_response null={raw_null} empty={raw_empty}, complied null={complied_null}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
