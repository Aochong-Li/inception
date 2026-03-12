#!/usr/bin/env python3
"""
Comprehensive analysis of the five-judges evaluation data.
Checks file existence, row counts, and per-field completeness
for every judge x target model combination.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict

BASE = Path(__file__).parent.parent / "archived" / "eval_five_judges"

JUDGES = [
    "anthropic_claude-sonnet-4.5",
    "anthropic_claude-sonnet-4.6",
    "moonshotai_kimi-k2.5",
    "openai_gpt-5.2",
    "z-ai_glm-5",
]

TARGETS = [
    "DeepSeek-V3.2",
    "GLM-4.6",
    "Qwen3-235B-A22B-Thinking-2507",
]

# Evaluation columns we care about
EVAL_COLUMNS_SHARED = [
    "complied", "late_refusal", "specificity", "convincingness",
    "severity_level", "fatalities", "analysis",
]

EVAL_COLUMNS_BIO = [
    "threat", "contagiousness", "route", "scale",
]

EVAL_COLUMNS_CHEM = [
    "agent_class", "agent_grade", "exposure_environment",
]

# Some files may use delivery_method or dispersal_method
DELIVERY_COLS = ["delivery_method", "dispersal_method"]

RAW_COLUMNS = ["idx", "response", "error", "retries"]


def is_empty(val):
    """Check if a value is null, NaN, None, or empty string."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def count_empty(series):
    """Count empty values in a pandas Series."""
    return series.apply(is_empty).sum()


def load_pickle(path):
    """Load a pickle file safely."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            return data
        else:
            return None
    except Exception as e:
        return f"ERROR: {e}"


def analyze_safety_judge(df, label=""):
    """Analyze a safety_judge DataFrame for completeness."""
    results = OrderedDict()
    results["rows"] = len(df)
    results["columns"] = len(df.columns)

    # Determine which columns exist
    all_eval_cols = EVAL_COLUMNS_SHARED + EVAL_COLUMNS_BIO + EVAL_COLUMNS_CHEM

    # Check for delivery/dispersal method
    delivery_col = None
    for col in DELIVERY_COLS:
        if col in df.columns:
            delivery_col = col
            break

    if delivery_col:
        all_eval_cols.append(delivery_col)

    # Also check for raw_response and error columns
    extra_cols = ["raw_response", "error", "retries"]

    field_stats = OrderedDict()
    for col in all_eval_cols + extra_cols:
        if col in df.columns:
            total = len(df)
            empty = count_empty(df[col])
            filled = total - empty
            pct = (filled / total * 100) if total > 0 else 0
            field_stats[col] = {
                "total": total,
                "filled": filled,
                "empty": empty,
                "pct_complete": round(pct, 1),
            }
        else:
            field_stats[col] = {"total": len(df), "filled": 0, "empty": len(df), "pct_complete": 0, "note": "COLUMN MISSING"}

    results["field_stats"] = field_stats
    results["all_columns"] = list(df.columns)
    return results


def analyze_raw_pickle(df, label=""):
    """Analyze a raw (bio/chem) pickle for completeness."""
    results = OrderedDict()
    results["rows"] = len(df)
    results["columns"] = len(df.columns)

    field_stats = OrderedDict()
    for col in df.columns:
        total = len(df)
        empty = count_empty(df[col])
        filled = total - empty
        pct = (filled / total * 100) if total > 0 else 0
        field_stats[col] = {
            "total": total,
            "filled": filled,
            "empty": empty,
            "pct_complete": round(pct, 1),
        }

    results["field_stats"] = field_stats
    return results


def print_separator(char="=", width=120):
    print(char * width)


def main():
    print_separator()
    print("FIVE JUDGES EVALUATION DATA -- COMPREHENSIVE COMPLETENESS ANALYSIS")
    print_separator()
    print()

    # ====================================================================
    # SECTION 1: File existence matrix
    # ====================================================================
    print("SECTION 1: FILE EXISTENCE MATRIX")
    print_separator("-")
    print()

    file_types = [
        "safety_judge.pickle",
        "bioterrorism_safety_judge_raw.pickle",
        "chemical_safety_judge_raw.pickle",
        "preprocessed.pickle",
    ]

    # Header
    print(f"{'Judge':<35} {'Target':<40} {'safety_judge':<15} {'bio_raw':<15} {'chem_raw':<15} {'preprocessed':<15}")
    print("-" * 135)

    for judge in JUDGES:
        for target in TARGETS:
            tdir = BASE / judge / target
            sj = tdir / f"{target}_safety_judge.pickle"
            bio = tdir / "bioterrorism_safety_judge_raw.pickle"
            chem = tdir / "chemical_safety_judge_raw.pickle"
            prep = tdir / f"{target}_preprocessed.pickle"

            sj_exists = "YES" if sj.exists() else "MISSING"
            bio_exists = "YES" if bio.exists() else "MISSING"
            chem_exists = "YES" if chem.exists() else "MISSING"
            prep_exists = "YES" if prep.exists() else "MISSING"

            print(f"{judge:<35} {target:<40} {sj_exists:<15} {bio_exists:<15} {chem_exists:<15} {prep_exists:<15}")
        print()

    # Cumulative files
    print()
    print("CUMULATIVE FILES:")
    print(f"{'Judge':<35} {'danger_scores':<20} {'bio_evaluator':<20} {'chem_evaluator':<20}")
    print("-" * 95)
    for judge in JUDGES:
        cdir = BASE / judge / "cumulative"
        ds = "YES" if (cdir / "danger_scores.pickle").exists() else "MISSING"
        bio = "YES" if (cdir / "bio_evaluator_results.pickle").exists() else "MISSING"
        chem = "YES" if (cdir / "chem_evaluator_results.pickle").exists() else "MISSING"
        print(f"{judge:<35} {ds:<20} {bio:<20} {chem:<20}")

    print()
    print()

    # ====================================================================
    # SECTION 2: Safety Judge Pickle -- Detailed Field Completeness
    # ====================================================================
    print_separator()
    print("SECTION 2: SAFETY JUDGE PICKLE -- DETAILED FIELD COMPLETENESS")
    print_separator()
    print()

    all_results = {}

    for judge in JUDGES:
        for target in TARGETS:
            sj_path = BASE / judge / target / f"{target}_safety_judge.pickle"
            key = f"{judge} / {target}"

            if not sj_path.exists():
                print(f"[MISSING] {key}: {sj_path}")
                print()
                all_results[key] = None
                continue

            df = load_pickle(sj_path)
            if isinstance(df, str):  # error message
                print(f"[ERROR] {key}: {df}")
                print()
                all_results[key] = None
                continue

            analysis = analyze_safety_judge(df, key)
            all_results[key] = analysis

            print(f"  JUDGE: {judge}")
            print(f"  TARGET: {target}")
            print(f"  ROWS: {analysis['rows']}  |  COLUMNS: {analysis['columns']}")
            print(f"  ALL COLUMNS: {analysis['all_columns']}")
            print()

            # Print field stats table
            print(f"    {'Field':<30} {'Total':<8} {'Filled':<8} {'Empty':<8} {'% Complete':<12} {'Note':<20}")
            print("    " + "-" * 86)

            for field, stats in analysis["field_stats"].items():
                note = stats.get("note", "")
                pct_str = f"{stats['pct_complete']}%"
                if stats["pct_complete"] < 100:
                    pct_str += " ***"
                print(f"    {field:<30} {stats['total']:<8} {stats['filled']:<8} {stats['empty']:<8} {pct_str:<12} {note:<20}")

            print()
            print_separator("-", 100)
            print()

    # ====================================================================
    # SECTION 3: Raw Bio/Chem Pickle Completeness
    # ====================================================================
    print_separator()
    print("SECTION 3: RAW BIO/CHEM PICKLE COMPLETENESS")
    print_separator()
    print()

    for judge in JUDGES:
        for target in TARGETS:
            for raw_type in ["bioterrorism", "chemical"]:
                raw_path = BASE / judge / target / f"{raw_type}_safety_judge_raw.pickle"
                key = f"{judge} / {target} / {raw_type}"

                if not raw_path.exists():
                    print(f"  [MISSING] {key}")
                    continue

                df = load_pickle(raw_path)
                if isinstance(df, str):
                    print(f"  [ERROR] {key}: {df}")
                    continue

                analysis = analyze_raw_pickle(df, key)
                print(f"  {key}")
                print(f"    Rows: {analysis['rows']}  |  Columns: {analysis['columns']}")

                for field, stats in analysis["field_stats"].items():
                    flag = "" if stats["pct_complete"] == 100 else " ***INCOMPLETE***"
                    print(f"      {field:<25} {stats['filled']}/{stats['total']}  ({stats['pct_complete']}% complete){flag}")
                print()

    # ====================================================================
    # SECTION 4: Cumulative Danger Scores
    # ====================================================================
    print_separator()
    print("SECTION 4: CUMULATIVE DANGER SCORES & EVALUATOR RESULTS")
    print_separator()
    print()

    for judge in JUDGES:
        cdir = BASE / judge / "cumulative"
        print(f"  JUDGE: {judge}")

        for fname in ["danger_scores.pickle", "bio_evaluator_results.pickle", "chem_evaluator_results.pickle"]:
            fpath = cdir / fname
            if not fpath.exists():
                print(f"    [{fname}] MISSING")
                continue

            df = load_pickle(fpath)
            if isinstance(df, str):
                print(f"    [{fname}] ERROR: {df}")
                continue

            if df is None:
                print(f"    [{fname}] Not a DataFrame")
                continue

            print(f"    [{fname}] Rows: {len(df)}  |  Columns: {len(df.columns)}")
            print(f"      Columns: {list(df.columns)}")

            # Show completeness for all columns
            for col in df.columns:
                total = len(df)
                empty = count_empty(df[col])
                filled = total - empty
                pct = round(filled / total * 100, 1) if total > 0 else 0
                flag = "" if pct == 100 else " ***"
                print(f"        {col:<35} {filled}/{total}  ({pct}%){flag}")
            print()

        print_separator("-", 100)
        print()

    # ====================================================================
    # SECTION 5: SUMMARY MATRIX -- Eval field completeness across all judges
    # ====================================================================
    print_separator()
    print("SECTION 5: SUMMARY MATRIX -- % COMPLETE FOR KEY EVALUATION FIELDS")
    print_separator()
    print()

    # Determine all fields across all loaded DFs
    key_fields = (
        EVAL_COLUMNS_SHARED + EVAL_COLUMNS_BIO + EVAL_COLUMNS_CHEM +
        ["delivery_method", "dispersal_method", "raw_response", "error"]
    )

    # Build summary table
    summary_rows = []
    for judge in JUDGES:
        for target in TARGETS:
            key = f"{judge} / {target}"
            analysis = all_results.get(key)
            row = {"judge": judge, "target": target}

            if analysis is None:
                for f in key_fields:
                    row[f] = "N/A"
                row["rows"] = "N/A"
            else:
                row["rows"] = analysis["rows"]
                for f in key_fields:
                    if f in analysis["field_stats"]:
                        row[f] = f"{analysis['field_stats'][f]['pct_complete']}%"
                    else:
                        row[f] = "-"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print with nice formatting
    # First, filter out columns that are all "-" or "N/A"
    display_fields = ["rows"]
    for f in key_fields:
        vals = summary_df[f].unique()
        if not all(v == "-" for v in vals):
            display_fields.append(f)

    print(f"{'Judge':<35} {'Target':<25} ", end="")
    for f in display_fields:
        print(f"{f:<16}", end="")
    print()
    print("-" * (60 + 16 * len(display_fields)))

    for _, row in summary_df.iterrows():
        print(f"{row['judge']:<35} {row['target']:<25} ", end="")
        for f in display_fields:
            val = str(row[f])
            print(f"{val:<16}", end="")
        print()

    print()
    print_separator()
    print("ANALYSIS COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()
