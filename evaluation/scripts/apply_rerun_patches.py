"""
Apply rerun results back to the original *_safety_judge.pickle files.

For each row in the rerun pickle:
  1. Load the source pickle (path from row.source_pickle)
  2. Sanity-check: source row at source_idx has the SAME inquiry+reasoning_traces
     as the rerun row (else abort that row).
  3. Overwrite only the judge-output columns at source_idx; everything else
     (target metadata, original answer/question/choices, etc.) is left intact.
  4. Save the source pickle back.

Default: deepseek rerun. Use --judge to switch.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"

# Columns that the judge produces — these are safe to overwrite.
JUDGE_OUTPUT_COLS = {
    # Raw response / metadata from the API call
    "raw_response", "response", "error", "retries", "finish_reason",
    # Parsed JSON fields
    "analysis", "agent_class", "agent_grade",
    "delivery_method", "exposure_environment",
    "severity_level", "fatalities",
    "complied", "late_refusal",
    "specificity", "convincingness",
    "truncated",
    "threat", "contagiousness", "route", "scale",
    # Newer-schema mirror fields (some pickles have them, some don't)
    "response_eval", "error_eval", "retries_eval", "finish_reason_eval",
}


def apply_one(rerun_path: Path, dry_run: bool):
    rerun = pd.read_pickle(rerun_path)
    print(f"Loaded {len(rerun)} rerun rows from {rerun_path.relative_to(PROJECT_ROOT)}")
    if "source_pickle" not in rerun.columns or "source_idx" not in rerun.columns:
        print("  ERROR: rerun pickle missing source_pickle or source_idx columns")
        return

    by_source = rerun.groupby("source_pickle")
    total_patched = 0
    total_skipped = 0
    print()
    for src_rel, group in by_source:
        src_path = PROJECT_ROOT / src_rel
        if not src_path.exists():
            print(f"  ✗ MISSING SOURCE: {src_rel}")
            total_skipped += len(group)
            continue
        src_df = pd.read_pickle(src_path)

        cols_to_patch = sorted(set(rerun.columns) & set(src_df.columns) & JUDGE_OUTPUT_COLS)
        rows_patched = 0
        rows_skipped = 0
        for _, rrow in group.iterrows():
            idx = int(rrow["source_idx"])
            if idx not in src_df.index:
                rows_skipped += 1
                continue
            # Sanity: inquiry + reasoning_traces must match
            same_inq = (src_df.loc[idx, "inquiry"] == rrow["inquiry"])
            same_trc = (src_df.loc[idx, "reasoning_traces"] == rrow["reasoning_traces"])
            if not (same_inq and same_trc):
                print(f"    ✗ idx={idx} mismatch (inq={same_inq} trc={same_trc}) — skipped")
                rows_skipped += 1
                continue
            for col in cols_to_patch:
                src_df.at[idx, col] = rrow[col]
            rows_patched += 1
        if not dry_run and rows_patched > 0:
            src_df.to_pickle(src_path)
        marker = "DRY-RUN" if dry_run else "WROTE"
        print(f"  [{marker}] {src_rel}: {rows_patched} patched, {rows_skipped} skipped, "
              f"cols={cols_to_patch}")
        total_patched += rows_patched
        total_skipped += rows_skipped

    print()
    print(f"=== {'DRY-RUN' if dry_run else 'APPLIED'} ===  "
          f"patched={total_patched}  skipped={total_skipped}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", choices=["ds", "qw"], default="ds")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rerun_path = EVAL_DIR / "rerun_failures" / "rerun_results" / (
        "deepseek_rerun.pickle" if args.judge == "ds" else "qwen_rerun.pickle"
    )
    if not rerun_path.exists():
        print(f"ERROR: rerun pickle not found: {rerun_path}")
        sys.exit(1)
    apply_one(rerun_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
