#!/usr/bin/env python3
"""
Verify that all evaluation results have no missing rows.
Reports which models/indices still have failed (empty) responses.
"""
import json
import sys
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
_eval_dir = _script_dir.parent
RESULTS_BASE = _eval_dir / "archived" / "eval_three_models_results"
SAMPLE_FILE = RESULTS_BASE / "sample_indices.json"

def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)
    expected = set(sample["chem"] + sample["bio"])
    print(f"Expected: {len(expected)} indices (25 chem + 25 bio)\n")

    all_ok = True
    for model_dir in sorted(RESULTS_BASE.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("cumulative", ".vscode") or model_dir.name.startswith("."):
            continue
        model_name = model_dir.name
        # Prefer {model_name}_safety_judge.pickle (from run_reeval), fallback to any main pickle
        main_pickle = model_dir / f"{model_name}_safety_judge.pickle"
        if not main_pickle.exists():
            pickles = [p for p in model_dir.glob("*safety_judge.pickle")
                       if "raw" not in p.name and "reeval" not in p.name]
            main_pickle = pickles[0] if pickles else None
        if main_pickle is None or not main_pickle.exists():
            print(f"{model_name}: No main safety_judge pickle found")
            all_ok = False
            continue

        df = pd.read_pickle(main_pickle)
        missing_idx = expected - set(df.index)
        if "raw_response" in df.columns:
            failed = df[df["raw_response"].isna()]
        else:
            failed = pd.DataFrame()
        failed_idx = set(failed.index) if len(failed) > 0 else set()

        status = "OK" if len(missing_idx) == 0 and len(failed_idx) == 0 else "INCOMPLETE"
        if status != "OK":
            all_ok = False
        print(f"{model_name}: {status}")
        print(f"  Rows: {len(df)}, expected indices: {len(expected)}")
        if missing_idx:
            print(f"  Missing indices: {sorted(missing_idx)}")
        if failed_idx:
            print(f"  Failed (no raw_response): {len(failed_idx)} rows, idx={sorted(failed_idx)}")
        print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
