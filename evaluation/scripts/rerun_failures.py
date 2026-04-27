"""
Rerun the broken rows in evaluation/rerun_failures/{deepseek,qwen}_failures.pickle
through the SafetyEvaluator using a hosted API (no SGLang).

For each (judge, manifest):
  1. Load manifest dataframe (has inquiry, reasoning_traces, category, source_pickle, source_idx, fail_kind)
  2. Run SafetyEvaluator on a tmp output dir
  3. Re-attach source_pickle + source_idx columns via the dataframe index
  4. Save merged result to evaluation/rerun_failures/rerun_results/{ds,qw}_rerun.pickle

Usage:
  python evaluation/scripts/rerun_failures.py --judge ds   # DeepSeek API on 15 rows
  python evaluation/scripts/rerun_failures.py --judge qw   # DeepInfra Qwen on 648 rows
  python evaluation/scripts/rerun_failures.py --judge all  # both, sequentially
"""
import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
RERUN_DIR = EVAL_DIR / "rerun_failures"
RESULTS_DIR = RERUN_DIR / "rerun_results"
LOGS_DIR = RESULTS_DIR / "logs"

load_dotenv(PROJECT_ROOT / "core" / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

# Load SafetyEvaluator
spec = importlib.util.spec_from_file_location("safety_judge", EVAL_DIR / "safety-judge.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SafetyEvaluator = mod.SafetyEvaluator

JUDGES = {
    "ds": dict(
        manifest=RERUN_DIR / "deepseek_failures.pickle",
        out=RESULTS_DIR / "deepseek_rerun.pickle",
        tmp_out=RESULTS_DIR / "_tmp_ds",
        eval_model="deepseek-chat",
        client_name="deepseek",
        max_tokens=8192,
        rps=10.0,
        nick="rerun_DS",
    ),
    "qw": dict(
        manifest=RERUN_DIR / "qwen_failures.pickle",
        out=RESULTS_DIR / "qwen_rerun.pickle",
        tmp_out=RESULTS_DIR / "_tmp_qw",
        eval_model="Qwen/Qwen3.5-397B-A17B",
        client_name="deepinfra",
        max_tokens=81920,
        rps=20.0,
        nick="rerun_QW",
    ),
}


async def run_one(label: str, cfg: dict):
    print(f"\n===== Rerun: {label} =====")
    df = pd.read_pickle(cfg["manifest"])
    print(f"Manifest:  {cfg['manifest'].relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print(f"Judge:     {cfg['eval_model']} via {cfg['client_name']} | rps={cfg['rps']} max_tokens={cfg['max_tokens']}")
    print(f"Output:    {cfg['out'].relative_to(PROJECT_ROOT)}")

    cfg["tmp_out"].mkdir(parents=True, exist_ok=True)

    # Preserve mapping columns; SafetyEvaluator should pass them through unchanged.
    # Input must have inquiry, reasoning_traces, category cols.
    input_df = df.reset_index(drop=True).copy()  # judge will set its own integer index

    evaluator = SafetyEvaluator(
        input_df=input_df,
        inquiry_col="inquiry",
        reasoning_trace_col="reasoning_traces",
        category_col="category",
        chemical_label="wmdp-chem",
        bioterrorism_label="wmdp-bio",
        output_dir=str(cfg["tmp_out"]),
        nick_name=cfg["nick"],
        eval_model=cfg["eval_model"],
        client_name=cfg["client_name"],
        temperature=0.0,
        max_tokens=cfg["max_tokens"],
        requests_per_second=cfg["rps"],
        max_validation_retries=3,
        max_consecutive_refusals=0,
        strongreject_only=False,
    )
    result = await evaluator.run()
    if result is None or len(result) == 0:
        print(f"  ✗ Evaluator returned empty result")
        return

    # Re-attach source_pickle + source_idx + fail_kind by aligning on the manifest order.
    # The evaluator returns one row per input row in the same order (after concat of bio+chem).
    # Safer: merge by matching (inquiry + first 200 chars of reasoning_traces) - but rely on row identity since input_df was already split inside the evaluator only by category.
    # The cleanest approach: re-attach via the manifest, joining on (inquiry, source_idx) won't work for the manifest cols.
    # Instead: SafetyEvaluator preserves all input columns, so source_pickle/source_idx/fail_kind should be in `result` already.
    keep_cols_present = [c for c in ("source_pickle", "source_idx", "fail_kind") if c in result.columns]
    print(f"  Result has {len(result)} rows | mapping cols carried through: {keep_cols_present}")
    if not keep_cols_present:
        print("  ⚠ mapping cols missing in result — falling back to manifest re-attach by row order")
        # The evaluator may concat bio/chem in the order it processed them. Try merging on inquiry+reasoning_traces hash.
        result = result.reset_index(drop=True)
        # Align by hashing inquiry+reasoning_traces+category — uniquely identifies each manifest row
        keymap = {(r["inquiry"], r["reasoning_traces"], r["category"]): (r["source_pickle"], int(r["source_idx"]), r["fail_kind"])
                  for _, r in input_df.iterrows()}
        sp, si, fk = [], [], []
        for _, r in result.iterrows():
            k = (r["inquiry"], r["reasoning_traces"], r["category"])
            mp = keymap.get(k, (None, None, None))
            sp.append(mp[0]); si.append(mp[1]); fk.append(mp[2])
        result["source_pickle"] = sp
        result["source_idx"]   = si
        result["fail_kind"]    = fk

    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(cfg["out"])
    print(f"  ✓ Saved {len(result)} rows -> {cfg['out'].relative_to(PROJECT_ROOT)}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", choices=["ds", "qw", "all"], required=True)
    args = p.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    targets = ["ds", "qw"] if args.judge == "all" else [args.judge]
    for label in targets:
        await run_one(label, JUDGES[label])


if __name__ == "__main__":
    asyncio.run(main())
