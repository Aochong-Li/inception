#!/usr/bin/env python3
"""
Run StrongREJECT evaluation on the three-model eval results.

Loads existing preprocessed pickles (50 rows each) and runs SafetyEvaluator
with strongreject_only=True. Produces per-model strongreject results and a
cumulative strongreject_danger_scores.pickle.

Requires DEEPSEEK_API_KEY in environment or .env for API calls.
"""
import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent
sys.path.insert(0, str(_parent_dir))
sys.path.insert(0, str(_eval_dir))

# Import SafetyEvaluator via importlib to avoid vllm side-effects
spec = importlib.util.spec_from_file_location("safety_judge", _eval_dir / "safety-judge.py")
_safety_judge_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_safety_judge_mod)
SafetyEvaluator = _safety_judge_mod.SafetyEvaluator

from danger_score import compute_strongreject_score_row

RESULTS_BASE = _eval_dir / "eval_three_models_results"
SAMPLE_FILE = RESULTS_BASE / "sample_indices.json"
MODELS = ["DeepSeek-V3.2", "Qwen3-235B-A22B-Thinking-2507", "GLM-4.6"]


async def run_model(model_name: str, args) -> None:
    """Run StrongREJECT evaluation for a single model."""
    model_dir = RESULTS_BASE / model_name
    preprocessed_path = model_dir / f"{model_name}_preprocessed.pickle"

    if not preprocessed_path.exists():
        print(f"Skipping {model_name}: {preprocessed_path} not found")
        return

    df = pd.read_pickle(preprocessed_path)
    print(f"Loaded {preprocessed_path} ({len(df)} rows)")

    evaluator = SafetyEvaluator(
        input_df=df,
        inquiry_col="inquiry",
        reasoning_trace_col="reasoning_traces",
        category_col="category",
        chemical_label="wmdp-chem",
        bioterrorism_label="wmdp-bio",
        output_dir=str(model_dir),
        nick_name=f"{model_name}_strongreject",
        eval_model=args.eval_model,
        client_name=args.client_name,
        temperature=0.0,
        max_tokens=512,
        requests_per_second=args.rate_limit,
        max_validation_retries=3,
        strongreject_only=True,
    )

    await evaluator.run(overwrite=args.overwrite)
    print(f"StrongREJECT evaluation completed for {model_name}")


def aggregate(models: list[str]) -> pd.DataFrame:
    """Compute per-model StrongREJECT summary and save cumulative pickle."""
    rows = []
    for model_name in models:
        merged_path = RESULTS_BASE / model_name / f"{model_name}_strongreject_safety_judge.pickle"
        if not merged_path.exists():
            print(f"WARNING: {merged_path} not found, skipping aggregation for {model_name}")
            continue

        df = pd.read_pickle(merged_path)
        scores = df.apply(compute_strongreject_score_row, axis=1)

        complied_col = df["complied"] if "complied" in df.columns else pd.Series(dtype=bool)
        complied_bool = complied_col.dropna()
        compliance_rate = complied_bool.mean() if len(complied_bool) > 0 else float("nan")

        spec_col = pd.to_numeric(df.get("specificity"), errors="coerce")
        conv_col = pd.to_numeric(df.get("convincingness"), errors="coerce")

        rows.append({
            "model_name": model_name,
            "compliance_rate": round(compliance_rate, 3),
            "avg_specificity": round(spec_col.mean(), 3) if spec_col.notna().any() else None,
            "avg_convincingness": round(conv_col.mean(), 3) if conv_col.notna().any() else None,
            "strongreject_score": round(scores.dropna().mean(), 3) if scores.notna().any() else None,
            "sample_count": len(df),
        })

    summary_df = pd.DataFrame(rows)

    cumulative_dir = RESULTS_BASE / "cumulative"
    cumulative_dir.mkdir(parents=True, exist_ok=True)
    out_path = cumulative_dir / "strongreject_danger_scores.pickle"
    summary_df.to_pickle(out_path)
    print(f"\nSaved cumulative scores to {out_path}")
    return summary_df


async def main():
    parser = argparse.ArgumentParser(
        description="Run StrongREJECT evaluation on three target models"
    )
    parser.add_argument("--eval_model", default="deepseek-chat")
    parser.add_argument("--client_name", default="deepseek")
    parser.add_argument("--rate_limit", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SAMPLE_FILE.exists():
        print(f"ERROR: Sample file not found: {SAMPLE_FILE}")
        sys.exit(1)

    # Run evaluation for each model sequentially
    for model_name in MODELS:
        await run_model(model_name, args)

    # Aggregate results
    summary_df = aggregate(MODELS)

    # Print summary table
    print("\n" + "=" * 80)
    print("StrongREJECT Evaluation Summary")
    print("=" * 80)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No results to display.")


if __name__ == "__main__":
    asyncio.run(main())
