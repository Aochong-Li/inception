#!/usr/bin/env python3
"""
StrongREJECT Demo: Safety Evaluation Pipeline

Demonstrates the full evaluation pipeline with:
- 30 prompts (15 bio, 15 chem)
- Safety evaluation with specificity/convincingness scoring
- Per-iteration analysis
- StrongREJECT composite score aggregation
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

import pandas as pd

# Setup paths
DEMO_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "core" / ".env")


def sample_demo_data(input_path: Path, n_bio: int = 15, n_chem: int = 15, seed: int = 42) -> pd.DataFrame:
    """Sample balanced demo data from the full dataset."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_pickle(input_path)

    # Sample from each category
    bio_df = df[df['category'] == 'wmdp-bio'].sample(n=n_bio, random_state=seed)
    chem_df = df[df['category'] == 'wmdp-chem'].sample(n=n_chem, random_state=seed)

    # Combine and reset index
    demo_df = pd.concat([bio_df, chem_df], ignore_index=False)

    logger.info(f"Sampled {len(demo_df)} rows: {n_bio} bio, {n_chem} chem")
    return demo_df


async def run_safety_evaluation(demo_df: pd.DataFrame, output_dir: Path, model_name: str = "demo_model"):
    """Run safety evaluation using SafetyEvaluator."""
    import importlib.util

    # Import SafetyEvaluator from safety-judge.py (hyphenated filename)
    spec = importlib.util.spec_from_file_location(
        "safety_judge",
        PROJECT_ROOT / "evaluation" / "safety-judge.py"
    )
    safety_judge_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safety_judge_module)
    SafetyEvaluator = safety_judge_module.SafetyEvaluator

    # Create output directory for this model
    model_output_dir = output_dir / "safety_evaluation" / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running safety evaluation for {model_name}...")

    evaluator = SafetyEvaluator(
        input_df=demo_df,
        inquiry_col='inquiry',
        reasoning_trace_col='reasoning_traces',
        category_col='category',
        chemical_label='wmdp-chem',
        cybersecurity_label='wmdp-cyber',  # Not used but required
        bioterrorism_label='wmdp-bio',
        output_dir=str(model_output_dir),
        nick_name=model_name,
        eval_model='gpt-4o-mini',
        client_name='openai',
        temperature=0.0,
        max_tokens=256,
        requests_per_second=2.0,
    )

    result_df = await evaluator.run(overwrite=True)

    # Save combined result
    result_path = model_output_dir / f"{model_name}_safety_judge.pickle"
    result_df.to_pickle(result_path)
    logger.info(f"Safety evaluation saved to {result_path}")

    return result_df


def run_iteration_analysis(demo_df: pd.DataFrame, output_dir: Path, model_name: str = "demo_model"):
    """Run per-iteration StrongREJECT analysis."""
    from iteration_evaluator import IterationQualityEvaluator, compute_iteration_summary, detect_quality_degradation

    iter_output_dir = output_dir / "iteration_analysis" / model_name
    iter_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running iteration analysis for {model_name}...")

    evaluator = IterationQualityEvaluator(
        eval_model='gpt-4o-mini',
        client_name='openai',
        temperature=0.0,
        max_tokens=128,
        requests_per_second=2.0,
        output_dir=str(iter_output_dir),
    )

    # Get available iteration columns
    iter_cols = [col for col in demo_df.columns if col.startswith('target_iteration_')]
    logger.info(f"Found {len(iter_cols)} iteration columns: {iter_cols}")

    result_df = evaluator.evaluate_iterations(
        df=demo_df,
        prompt_col='inquiry',
        iteration_cols=iter_cols,
        nick_name=model_name
    )

    # Compute summary statistics
    summary_df = compute_iteration_summary(result_df, num_iterations=len(iter_cols))
    degradation = detect_quality_degradation(summary_df)

    # Save results
    result_df.to_pickle(iter_output_dir / f"{model_name}_iteration_results.pickle")
    summary_df.to_csv(iter_output_dir / f"{model_name}_iteration_summary.csv", index=False)

    # Save degradation analysis
    import json
    with open(iter_output_dir / f"{model_name}_degradation_analysis.json", 'w') as f:
        json.dump(degradation, f, indent=2)

    logger.info(f"Iteration analysis saved to {iter_output_dir}")

    return result_df, summary_df, degradation


def compute_strongreject_aggregation(safety_df: pd.DataFrame, output_dir: Path, model_name: str):
    """Compute StrongREJECT aggregate scores."""
    from danger_score import (
        compute_strongreject_score_row,
        compute_frequency_dict,
        compute_average,
        compute_boolean_distribution,
    )

    agg_dir = output_dir / "aggregation"
    agg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing StrongREJECT aggregation...")

    # Compute per-row StrongREJECT scores
    safety_df['strongreject_score'] = safety_df.apply(compute_strongreject_score_row, axis=1)

    # Aggregate by category
    results = []
    for category in safety_df['category'].unique():
        cat_df = safety_df[safety_df['category'] == category]

        result = {
            'model_name': model_name,
            'category': category,
            'sample_count': len(cat_df),
        }

        # Compliance metrics
        if 'complied' in cat_df.columns:
            result['complied'] = compute_boolean_distribution(cat_df['complied'])

        # Quality metrics
        for col in ['specificity', 'convincingness', 'strongreject_score', 'severity_level']:
            if col in cat_df.columns:
                result[col] = compute_average(cat_df[col])

        results.append(result)

    # Overall aggregation
    overall = {
        'model_name': model_name,
        'category': 'all',
        'sample_count': len(safety_df),
    }
    if 'complied' in safety_df.columns:
        overall['complied'] = compute_boolean_distribution(safety_df['complied'])
    for col in ['specificity', 'convincingness', 'strongreject_score', 'severity_level']:
        if col in safety_df.columns:
            overall[col] = compute_average(safety_df[col])
    results.append(overall)

    # Create summary DataFrame
    agg_df = pd.DataFrame(results)
    agg_df.to_csv(agg_dir / f"{model_name}_strongreject_summary.csv", index=False)
    agg_df.to_pickle(agg_dir / f"{model_name}_strongreject_summary.pickle")

    logger.info(f"Aggregation saved to {agg_dir}")

    return agg_df


def print_summary(safety_df: pd.DataFrame, iter_summary: pd.DataFrame, degradation: dict, agg_df: pd.DataFrame):
    """Print demo summary to console."""
    print("\n" + "=" * 70)
    print("STRONGREJECT DEMO SUMMARY")
    print("=" * 70)

    print("\n--- Safety Evaluation Results ---")
    print(f"Total samples evaluated: {len(safety_df)}")

    if 'complied' in safety_df.columns:
        complied_count = safety_df['complied'].sum()
        print(f"Compliance rate: {complied_count}/{len(safety_df)} ({100*complied_count/len(safety_df):.1f}%)")

    if 'specificity' in safety_df.columns:
        print(f"Avg specificity: {safety_df['specificity'].mean():.2f}")
    if 'convincingness' in safety_df.columns:
        print(f"Avg convincingness: {safety_df['convincingness'].mean():.2f}")
    if 'strongreject_score' in safety_df.columns:
        print(f"Avg StrongREJECT score: {safety_df['strongreject_score'].mean():.3f}")

    print("\n--- Per-Iteration Analysis ---")
    if not iter_summary.empty:
        print(iter_summary.to_string(index=False))

    print("\n--- Degradation Analysis ---")
    print(f"Peak iteration: {degradation.get('peak_iteration', 'N/A')}")
    print(f"Peak score: {degradation.get('peak_score', 'N/A'):.3f}" if degradation.get('peak_score') else "Peak score: N/A")
    print(f"Final score: {degradation.get('final_score', 'N/A'):.3f}" if degradation.get('final_score') else "Final score: N/A")
    print(f"Degradation detected: {degradation.get('degradation_detected', 'N/A')}")
    print(f"Score trend: {degradation.get('score_trend', 'N/A')}")

    print("\n--- Category Breakdown ---")
    print(agg_df.to_string(index=False))

    print("\n" + "=" * 70)


async def main():
    """Run the full demo pipeline."""
    # Configuration
    INPUT_FILE = PROJECT_ROOT / "max_iterations_5" / "DeepSeek-R1-0528_preprocessed.pickle"
    OUTPUT_DIR = DEMO_DIR / "results"
    MODEL_NAME = "DeepSeek-R1-demo"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("StrongREJECT Demo: Safety Evaluation Pipeline")
    print("=" * 70)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Samples: 15 bio + 15 chem = 30 total")
    print("=" * 70 + "\n")

    # Step 1: Sample demo data
    demo_df = sample_demo_data(INPUT_FILE, n_bio=15, n_chem=15)
    demo_df.to_pickle(OUTPUT_DIR / "demo_input_data.pickle")

    # Step 2: Run safety evaluation (API calls)
    safety_df = await run_safety_evaluation(demo_df, OUTPUT_DIR, MODEL_NAME)

    # Step 3: Run iteration analysis (API calls)
    iter_df, iter_summary, degradation = run_iteration_analysis(demo_df, OUTPUT_DIR, MODEL_NAME)

    # Step 4: Compute aggregation
    agg_df = compute_strongreject_aggregation(safety_df, OUTPUT_DIR, MODEL_NAME)

    # Step 5: Print summary
    print_summary(safety_df, iter_summary, degradation, agg_df)

    # Save final combined report
    report = {
        'model_name': MODEL_NAME,
        'total_samples': len(demo_df),
        'categories': demo_df['category'].value_counts().to_dict(),
        'avg_strongreject_score': float(safety_df['strongreject_score'].mean()) if 'strongreject_score' in safety_df.columns else None,
        'degradation': degradation,
    }

    import json
    with open(OUTPUT_DIR / "demo_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nDemo complete! Results saved to {OUTPUT_DIR}")
    print(f"\nDemo complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
