#!/usr/bin/env python3
"""
CLI for running per-iteration StrongREJECT evaluation.

Usage:
    python run_iteration_analysis.py \
        --input_filepath ./max_iterations_5/DeepSeek-R1-0528.pickle \
        --output_dir ./results/iteration_analysis \
        --nick_name DeepSeek-R1

Output format per iteration:
{
    "score": 0.65,           # Composite: (1 - refusal) × (specificity + convincingness) / 2
    "refusal": 0,            # Binary: 0 = complied, 1 = refused
    "convincingness": 3.5,   # 1.0-5.0 Likert scale
    "specificity": 4.0,      # 1.0-5.0 Likert scale
    "judge_model": "openai/gpt-4o-mini"
}
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from iteration_evaluator import (
    IterationQualityEvaluator,
    compute_iteration_summary,
    detect_quality_degradation
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Per-iteration StrongREJECT quality evaluation"
    )

    parser.add_argument(
        "--input_filepath", "-i", type=str, required=True,
        help="Path to input pickle file with iteration columns"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./results/iteration_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--nick_name", "-n", type=str, default="iteration_eval",
        help="Identifier for this evaluation run"
    )
    parser.add_argument(
        "--prompt_col", type=str, default="inquiry",
        help="Column containing original harmful prompt"
    )
    parser.add_argument(
        "--eval_model", type=str, default="gpt-4o-mini",
        help="Model for evaluation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--client_name", type=str, default="openai",
        choices=["openai", "deepseek", "togetherai", "openrouter"],
        help="API client to use"
    )
    parser.add_argument(
        "--rate_limit", type=float, default=5.0,
        help="Requests per second (default: 5.0)"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=5,
        help="Number of iterations to evaluate (default: 5)"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input_filepath}")
    df = pd.read_pickle(args.input_filepath)

    # Build iteration columns
    iteration_cols = [f"target_iteration_{i}" for i in range(args.max_iterations)]

    # Initialize evaluator
    evaluator = IterationQualityEvaluator(
        eval_model=args.eval_model,
        client_name=args.client_name,
        requests_per_second=args.rate_limit,
        output_dir=args.output_dir,
    )

    # Run evaluation
    logger.info(f"Evaluating {len(df)} samples x {len(iteration_cols)} iterations...")
    result_df = evaluator.evaluate_iterations(
        df=df,
        prompt_col=args.prompt_col,
        iteration_cols=iteration_cols,
        nick_name=args.nick_name
    )

    # Compute summary
    summary_df = compute_iteration_summary(result_df, len(iteration_cols))
    degradation = detect_quality_degradation(summary_df)

    # Print results
    print("\n" + "=" * 60)
    print("ITERATION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nPer-Iteration Metrics:")
    print(summary_df.to_string(index=False))

    print(f"\nDegradation Analysis:")
    print(f"  Peak iteration: {degradation.get('peak_iteration')}")
    print(f"  Peak score: {degradation.get('peak_score', 0):.3f}")
    print(f"  Final score: {degradation.get('final_score', 0):.3f}")
    print(f"  Degradation detected: {degradation.get('degradation_detected')}")
    print(f"  Compliance trend: {degradation.get('compliance_trend')}")
    print(f"  Score trend: {degradation.get('score_trend')}")

    # Save summary
    summary_path = Path(args.output_dir) / f"{args.nick_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
