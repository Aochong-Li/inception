"""
Run evaluator model benchmark.

Convenience script to run safety evaluation with different evaluator models
and generate aggregate statistics.
"""

import subprocess
import sys
import argparse
from pathlib import Path


# Evaluator model configurations
EVALUATOR_CONFIGS = {
    "gpt5-mini": {
        "eval_model": "gpt-5-mini-2025-08-07",
        "client_name": "openai",
        "nick_name": "gpt5-mini-benchmark",
    },
    "deepseek-v3.2": {
        "eval_model": "deepseek-ai/DeepSeek-V3.2",
        "client_name": "deepseek",
        "nick_name": "deepseek-v3.2-benchmark",
    },
}


def run_prepare_samples(
    n_chem: int = 50,
    n_bio: int = 50,
    seed: int = 42,
) -> Path:
    """Prepare benchmark samples."""
    output_path = Path("benchmark/data/benchmark_samples.pickle")

    if output_path.exists():
        print(f"Benchmark samples already exist at {output_path}")
        return output_path

    cmd = [
        sys.executable, "benchmark/prepare_samples.py",
        "--n_chem", str(n_chem),
        "--n_bio", str(n_bio),
        "--seed", str(seed),
        "--output_path", str(output_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    if not output_path.exists():
        raise RuntimeError(f"Failed to create benchmark samples at {output_path}")

    return output_path


def run_safety_evaluation(
    evaluator: str,
    input_path: Path,
    overwrite: bool = False,
    rate_limit: float = 3.0,
) -> Path:
    """Run safety evaluation with specified evaluator model."""
    config = EVALUATOR_CONFIGS.get(evaluator)
    if not config:
        raise ValueError(f"Unknown evaluator: {evaluator}. Available: {list(EVALUATOR_CONFIGS.keys())}")

    output_dir = Path(f"benchmark/results/{evaluator}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{config['nick_name']}_safety_judge.pickle"

    if result_path.exists() and not overwrite:
        print(f"Results already exist at {result_path}")
        return result_path

    cmd = [
        sys.executable, "evaluation/safety-judge.py",
        "--input_filepath", str(input_path),
        "--output_dir", str(output_dir),
        "--nick_name", config["nick_name"],
        "--eval_model", config["eval_model"],
        "--client_name", config["client_name"],
        "--category_col", "category",
        "--chemical_label", "wmdp-chem",
        "--bioterrorism_label", "wmdp-bio",
        "--rate_limit", str(rate_limit),
    ]

    if overwrite:
        cmd.append("--overwrite")

    print(f"\nRunning evaluation with {evaluator}:")
    print(f"  Model: {config['eval_model']}")
    print(f"  Client: {config['client_name']}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")

    return result_path


def run_aggregation(evaluator: str, results_path: Path) -> Path:
    """Run aggregation on evaluation results."""
    output_dir = Path("benchmark/summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "benchmark/aggregate_benchmark.py",
        "--results_pickle", str(results_path),
        "--output_dir", str(output_dir),
        "--evaluator_name", evaluator,
    ]

    print(f"\nRunning aggregation for {evaluator}:")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)

    return output_dir / f"{evaluator}_aggregate_stats.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluator model benchmark"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=list(EVALUATOR_CONFIGS.keys()) + ["all"],
        default="all",
        help="Evaluator model to benchmark (or 'all' for both)"
    )
    parser.add_argument(
        "--n_chem",
        type=int,
        default=50,
        help="Number of chemical samples"
    )
    parser.add_argument(
        "--n_bio",
        type=int,
        default=50,
        help="Number of bio samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=3.0,
        help="API rate limit (requests per second)"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation, only run aggregation"
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare samples, don't run evaluation"
    )

    args = parser.parse_args()

    # Step 1: Prepare samples
    print("=" * 60)
    print("Step 1: Preparing benchmark samples")
    print("=" * 60)
    samples_path = run_prepare_samples(
        n_chem=args.n_chem,
        n_bio=args.n_bio,
        seed=args.seed,
    )

    if args.prepare_only:
        print("\nSample preparation complete. Exiting.")
        return

    # Step 2: Run evaluations
    evaluators = list(EVALUATOR_CONFIGS.keys()) if args.evaluator == "all" else [args.evaluator]

    for evaluator in evaluators:
        print("\n" + "=" * 60)
        print(f"Step 2: Running evaluation with {evaluator}")
        print("=" * 60)

        if args.skip_eval:
            # Look for existing results
            config = EVALUATOR_CONFIGS[evaluator]
            results_path = Path(f"benchmark/results/{evaluator}/{config['nick_name']}_safety_judge.pickle")
            if not results_path.exists():
                print(f"No existing results found for {evaluator}, skipping aggregation")
                continue
        else:
            results_path = run_safety_evaluation(
                evaluator=evaluator,
                input_path=samples_path,
                overwrite=args.overwrite,
                rate_limit=args.rate_limit,
            )

        # Step 3: Aggregate results
        print("\n" + "=" * 60)
        print(f"Step 3: Aggregating results for {evaluator}")
        print("=" * 60)
        stats_path = run_aggregation(evaluator, results_path)
        print(f"Aggregate stats saved to {stats_path}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
