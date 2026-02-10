"""
Prepare benchmark samples from existing evaluation results.

Samples 100 inception iterations (50 chem + 50 bio) for evaluator model benchmarking.
"""

import pandas as pd
from pathlib import Path
import argparse


def sample_benchmark_data(
    results_dir: Path,
    n_chem: int = 50,
    n_bio: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample balanced benchmark data from existing evaluation results.

    Args:
        results_dir: Path to evaluation-results directory
        n_chem: Number of chemical samples (default: 50)
        n_bio: Number of bio samples (default: 50)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sampled benchmark data
    """
    all_data = []

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name in ['cumulative', 'iterations']:
            continue

        safety_pickle = model_dir / f'{model_dir.name}_safety_judge.pickle'
        if safety_pickle.exists():
            df = pd.read_pickle(safety_pickle)
            df['source_model'] = model_dir.name
            all_data.append(df)
            print(f"Loaded {model_dir.name}: {len(df)} rows")

    if not all_data:
        raise ValueError(f"No data found in {results_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows available: {len(combined)}")

    # Separate by category
    chem_df = combined[combined['category'] == 'wmdp-chem']
    bio_df = combined[combined['category'] == 'wmdp-bio']

    print(f"Chemical samples available: {len(chem_df)}")
    print(f"Bio samples available: {len(bio_df)}")

    # Sample with replacement if needed
    if len(chem_df) < n_chem:
        print(f"Warning: Only {len(chem_df)} chem samples available, using all")
        chem_sample = chem_df
    else:
        chem_sample = chem_df.sample(n=n_chem, random_state=seed)

    if len(bio_df) < n_bio:
        print(f"Warning: Only {len(bio_df)} bio samples available, using all")
        bio_sample = bio_df
    else:
        bio_sample = bio_df.sample(n=n_bio, random_state=seed)

    # Combine and shuffle
    benchmark_df = pd.concat([chem_sample, bio_sample], ignore_index=True)
    benchmark_df = benchmark_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Keep only columns needed for re-evaluation
    required_cols = [
        'inquiry', 'reasoning_traces', 'category', 'source_model',
        'question', 'choices', 'example', 'situation', 'reasoning', 'prompt',
    ]
    # Add iteration columns if present
    for i in range(5):
        required_cols.extend([f'architect_iteration_{i}', f'target_iteration_{i}'])

    available_cols = [c for c in required_cols if c in benchmark_df.columns]
    benchmark_df = benchmark_df[available_cols]

    print(f"\nSampled {len(benchmark_df)} rows ({len(chem_sample)} chem + {len(bio_sample)} bio)")
    return benchmark_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare benchmark samples for evaluator model comparison"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/evaluation-results",
        help="Path to evaluation-results directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="benchmark/data/benchmark_samples.pickle",
        help="Output pickle file path"
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

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_df = sample_benchmark_data(
        results_dir=results_dir,
        n_chem=args.n_chem,
        n_bio=args.n_bio,
        seed=args.seed,
    )

    benchmark_df.to_pickle(output_path)
    print(f"\nSaved benchmark samples to {output_path}")

    # Print sample info
    print("\nSample preview:")
    print(f"Columns: {list(benchmark_df.columns)}")
    print(f"Shape: {benchmark_df.shape}")


if __name__ == "__main__":
    main()
