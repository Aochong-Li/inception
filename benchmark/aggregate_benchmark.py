"""
Aggregate benchmark results and compute summary statistics.

Computes metrics across evaluator model runs for comparison.
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.danger_score import compute_strongreject_score_row


def compute_aggregate_stats(results_df: pd.DataFrame) -> dict:
    """
    Compute aggregate statistics from benchmark results.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        Dictionary of aggregate statistics
    """
    stats = {}

    # Basic counts
    stats['total_samples'] = len(results_df)
    stats['category_breakdown'] = results_df['category'].value_counts().to_dict()

    # Error rate
    if 'error' in results_df.columns:
        error_count = results_df['error'].notna().sum()
        stats['error_rate'] = error_count / len(results_df) if len(results_df) > 0 else 0.0
    else:
        stats['error_rate'] = 0.0

    # StrongREJECT metrics
    strongreject_cols = ['refusal', 'convincingness', 'specificity']
    if all(col in results_df.columns for col in strongreject_cols):
        # Compute StrongREJECT scores
        scores = results_df.apply(compute_strongreject_score_row, axis=1)
        valid_scores = scores.dropna()

        stats['strongreject'] = {
            'mean_score': float(valid_scores.mean()) if len(valid_scores) > 0 else None,
            'std_score': float(valid_scores.std()) if len(valid_scores) > 0 else None,
            'valid_count': len(valid_scores),
        }

        # Refusal rate
        if 'refusal' in results_df.columns:
            refusal_values = pd.to_numeric(results_df['refusal'], errors='coerce')
            stats['strongreject']['refusal_rate'] = float(refusal_values.mean()) if len(refusal_values.dropna()) > 0 else None

    # Dimensional means (for StrongREJECT dimensions)
    dimension_cols = ['convincingness', 'specificity', 'material_access', 'population_size', 'money_feasibility']
    stats['dimension_means'] = {}
    for col in dimension_cols:
        if col in results_df.columns:
            values = pd.to_numeric(results_df[col], errors='coerce')
            valid_values = values.dropna()
            if len(valid_values) > 0:
                stats['dimension_means'][col] = {
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'count': len(valid_values),
                }

    # Legacy severity metrics (if present)
    if 'severity_level' in results_df.columns:
        severity = pd.to_numeric(results_df['severity_level'], errors='coerce')
        valid_severity = severity.dropna()
        if len(valid_severity) > 0:
            stats['mean_severity_level'] = float(valid_severity.mean())
            stats['std_severity_level'] = float(valid_severity.std())

    if 'actor_count' in results_df.columns:
        actors = pd.to_numeric(results_df['actor_count'], errors='coerce')
        valid_actors = actors.dropna()
        if len(valid_actors) > 0:
            stats['mean_actor_count'] = float(valid_actors.mean())

    if 'cost_usd' in results_df.columns:
        cost = pd.to_numeric(results_df['cost_usd'], errors='coerce')
        valid_cost = cost.dropna()
        if len(valid_cost) > 0:
            stats['mean_cost_usd'] = float(valid_cost.mean())

    if 'fatalities' in results_df.columns:
        fatalities = pd.to_numeric(results_df['fatalities'], errors='coerce')
        valid_fatalities = fatalities.dropna()
        if len(valid_fatalities) > 0:
            stats['mean_fatalities'] = float(valid_fatalities.mean())

    # Categorical distributions
    categorical_cols = ['delivery_method', 'material_access', 'attack_vector']
    stats['distributions'] = {}
    for col in categorical_cols:
        if col in results_df.columns:
            dist = results_df[col].value_counts().to_dict()
            stats['distributions'][col] = {str(k): int(v) for k, v in dist.items()}

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark evaluation results"
    )
    parser.add_argument(
        "--results_pickle",
        type=str,
        required=True,
        help="Path to evaluation results pickle file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark/summary",
        help="Output directory for aggregate stats"
    )
    parser.add_argument(
        "--evaluator_name",
        type=str,
        default="unknown",
        help="Name of the evaluator model (for output filename)"
    )

    args = parser.parse_args()

    results_path = Path(args.results_pickle)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_df = pd.read_pickle(results_path)
    print(f"Loaded {len(results_df)} results from {results_path}")

    # Compute stats
    stats = compute_aggregate_stats(results_df)
    stats['evaluator_model'] = args.evaluator_name
    stats['source_file'] = str(results_path)

    # Save as JSON
    json_path = output_dir / f"{args.evaluator_name}_aggregate_stats.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved JSON stats to {json_path}")

    # Save as CSV (flattened)
    flat_stats = {
        'evaluator_model': stats['evaluator_model'],
        'total_samples': stats['total_samples'],
        'error_rate': stats['error_rate'],
    }

    if 'strongreject' in stats:
        flat_stats['strongreject_mean'] = stats['strongreject'].get('mean_score')
        flat_stats['strongreject_std'] = stats['strongreject'].get('std_score')
        flat_stats['refusal_rate'] = stats['strongreject'].get('refusal_rate')

    if 'mean_severity_level' in stats:
        flat_stats['mean_severity_level'] = stats['mean_severity_level']

    for col, values in stats.get('dimension_means', {}).items():
        flat_stats[f'{col}_mean'] = values.get('mean')

    csv_path = output_dir / f"{args.evaluator_name}_aggregate_stats.csv"
    pd.DataFrame([flat_stats]).to_csv(csv_path, index=False)
    print(f"Saved CSV stats to {csv_path}")

    # Print summary
    print("\n=== Aggregate Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Error rate: {stats['error_rate']:.2%}")
    if 'strongreject' in stats and stats['strongreject'].get('mean_score') is not None:
        print(f"StrongREJECT mean: {stats['strongreject']['mean_score']:.3f}")
        print(f"Refusal rate: {stats['strongreject'].get('refusal_rate', 0):.2%}")

    print(f"\nCategory breakdown: {stats['category_breakdown']}")


if __name__ == "__main__":
    main()
