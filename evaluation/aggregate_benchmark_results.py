#!/usr/bin/env python3
"""
Aggregate benchmark evaluation results into summary DataFrames.

Creates separate aggregates for think and instruct models with
StrongREJECT scores and category breakdowns.
"""
import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from danger_score import compute_strongreject_score_row

BENCHMARK_RESULTS_DIR = Path(__file__).parent.parent / "evaluation-results" / "benchmark"


def load_model_results(model_dir: Path) -> Optional[pd.DataFrame]:
    """Load safety judge results for a single model."""
    model_name = model_dir.name
    result_file = model_dir / f"{model_name}_safety_judge.pickle"

    if not result_file.exists():
        print(f"WARNING: No results found for {model_name}")
        return None

    loaded = pd.read_pickle(result_file)
    if not isinstance(loaded, pd.DataFrame):
        print(f"WARNING: {result_file} is not a DataFrame")
        return None
    df: pd.DataFrame = loaded
    df['model_name'] = model_name
    return df


def compute_model_summary(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Compute summary statistics for a single model."""
    summary: dict[str, Any] = {'model_name': model_name}

    # Sample count
    summary['sample_count'] = len(df)

    # Compliance rate
    if 'complied' in df.columns:
        complied = df['complied'].dropna()
        summary['compliance_rate'] = float(complied.mean()) if len(complied) > 0 else None
        summary['complied_true'] = int(complied.sum())
        summary['complied_false'] = len(complied) - int(complied.sum())

    # Average metrics
    for col in ['specificity', 'convincingness']:
        if col in df.columns:
            values: pd.Series = pd.to_numeric(df[col], errors='coerce')  # type: ignore[assignment]
            has_values = bool(values.notna().any())
            summary[f'avg_{col}'] = float(values.mean()) if has_values else None

    # StrongREJECT score
    if all(c in df.columns for c in ['complied', 'specificity', 'convincingness']):
        scores: pd.Series = df.apply(compute_strongreject_score_row, axis=1)  # type: ignore[assignment]
        has_scores = bool(scores.notna().any())
        summary['strongreject_score'] = float(scores.mean()) if has_scores else None

    # Category breakdown
    if 'category' in df.columns:
        summary['bio_count'] = len(df[df['category'] == 'wmdp-bio'])
        summary['chem_count'] = len(df[df['category'] == 'wmdp-chem'])

    return summary


def aggregate_model_type(model_type: str) -> pd.DataFrame:
    """Aggregate all models of a given type (think or instruct)."""
    results_dir = BENCHMARK_RESULTS_DIR / model_type

    if not results_dir.exists():
        print(f"WARNING: {model_type} results directory not found")
        return pd.DataFrame()

    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    summaries = []
    all_results = []

    for model_dir in sorted(model_dirs):
        df = load_model_results(model_dir)
        if df is not None:
            summary = compute_model_summary(df, model_dir.name)
            summaries.append(summary)
            all_results.append(df)
            print(f"  Processed: {model_dir.name} ({len(df)} samples)")

    if not summaries:
        return pd.DataFrame()

    aggregate_df = pd.DataFrame(summaries)

    # Reorder columns
    col_order = [
        'model_name', 'sample_count', 'compliance_rate',
        'complied_true', 'complied_false',
        'avg_specificity', 'avg_convincingness', 'strongreject_score',
        'bio_count', 'chem_count'
    ]
    result = aggregate_df[[c for c in col_order if c in aggregate_df.columns]]  # type: ignore[assignment]

    return result


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark evaluation results")
    parser.add_argument("--output_dir", type=str,
                        default=str(BENCHMARK_RESULTS_DIR),
                        help="Output directory for aggregate files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Benchmark Evaluation Aggregator")
    print("=" * 60)

    # Aggregate think models
    print("\nProcessing THINK models:")
    think_df = aggregate_model_type("think")
    if not think_df.empty:
        think_df.to_pickle(output_dir / "think_aggregate.pickle")
        think_df.to_csv(output_dir / "think_aggregate.csv", index=False)
        print(f"\nSaved: {output_dir / 'think_aggregate.pickle'}")
        print(think_df.to_string(index=False))

    # Aggregate instruct models
    print("\n\nProcessing INSTRUCT models:")
    instruct_df = aggregate_model_type("instruct")
    if not instruct_df.empty:
        instruct_df.to_pickle(output_dir / "instruct_aggregate.pickle")
        instruct_df.to_csv(output_dir / "instruct_aggregate.csv", index=False)
        print(f"\nSaved: {output_dir / 'instruct_aggregate.pickle'}")
        print(instruct_df.to_string(index=False))

    # Print comparison summary
    if not think_df.empty and not instruct_df.empty:
        print("\n" + "=" * 60)
        print("THINK vs INSTRUCT Comparison")
        print("=" * 60)

        think_avg_sr = think_df['strongreject_score'].mean()
        instruct_avg_sr = instruct_df['strongreject_score'].mean()

        print(f"Think avg StrongREJECT:    {think_avg_sr:.3f}")
        print(f"Instruct avg StrongREJECT: {instruct_avg_sr:.3f}")
        print(f"Difference:                {think_avg_sr - instruct_avg_sr:.3f}")


if __name__ == "__main__":
    main()
