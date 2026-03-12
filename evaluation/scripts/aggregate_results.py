#!/usr/bin/env python3
"""
Aggregate per-model safety evaluation results into cumulative summary files.

Creates:
- results/cumulative/chem_evaluator_results.pickle
- results/cumulative/bio_evaluator_results.pickle
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
_eval_dir = _script_dir.parent
DEFAULT_RESULTS_DIR = _eval_dir / "archived" / "results"

# Schema definitions
CHEM_CATEGORICAL = ["agent_class", "delivery_method", "agent_grade", "exposure_environment"]
CHEM_NUMERIC = ["severity_level", "fatalities", "specificity", "convincingness"]

BIO_CATEGORICAL = ["threat", "delivery_method", "contagiousness", "route", "scale"]
BIO_NUMERIC = ["severity_level", "fatalities", "specificity", "convincingness"]


def compute_frequency_dict(series: pd.Series) -> dict:
    """Compute frequency dictionary for a categorical column."""
    # Filter out None/NaN values and convert to string
    values = series.dropna().astype(str).tolist()
    return dict(Counter(values))


def compute_average(series: pd.Series) -> float:
    """Compute average for a numeric column, handling non-numeric values."""
    numeric_values = pd.to_numeric(series, errors="coerce")
    return numeric_values.mean()


def aggregate_model_results(df: pd.DataFrame, category: str) -> dict:
    """Aggregate a single model's results for a given category."""
    if category == "wmdp-chem":
        categorical_cols = CHEM_CATEGORICAL
        numeric_cols = CHEM_NUMERIC
    else:  # wmdp-bio
        categorical_cols = BIO_CATEGORICAL
        numeric_cols = BIO_NUMERIC

    result = {}

    # Compute frequency dicts for categorical columns
    for col in categorical_cols:
        if col in df.columns:
            result[col] = compute_frequency_dict(df[col])
        else:
            result[col] = {}

    # Compute averages for numeric columns
    for col in numeric_cols:
        if col in df.columns:
            result[col] = compute_average(df[col])
        else:
            result[col] = None

    return result


def aggregate_all_models(results_dir: Path = None):
    """Aggregate results from all models."""
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    cumulative_dir = results_dir / "cumulative"
    cumulative_dir.mkdir(parents=True, exist_ok=True)

    chem_results = []
    bio_results = []

    # Find all model result directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "cumulative"]

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        result_file = model_dir / f"{model_name}_safety_judge.pickle"

        if not result_file.exists():
            print(f"WARNING: No results found for {model_name}, skipping...")
            continue

        print(f"Processing: {model_name}")
        df = pd.read_pickle(result_file)

        # Split by category
        chem_df = df[df["category"] == "wmdp-chem"]
        bio_df = df[df["category"] == "wmdp-bio"]

        # Aggregate chem results
        if len(chem_df) > 0:
            chem_agg = aggregate_model_results(chem_df, "wmdp-chem")
            chem_agg["model_name"] = model_name
            chem_results.append(chem_agg)
            print(f"  Chem: {len(chem_df)} rows")

        # Aggregate bio results
        if len(bio_df) > 0:
            bio_agg = aggregate_model_results(bio_df, "wmdp-bio")
            bio_agg["model_name"] = model_name
            bio_results.append(bio_agg)
            print(f"  Bio: {len(bio_df)} rows")

    # Create and save cumulative dataframes
    if chem_results:
        chem_df = pd.DataFrame(chem_results)
        # Reorder columns to put model_name first
        cols = ["model_name"] + [c for c in chem_df.columns if c != "model_name"]
        chem_df = chem_df[cols]
        chem_path = cumulative_dir / "chem_evaluator_results.pickle"
        chem_df.to_pickle(chem_path)
        print(f"\nSaved chem results: {chem_path}")
        print(f"  Models: {len(chem_df)}")
        print(f"  Columns: {chem_df.columns.tolist()}")

    if bio_results:
        bio_df = pd.DataFrame(bio_results)
        # Reorder columns to put model_name first
        cols = ["model_name"] + [c for c in bio_df.columns if c != "model_name"]
        bio_df = bio_df[cols]
        bio_path = cumulative_dir / "bio_evaluator_results.pickle"
        bio_df.to_pickle(bio_path)
        print(f"\nSaved bio results: {bio_path}")
        print(f"  Models: {len(bio_df)}")
        print(f"  Columns: {bio_df.columns.tolist()}")

    return chem_df if chem_results else None, bio_df if bio_results else None


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-model results into cumulative files")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing per-model result subdirectories (default: evaluation/archived/results)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR

    print("Aggregating model results...")
    print(f"Results directory: {results_dir}")
    print()

    chem_df, bio_df = aggregate_all_models(results_dir)

    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print("=" * 60)

    if chem_df is not None:
        print("\nChem summary:")
        print(chem_df.to_string())

    if bio_df is not None:
        print("\nBio summary:")
        print(bio_df.to_string())


if __name__ == "__main__":
    main()
