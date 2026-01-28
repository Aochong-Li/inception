#!/usr/bin/env python3
"""
Category-wise safety evaluation aggregator.

Aggregates per-model safety evaluation results into a unified DataFrame
with categorical and numeric metrics per model.

Creates:
- results/cumulative/danger_scores.pickle
"""
import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from math import floor, log10
from pathlib import Path
from typing import Optional

import pandas as pd


def round_to_sig_figs(x, n=2):
    """Round a number to n significant figures."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return x
    if x == 0:
        return 0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

_script_dir = Path(__file__).parent.resolve()
RESULTS_DIR = _script_dir / "evaluation-results"
CUMULATIVE_DIR = RESULTS_DIR / "cumulative"

# Field definitions per category
BIO_CATEGORICAL = ["delivery_method", "material_access", "requires_government_intervention"]
BIO_NUMERIC = ["severity_level", "actor_count", "cost_usd", "fatalities", "injured", "economic_damage_usd"]

CHEM_CATEGORICAL = ["delivery_method", "material_access", "requires_government_intervention"]
CHEM_NUMERIC = ["severity_level", "actor_count", "cost_usd", "fatalities", "injured", "economic_damage_usd"]

CYBER_CATEGORICAL = ["attack_vector", "sophistication", "requires_government_intervention"]
CYBER_NUMERIC = ["severity_level", "actor_count", "cost_usd", "economic_damage_usd"]

# Unified schema fields
UNIFIED_CATEGORICAL = [
    "delivery_method",
    "material_access",
    "attack_vector",
    "sophistication",
    "requires_government_intervention",
]
UNIFIED_NUMERIC = [
    "severity_level",
    "actor_count",
    "cost_usd",
    "fatalities",
    "injured",
    "economic_damage_usd",
]



@dataclass
class DangerScoreRow:
    """Unified output schema for safety evaluation aggregation."""

    model_name: str

    # Categorical frequency distributions (ordered by frequency)
    delivery_method: dict = field(default_factory=dict)
    material_access: dict = field(default_factory=dict)
    attack_vector: dict = field(default_factory=dict)
    sophistication: dict = field(default_factory=dict)
    requires_government_intervention: dict = field(default_factory=dict)

    # Numeric averages
    severity_level: Optional[float] = None
    actor_count: Optional[float] = None
    cost_usd: Optional[float] = None

    # Impact metrics (averages)
    fatalities: Optional[float] = None
    injured: Optional[float] = None
    economic_damage_usd: Optional[float] = None

    # Metadata
    sample_count: int = 0
    category_breakdown: dict = field(default_factory=dict)


def compute_frequency_dict(series: pd.Series) -> dict:
    """
    Compute frequency dictionary for a categorical column.

    Returns dict ordered by frequency (most common first).
    """
    values = series.dropna().astype(str).tolist()
    counts = Counter(values)
    # Sort by frequency descending
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def compute_average(series: pd.Series) -> Optional[float]:
    """
    Compute average for a numeric column, handling non-numeric values.

    Returns None if no valid numeric values.
    """
    numeric_values = pd.to_numeric(series, errors="coerce")
    mean_val = numeric_values.mean()
    if pd.isna(mean_val):
        return None
    return float(mean_val)


def aggregate_category_results(df: pd.DataFrame, category: str) -> dict:
    """Aggregate results for a single category (bio/chem/cyber)."""
    if category == "wmdp-bio":
        categorical_cols = BIO_CATEGORICAL
        numeric_cols = BIO_NUMERIC
    elif category == "wmdp-chem":
        categorical_cols = CHEM_CATEGORICAL
        numeric_cols = CHEM_NUMERIC
    else:  # wmdp-cyber
        categorical_cols = CYBER_CATEGORICAL
        numeric_cols = CYBER_NUMERIC

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


def merge_frequency_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two frequency dictionaries, summing counts."""
    merged = Counter(dict1)
    merged.update(dict2)
    # Sort by frequency descending
    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True))


def merge_averages(avg1: Optional[float], count1: int, avg2: Optional[float], count2: int) -> Optional[float]:
    """Compute weighted average of two averages."""
    if avg1 is None and avg2 is None:
        return None
    if avg1 is None:
        return avg2
    if avg2 is None:
        return avg1
    if count1 + count2 == 0:
        return None
    return (avg1 * count1 + avg2 * count2) / (count1 + count2)


def aggregate_model(model_dir: Path) -> Optional[DangerScoreRow]:
    """Aggregate all results for a single model into a DangerScoreRow."""
    model_name = model_dir.name
    result_file = model_dir / f"{model_name}_safety_judge.pickle"

    if not result_file.exists():
        print(f"WARNING: No results found for {model_name}, skipping...")
        return None

    print(f"Processing: {model_name}")
    df = pd.read_pickle(result_file)

    # Split by category
    bio_df = df[df["category"] == "wmdp-bio"]
    chem_df = df[df["category"] == "wmdp-chem"]
    cyber_df = df[df["category"] == "wmdp-cyber"] if "wmdp-cyber" in df["category"].values else pd.DataFrame()

    # Aggregate per category
    bio_agg = aggregate_category_results(bio_df, "wmdp-bio") if len(bio_df) > 0 else {}
    chem_agg = aggregate_category_results(chem_df, "wmdp-chem") if len(chem_df) > 0 else {}
    cyber_agg = aggregate_category_results(cyber_df, "wmdp-cyber") if len(cyber_df) > 0 else {}

    bio_count = len(bio_df)
    chem_count = len(chem_df)
    cyber_count = len(cyber_df)
    total_count = bio_count + chem_count + cyber_count

    print(f"  Bio: {bio_count}, Chem: {chem_count}, Cyber: {cyber_count}")

    if total_count == 0:
        print(f"  No data found for {model_name}")
        return None

    # Merge categorical fields
    delivery_method = merge_frequency_dicts(
        bio_agg.get("delivery_method", {}), chem_agg.get("delivery_method", {})
    )
    material_access = merge_frequency_dicts(
        bio_agg.get("material_access", {}), chem_agg.get("material_access", {})
    )
    attack_vector = cyber_agg.get("attack_vector", {})
    sophistication = cyber_agg.get("sophistication", {})

    # Merge requires_government_intervention across all categories
    gov_bio = bio_agg.get("requires_government_intervention", {})
    gov_chem = chem_agg.get("requires_government_intervention", {})
    gov_cyber = cyber_agg.get("requires_government_intervention", {})
    requires_government_intervention = merge_frequency_dicts(
        merge_frequency_dicts(gov_bio, gov_chem), gov_cyber
    )

    # Merge numeric fields (weighted averages)
    bio_chem_count = bio_count + chem_count

    severity_level = merge_averages(
        merge_averages(bio_agg.get("severity_level"), bio_count, chem_agg.get("severity_level"), chem_count),
        bio_chem_count,
        cyber_agg.get("severity_level"),
        cyber_count,
    )

    actor_count = merge_averages(
        merge_averages(bio_agg.get("actor_count"), bio_count, chem_agg.get("actor_count"), chem_count),
        bio_chem_count,
        cyber_agg.get("actor_count"),
        cyber_count,
    )

    cost_usd = merge_averages(
        merge_averages(bio_agg.get("cost_usd"), bio_count, chem_agg.get("cost_usd"), chem_count),
        bio_chem_count,
        cyber_agg.get("cost_usd"),
        cyber_count,
    )

    # fatalities and injured only from bio/chem
    fatalities = merge_averages(bio_agg.get("fatalities"), bio_count, chem_agg.get("fatalities"), chem_count)
    injured = merge_averages(bio_agg.get("injured"), bio_count, chem_agg.get("injured"), chem_count)

    # economic_damage_usd from all categories
    economic_damage_usd = merge_averages(
        merge_averages(bio_agg.get("economic_damage_usd"), bio_count, chem_agg.get("economic_damage_usd"), chem_count),
        bio_chem_count,
        cyber_agg.get("economic_damage_usd"),
        cyber_count,
    )

    return DangerScoreRow(
        model_name=model_name,
        delivery_method=delivery_method,
        material_access=material_access,
        attack_vector=attack_vector,
        sophistication=sophistication,
        requires_government_intervention=requires_government_intervention,
        severity_level=severity_level,
        actor_count=actor_count,
        cost_usd=cost_usd,
        fatalities=fatalities,
        injured=injured,
        economic_damage_usd=economic_damage_usd,
        sample_count=total_count,
        category_breakdown={"bio": bio_count, "chem": chem_count, "cyber": cyber_count},
    )


def aggregate_all_models() -> pd.DataFrame:
    """Aggregate results from all models into a single DataFrame."""
    CUMULATIVE_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Find all model result directories
    model_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name != "cumulative"]

    for model_dir in sorted(model_dirs):
        row = aggregate_model(model_dir)
        if row is not None:
            results.append(row)

    if not results:
        print("No results found to aggregate")
        return pd.DataFrame()

    # Convert dataclass instances to DataFrame
    df = pd.DataFrame([vars(r) for r in results])

    # Reorder columns: model_name first, metadata last
    primary_cols = ["model_name"]
    categorical_cols = ["delivery_method", "material_access", "attack_vector", "sophistication", "requires_government_intervention"]
    numeric_cols = ["severity_level", "actor_count", "cost_usd", "fatalities", "injured", "economic_damage_usd"]
    meta_cols = ["sample_count", "category_breakdown"]

    col_order = primary_cols + categorical_cols + numeric_cols + meta_cols
    df = df[[c for c in col_order if c in df.columns]]

    # Round numeric columns to 2 significant figures
    cols_to_round = numeric_cols
    for col in cols_to_round:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round_to_sig_figs(x, 2))

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-model safety evaluation results into a unified DataFrame."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(CUMULATIVE_DIR / "danger_scores.pickle"),
        help="Output path for the aggregated pickle file (default: results/cumulative/danger_scores.pickle)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["pickle", "csv", "json"],
        default="pickle",
        help="Output format (default: pickle)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Safety Evaluation Aggregator")
    print("=" * 60)
    print(f"Results directory: {RESULTS_DIR}")
    print()

    df = aggregate_all_models()

    if df.empty:
        print("No data to save.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save in requested format
    if args.format == "csv":
        # Convert dict columns to JSON strings for CSV
        df_csv = df.copy()
        for col in ["delivery_method", "material_access", "attack_vector", "sophistication", "requires_government_intervention", "category_breakdown"]:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        csv_path = output_path.with_suffix(".csv")
        df_csv.to_csv(csv_path, index=False)
        print(f"\nSaved CSV: {csv_path}")
    elif args.format == "json":
        json_path = output_path.with_suffix(".json")
        df.to_json(json_path, orient="records", indent=2)
        print(f"\nSaved JSON: {json_path}")
    else:
        df.to_pickle(output_path)
        print(f"\nSaved pickle: {output_path}")

    print(f"  Models: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df[["model_name", "severity_level", "sample_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
