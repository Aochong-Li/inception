"""Data loading utilities for safety evaluation results.

Provides functions to load and merge model evaluation results from pickle files.
"""
from pathlib import Path
from typing import Optional, cast

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
DEFAULT_RESULTS_DIR = _script_dir.parent / "evaluation" / "evaluation-results"

# Safety metrics columns
SAFETY_METRICS = [
    "severity_level",
    "delivery_method",
    "material_access",
    "actor_count",
    "cost_usd",
    "fatalities",
    "injured",
    "economic_damage_usd",
    "requires_government_intervention",
]

# Cybersecurity-specific columns
CYBER_METRICS = ["attack_vector", "sophistication"]


def load_all_model_results(
    results_dir: Optional[Path] = None,
    models: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load and merge all per-model safety evaluation results.

    Args:
        results_dir: Path to results directory. Defaults to evaluation/evaluation-results/.
        models: List of specific model names to load. If None, loads all available.

    Returns:
        DataFrame with all model results merged, including 'model_name' column.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    results_dir = Path(results_dir)
    all_results: list[pd.DataFrame] = []

    # Find model directories (exclude 'cumulative' and 'iterations')
    model_dirs = [
        d
        for d in results_dir.iterdir()
        if d.is_dir() and d.name not in ("cumulative", "iterations")
    ]

    if models is not None:
        model_dirs = [d for d in model_dirs if d.name in models]

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        result_file = model_dir / f"{model_name}_safety_judge.pickle"

        if not result_file.exists():
            continue

        df = cast(pd.DataFrame, pd.read_pickle(result_file))
        df["model_name"] = model_name
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    merged = pd.concat(all_results, ignore_index=True)

    # Ensure model_name is first column
    cols = ["model_name"] + [c for c in merged.columns if c != "model_name"]
    return cast(pd.DataFrame, merged[cols])


def load_cumulative_results(
    results_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cumulative aggregated results for bio and chem categories.

    Args:
        results_dir: Path to results directory. Defaults to evaluation/evaluation-results/.

    Returns:
        Tuple of (bio_df, chem_df) DataFrames with aggregated results per model.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    results_dir = Path(results_dir)
    cumulative_dir = results_dir / "cumulative"

    bio_path = cumulative_dir / "bio_evaluator_results.pickle"
    chem_path = cumulative_dir / "chem_evaluator_results.pickle"

    bio_df = cast(pd.DataFrame, pd.read_pickle(bio_path)) if bio_path.exists() else pd.DataFrame()
    chem_df = cast(pd.DataFrame, pd.read_pickle(chem_path)) if chem_path.exists() else pd.DataFrame()

    return bio_df, chem_df


def get_model_names(results_dir: Optional[Path] = None) -> list[str]:
    """Get list of available model names.

    Args:
        results_dir: Path to results directory.

    Returns:
        Sorted list of model names with evaluation results.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    results_dir = Path(results_dir)
    model_dirs = [
        d.name
        for d in results_dir.iterdir()
        if d.is_dir()
        and d.name not in ("cumulative", "iterations")
        and (d / f"{d.name}_safety_judge.pickle").exists()
    ]
    return sorted(model_dirs)


def get_severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute severity level distribution per model (normalized to percentages).

    Args:
        df: DataFrame with model_name and severity_level columns.

    Returns:
        DataFrame with models as rows, severity levels (1-5) as columns, values as percentages.
    """
    # Filter out None/NaN severity values
    df_valid = df[df["severity_level"].notna()].copy()
    df_valid["severity_level"] = pd.to_numeric(
        df_valid["severity_level"], errors="coerce"
    )
    df_valid = cast(pd.DataFrame, df_valid.dropna(subset=["severity_level"]))

    # Count by model and severity
    counts = df_valid.groupby(["model_name", "severity_level"]).size().unstack(fill_value=0)

    # Normalize to percentages
    row_sums = counts.sum(axis=1)
    percentages = counts.div(row_sums, axis=0) * 100

    # Ensure all severity levels 1-5 are present
    for level in range(1, 6):
        if level not in percentages.columns:
            percentages[level] = 0.0

    return cast(pd.DataFrame, percentages[[1, 2, 3, 4, 5]])


def get_categorical_distribution(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Compute distribution of a categorical column per model.

    Args:
        df: DataFrame with model_name column.
        column: Name of the categorical column to analyze.

    Returns:
        DataFrame with models as rows, categories as columns, values as counts.
    """
    df_valid = df[df[column].notna()].copy()

    # Replace None string with 'Failed Evaluation'
    df_valid.loc[:, column] = df_valid[column].apply(
        lambda x: "Failed Evaluation" if str(x) == "None" else str(x)
    )

    counts = df_valid.groupby(["model_name", column]).size().unstack(fill_value=0)
    return cast(pd.DataFrame, counts)


def get_categorical_distribution_pct(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Compute percentage distribution of a categorical column per model.

    Args:
        df: DataFrame with model_name column.
        column: Name of the categorical column to analyze.

    Returns:
        DataFrame with models as rows, categories as columns, values as percentages.
    """
    counts = get_categorical_distribution(df, column)
    row_sums = counts.sum(axis=1)
    return counts.div(row_sums, axis=0) * 100
