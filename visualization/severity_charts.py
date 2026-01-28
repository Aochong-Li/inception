"""Severity-related visualizations for safety evaluation results.

Phase 1 charts:
- severity_comparison_bar: Horizontal bar chart comparing mean severity across models
- severity_heatmap: Heatmap showing severity distribution per model
"""
from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.data_loader import (
    get_severity_distribution,
    load_all_model_results,
)
from visualization.theme import (
    FIGURE_SIZES,
    SEVERITY_COLORS,
    apply_theme,
    get_model_palette,
)


def severity_comparison_bar(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create horizontal bar chart comparing mean severity across models.

    Shows mean severity level per model with 95% confidence interval error bars.
    Higher severity = more dangerous outputs.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Filter valid severity values and convert to numeric
    df_valid = df[df["severity_level"].notna()].copy()
    df_valid["severity_level"] = pd.to_numeric(
        df_valid["severity_level"], errors="coerce"
    )
    df_valid = cast(pd.DataFrame, df_valid.dropna(subset=["severity_level"]))

    # Calculate mean and confidence intervals per model
    stats = df_valid.groupby("model_name")["severity_level"].agg(["mean", "std", "count"])
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci95"] = stats["se"] * 1.96
    stats = stats.sort_values("mean", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    # Get colors for models
    colors = get_model_palette(stats.index.tolist())

    # Create horizontal bar chart
    y_pos = np.arange(len(stats))
    bars = ax.barh(
        y_pos,
        stats["mean"],
        xerr=stats["ci95"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )

    # Add value labels
    for i, (mean, ci) in enumerate(zip(stats["mean"], stats["ci95"])):
        ax.text(
            mean + ci + 0.1,
            i,
            f"{mean:.2f}",
            va="center",
            fontsize=10,
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats.index)
    ax.set_xlabel("Mean Severity Level")
    ax.set_title("Model Severity Comparison\n(Higher = More Dangerous Outputs)")
    ax.set_xlim(0, 6)

    # Add severity level reference lines
    for level in range(1, 6):
        ax.axvline(x=level, color=SEVERITY_COLORS[level], alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def severity_heatmap(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    return_data: bool = False,
) -> plt.Figure | tuple[plt.Figure, pd.DataFrame]:
    """Create heatmap showing severity distribution per model.

    Rows are models, columns are severity levels 1-5.
    Values are percentages (each row sums to 100%).

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.
        return_data: If True, also returns the underlying data.

    Returns:
        Matplotlib Figure, or tuple of (Figure, DataFrame) if return_data=True.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Get severity distribution
    dist = get_severity_distribution(df)

    # Sort models by mean severity (descending)
    dist["mean"] = (
        dist[1] * 1 + dist[2] * 2 + dist[3] * 3 + dist[4] * 4 + dist[5] * 5
    ) / 100
    dist = dist.sort_values("mean", ascending=False)
    dist = dist.drop(columns=["mean"])

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    # Create heatmap
    sns.heatmap(
        dist,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",  # Red for high severity, green for low
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.5,
    )

    # Styling
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Model")
    ax.set_title("Severity Distribution by Model\n(% of evaluations at each level)")

    # Update x-axis labels with severity descriptions
    severity_labels = [
        "1\nAssassination",
        "2\nLocal",
        "3\nNational",
        "4\nRegional",
        "5\nMass Atrocity",
    ]
    ax.set_xticklabels(severity_labels, rotation=0)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if return_data:
        return fig, dist
    return fig


def severity_by_category(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create grouped bar chart showing severity by category (bio vs chem).

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Filter valid severity values
    df_valid = df[df["severity_level"].notna()].copy()
    df_valid["severity_level"] = pd.to_numeric(
        df_valid["severity_level"], errors="coerce"
    )
    df_valid = cast(pd.DataFrame, df_valid.dropna(subset=["severity_level"]))

    # Calculate mean severity per model and category
    stats = df_valid.groupby(["model_name", "category"])["severity_level"].mean().unstack()

    # Sort by overall mean severity
    stats["overall"] = stats.mean(axis=1)
    stats = stats.sort_values("overall", ascending=False)
    stats = stats.drop(columns=["overall"])

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    # Create grouped bar chart
    x = np.arange(len(stats))
    width = 0.35

    if "wmdp-bio" in stats.columns:
        bars1 = ax.bar(x - width / 2, stats["wmdp-bio"], width, label="Bio", color="#3b528b")
    if "wmdp-chem" in stats.columns:
        bars2 = ax.bar(x + width / 2, stats["wmdp-chem"], width, label="Chem", color="#21918c")

    # Styling
    ax.set_ylabel("Mean Severity Level")
    ax.set_title("Severity Level by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 6)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
