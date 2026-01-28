"""Distribution visualizations for safety evaluation metrics.

Phase 1 charts:
- material_access_stacked: 100% stacked bar showing material access distribution
- delivery_method_grouped: Grouped bar chart of delivery methods
- category_breakdown_donuts: Small multiples donut charts for bio/chem composition
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.data_loader import (
    get_categorical_distribution,
    get_categorical_distribution_pct,
    load_all_model_results,
)
from visualization.theme import (
    FIGURE_SIZES,
    MATERIAL_ACCESS_COLORS,
    apply_theme,
    get_model_palette,
)


def material_access_stacked(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create 100% stacked horizontal bar chart of material access distribution.

    Lower barrier materials (basic, commercial) indicate higher risk as
    they're more accessible to potential attackers.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Get percentage distribution
    dist = get_categorical_distribution_pct(df, "material_access")

    # Order columns by risk (basic = highest risk)
    risk_order = ["basic", "commercial", "restricted", "military_grade", "synthetic", "Failed Evaluation"]
    columns = [c for c in risk_order if c in dist.columns]
    dist = dist[columns]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    # Create stacked bar chart
    y_pos = np.arange(len(dist))
    left = np.zeros(len(dist))

    for col in columns:
        color = MATERIAL_ACCESS_COLORS.get(col, "#7f7f7f")
        ax.barh(
            y_pos,
            dist[col],
            left=left,
            label=col,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        left += dist[col].values

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dist.index)
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Material Access Distribution by Model\n(Basic/Commercial = Higher Risk)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", title="Access Level")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def delivery_method_grouped(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create grouped bar chart comparing delivery methods across models.

    Shows the distribution of attack delivery methods for each model.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Get count distribution
    dist = get_categorical_distribution(df, "delivery_method")

    # Get top delivery methods
    method_totals = dist.sum().sort_values(ascending=False)
    top_methods = method_totals.head(6).index.tolist()
    dist = dist[top_methods]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["large"])

    # Create grouped bar chart
    x = np.arange(len(dist))
    width = 0.12
    n_methods = len(top_methods)

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_methods))

    for i, method in enumerate(top_methods):
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            dist[method],
            width,
            label=method,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    # Styling
    ax.set_ylabel("Count")
    ax.set_xlabel("Model")
    ax.set_title("Delivery Method Distribution by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(dist.index, rotation=45, ha="right")
    ax.legend(loc="upper right", title="Delivery Method")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def category_breakdown_donuts(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create small multiples donut charts showing bio/chem composition per model.

    Ensures fair comparison by showing sample composition for each model.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Get category counts per model
    category_counts = df.groupby(["model_name", "category"]).size().unstack(fill_value=0)
    models = category_counts.index.tolist()

    # Determine grid layout
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    colors = {"wmdp-bio": "#3b528b", "wmdp-chem": "#21918c"}

    for idx, model in enumerate(models):
        ax = axes[idx]
        data = category_counts.loc[model]
        labels = [c.replace("wmdp-", "") for c in data.index]
        color_list = [colors.get(c, "#7f7f7f") for c in data.index]

        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            data.values,
            labels=labels,
            colors=color_list,
            autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
            pctdistance=0.75,
            startangle=90,
            wedgeprops={"width": 0.5, "edgecolor": "white"},
        )

        # Add total count in center
        total = data.sum()
        ax.text(0, 0, f"n={total}", ha="center", va="center", fontsize=12)

        ax.set_title(model, fontsize=10)

    # Hide empty axes
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Category Breakdown by Model (Bio vs Chem)", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def government_intervention_bar(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create diverging bar chart for government intervention requirement.

    Shows what proportion of scenarios require government-level resources
    to execute vs. can be done by non-state actors.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Get intervention distribution
    dist = get_categorical_distribution_pct(df, "requires_government_intervention")

    # Ensure Yes/No columns exist
    if "Yes" not in dist.columns:
        dist["Yes"] = 0
    if "No" not in dist.columns:
        dist["No"] = 0

    # Sort by "No" percentage (higher = more dangerous)
    dist = dist.sort_values("No", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    y_pos = np.arange(len(dist))

    # Create diverging bars
    ax.barh(y_pos, dist["No"], color="#d62728", label="No (Higher Risk)", edgecolor="white")
    ax.barh(y_pos, -dist["Yes"], color="#2ca02c", label="Yes (Lower Risk)", edgecolor="white")

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dist.index)
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Government Intervention Required\n(No = More Achievable by Non-State Actors)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlim(-100, 100)
    ax.legend(loc="lower right")

    # Custom x-tick labels
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["100%", "50%", "0", "50%", "100%"])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
