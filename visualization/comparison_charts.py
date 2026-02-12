"""Multi-metric comparison visualizations.

Phase 2 charts:
- impact_radar: Radar chart comparing normalized impact metrics
- cost_fatalities_scatter: Log-log scatter of cost vs. fatalities
- severity_category_boxplot: Faceted boxplots by category
- feasibility_bubble: Bubble chart of actor_count vs cost vs severity
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from visualization.data_loader import load_all_model_results
from visualization.theme import FIGURE_SIZES, apply_theme, get_model_color, get_model_palette


def impact_radar(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create radar chart comparing normalized impact metrics across models.

    Metrics included: severity_level, fatalities, injured, economic_damage,
    actor_count (inverted - fewer is worse), cost_usd (inverted - lower is worse).

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Metrics to include (normalized 0-1, higher = more dangerous)
    metrics = [
        "severity_level",
        "fatalities",
        "injured",
        "economic_damage_usd",
        "actor_count_inv",  # inverted: fewer actors = more dangerous
        "cost_usd_inv",  # inverted: lower cost = more dangerous
    ]

    metric_labels = [
        "Severity",
        "Fatalities",
        "Injured",
        "Economic\nDamage",
        "Ease of\nOrganization",
        "Cost\nAccessibility",
    ]

    # Calculate mean metrics per model
    numeric_cols = ["severity_level", "fatalities", "injured", "economic_damage_usd", "actor_count", "cost_usd"]
    model_stats = df.groupby("model_name")[numeric_cols].mean()

    # Normalize to 0-1 scale
    for col in numeric_cols:
        col_min = model_stats[col].min()
        col_max = model_stats[col].max()
        if col_max > col_min:
            model_stats[col] = (model_stats[col] - col_min) / (col_max - col_min)
        else:
            model_stats[col] = 0.5

    # Invert actor_count and cost_usd (lower = more dangerous)
    model_stats["actor_count_inv"] = 1 - model_stats["actor_count"]
    model_stats["cost_usd_inv"] = 1 - model_stats["cost_usd"]

    # Setup radar chart
    n_metrics = len(metrics)
    angles = [n / n_metrics * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["square"], subplot_kw={"polar": True})

    # Plot each model
    models = model_stats.index.tolist()
    colors = get_model_palette(models)

    for model, color in zip(models, colors):
        values = [model_stats.loc[model, m] for m in metrics]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Model Risk Profile Comparison\n(Higher = More Dangerous)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def cost_fatalities_scatter(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create log-log scatter plot of cost vs. fatalities.

    Bottom-right quadrant (low cost, high fatalities) is most dangerous.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Filter rows with valid numeric values
    df_valid = df[
        (df["cost_usd"].notna()) &
        (df["fatalities"].notna()) &
        (pd.to_numeric(df["cost_usd"], errors="coerce") > 0) &
        (pd.to_numeric(df["fatalities"], errors="coerce") > 0)
    ].copy()

    df_valid["cost_usd"] = pd.to_numeric(df_valid["cost_usd"], errors="coerce")
    df_valid["fatalities"] = pd.to_numeric(df_valid["fatalities"], errors="coerce")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    # Plot each model
    models = df_valid["model_name"].unique()
    colors = get_model_palette(list(models))

    for model, color in zip(models, colors):
        model_data = df_valid[df_valid["model_name"] == model]
        ax.scatter(
            model_data["cost_usd"],
            model_data["fatalities"],
            label=model,
            color=color,
            alpha=0.6,
            s=50,
            edgecolor="white",
            linewidth=0.5,
        )

    # Log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Styling
    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Fatalities")
    ax.set_title("Cost vs. Fatalities by Model\n(Bottom-Right = Highest Risk)")
    ax.legend(loc="upper left", fontsize=8)

    # Add danger zone indicator
    ax.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="_nolegend_")
    ax.axvline(x=10000, color="red", linestyle="--", alpha=0.5, label="_nolegend_")
    ax.fill_between([1, 10000], [1000, 1000], [1e8, 1e8], color="red", alpha=0.05, label="_nolegend_")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def severity_category_boxplot(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create faceted boxplots showing severity distribution by category.

    Allows comparison of model performance within bio vs. chem categories.

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
    df_valid["severity_level"] = pd.to_numeric(df_valid["severity_level"], errors="coerce")
    df_valid = df_valid.dropna(subset=["severity_level"])

    # Get unique categories and models
    categories = sorted(df_valid["category"].unique())
    models = sorted(df_valid["model_name"].unique())

    # Create figure with subplots for each category
    fig, axes = plt.subplots(1, len(categories), figsize=(7 * len(categories), 6), sharey=True)
    if len(categories) == 1:
        axes = [axes]

    colors = get_model_palette(models)

    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_data = df_valid[df_valid["category"] == category]

        # Create boxplot data
        box_data = [cat_data[cat_data["model_name"] == m]["severity_level"].values for m in models]
        box_data = [d for d in box_data if len(d) > 0]
        box_models = [m for m, d in zip(models, [cat_data[cat_data["model_name"] == m]["severity_level"].values for m in models]) if len(d) > 0]

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            labels=box_models,
        )

        # Color boxes
        box_colors = get_model_palette(box_models)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(category.replace("wmdp-", "").upper())
        ax.set_ylabel("Severity Level" if idx == 0 else "")
        ax.set_ylim(0, 6)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Severity Distribution by Category", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def feasibility_bubble(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create bubble chart of actor_count vs cost vs severity.

    X-axis: actor_count (fewer = easier to organize)
    Y-axis: cost_usd (lower = more accessible)
    Bubble size: severity (larger = more dangerous)
    Color: model

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Filter valid rows
    df_valid = df[
        (df["actor_count"].notna()) &
        (df["cost_usd"].notna()) &
        (df["severity_level"].notna()) &
        (pd.to_numeric(df["actor_count"], errors="coerce") > 0) &
        (pd.to_numeric(df["cost_usd"], errors="coerce") > 0)
    ].copy()

    df_valid["actor_count"] = pd.to_numeric(df_valid["actor_count"], errors="coerce")
    df_valid["cost_usd"] = pd.to_numeric(df_valid["cost_usd"], errors="coerce")
    df_valid["severity_level"] = pd.to_numeric(df_valid["severity_level"], errors="coerce")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])

    models = df_valid["model_name"].unique()

    for model in models:
        model_data = df_valid[df_valid["model_name"] == model]
        color = get_model_color(model)

        # Scale bubble sizes (severity 1-5 -> 50-250)
        sizes = model_data["severity_level"].values * 50

        ax.scatter(
            model_data["actor_count"],
            model_data["cost_usd"],
            s=sizes,
            label=model,
            color=color,
            alpha=0.5,
            edgecolor="white",
            linewidth=0.5,
        )

    # Log scale for cost
    ax.set_yscale("log")

    # Styling
    ax.set_xlabel("Actor Count (Fewer = Easier to Organize)")
    ax.set_ylabel("Cost (USD) - Lower = More Accessible")
    ax.set_title("Attack Feasibility Analysis\n(Large bubbles in bottom-left = Highest Risk)")

    # Legend for models
    ax.legend(loc="upper right", fontsize=8)

    # Add size legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.5, label=f"Severity {i}") for i in [1, 3, 5]
    ]

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
