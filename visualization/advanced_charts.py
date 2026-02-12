"""Advanced visualizations for safety evaluation analysis.

Phase 3 charts:
- model_similarity_dendrogram: Hierarchical clustering of models by risk profile
- economic_damage_violin: Violin plots of economic damage distribution
- worst_case_table: Formatted table of worst-case scenarios per model
- delivery_material_sankey: Sankey diagram of delivery method to material access
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from visualization.data_loader import load_all_model_results
from visualization.theme import FIGURE_SIZES, apply_theme, get_model_color, get_model_palette


def model_similarity_dendrogram(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create dendrogram showing hierarchical clustering of models by risk profile.

    Models are clustered based on their mean values across all numeric risk metrics.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Calculate mean metrics per model
    numeric_cols = ["severity_level", "fatalities", "injured", "economic_damage_usd", "actor_count", "cost_usd"]

    # Filter to available columns
    available_cols = [c for c in numeric_cols if c in df.columns]

    model_stats = df.groupby("model_name")[available_cols].mean()

    # Normalize to z-scores
    model_stats_norm = (model_stats - model_stats.mean()) / model_stats.std()
    model_stats_norm = model_stats_norm.infer_objects(copy=False).fillna(0)

    # Calculate pairwise distances
    if len(model_stats_norm) < 2:
        # Not enough models for clustering
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])
        ax.text(0.5, 0.5, "Insufficient data for clustering", ha="center", va="center")
        return fig

    distances = pdist(model_stats_norm.values, metric="euclidean")

    # Hierarchical clustering
    linkage = hierarchy.linkage(distances, method="ward")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    # Create dendrogram
    hierarchy.dendrogram(
        linkage,
        labels=model_stats_norm.index.tolist(),
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=10,
    )

    # Styling
    ax.set_title("Model Similarity Clustering\n(Based on Risk Profile Metrics)")
    ax.set_ylabel("Distance (Ward)")
    ax.set_xlabel("Model")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def economic_damage_violin(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create violin plots showing economic damage distribution per model.

    Uses log scale to handle the wide range of values.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Filter valid economic damage values
    df_valid = df[
        (df["economic_damage_usd"].notna()) &
        (pd.to_numeric(df["economic_damage_usd"], errors="coerce") > 0)
    ].copy()

    df_valid["economic_damage_usd"] = pd.to_numeric(df_valid["economic_damage_usd"], errors="coerce")
    df_valid["log_damage"] = np.log10(df_valid["economic_damage_usd"])

    # Sort models by median damage
    model_order = df_valid.groupby("model_name")["log_damage"].median().sort_values(ascending=False).index.tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])

    colors = get_model_palette(model_order)

    # Create violin plots
    parts = ax.violinplot(
        [df_valid[df_valid["model_name"] == m]["log_damage"].values for m in model_order],
        positions=range(len(model_order)),
        showmeans=True,
        showmedians=True,
    )

    # Color the violins
    for idx, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Styling
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_ylabel("Economic Damage (USD, log10 scale)")
    ax.set_title("Economic Damage Distribution by Model")

    # Custom y-axis labels with fixed ticks
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)  # Fix the ticks first
    ax.set_yticklabels([f"$10^{{{int(y)}}}$" for y in yticks])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def worst_case_table(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create formatted table showing worst-case scenario per model.

    Selects the row with highest severity, breaking ties by fatalities.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object with table.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    # Convert relevant columns to numeric
    df_valid = df.copy()
    df_valid["severity_level"] = pd.to_numeric(df_valid["severity_level"], errors="coerce")
    df_valid["fatalities"] = pd.to_numeric(df_valid["fatalities"], errors="coerce")
    df_valid["cost_usd"] = pd.to_numeric(df_valid["cost_usd"], errors="coerce")
    df_valid["economic_damage_usd"] = pd.to_numeric(df_valid["economic_damage_usd"], errors="coerce")

    # Find worst case per model
    worst_cases = []
    for model in df_valid["model_name"].unique():
        model_data = df_valid[df_valid["model_name"] == model]
        model_data = model_data.dropna(subset=["severity_level"])

        if len(model_data) == 0:
            continue

        # Sort by severity (desc), then fatalities (desc)
        model_data = model_data.sort_values(
            ["severity_level", "fatalities"],
            ascending=[False, False]
        )
        worst = model_data.iloc[0]
        worst_cases.append({
            "Model": model,
            "Severity": int(worst["severity_level"]) if pd.notna(worst["severity_level"]) else "N/A",
            "Fatalities": f"{int(worst['fatalities']):,}" if pd.notna(worst["fatalities"]) else "N/A",
            "Cost": f"${int(worst['cost_usd']):,}" if pd.notna(worst["cost_usd"]) else "N/A",
            "Econ Damage": f"${int(worst['economic_damage_usd']):,}" if pd.notna(worst["economic_damage_usd"]) else "N/A",
            "Category": worst["category"].replace("wmdp-", "") if pd.notna(worst["category"]) else "N/A",
        })

    worst_df = pd.DataFrame(worst_cases)
    worst_df = worst_df.sort_values("Severity", ascending=False)

    # Create figure with table
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=worst_df.values,
        colLabels=worst_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.1, 0.15, 0.15, 0.2, 0.1],
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Header styling
    for j, col in enumerate(worst_df.columns):
        table[(0, j)].set_facecolor("#3b528b")
        table[(0, j)].set_text_props(color="white", weight="bold")

    # Row striping
    for i in range(1, len(worst_df) + 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(worst_df.columns)):
            table[(i, j)].set_facecolor(color)

    ax.set_title("Worst-Case Scenario Per Model", fontsize=14, pad=20)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def impact_summary_stats(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create summary statistics table for key impact metrics.

    Shows mean, median, min, max for fatalities, injuries, economic damage.

    Args:
        df: DataFrame with model results. If None, loads from default location.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object with table.
    """
    apply_theme()

    if df is None:
        df = load_all_model_results()

    metrics = ["fatalities", "injured", "economic_damage_usd"]

    # Convert to numeric
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate stats per model
    stats_list = []
    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        stats = {"Model": model}

        for metric in metrics:
            if metric in model_data.columns:
                valid = model_data[metric].dropna()
                if len(valid) > 0:
                    stats[f"{metric}_mean"] = valid.mean()
                    stats[f"{metric}_max"] = valid.max()

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Format for display
    display_data = []
    for _, row in stats_df.iterrows():
        display_row = [row["Model"]]
        for metric in metrics:
            mean_key = f"{metric}_mean"
            max_key = f"{metric}_max"
            if mean_key in row and pd.notna(row[mean_key]):
                display_row.append(f"{row[mean_key]:,.0f}")
                display_row.append(f"{row[max_key]:,.0f}")
            else:
                display_row.extend(["N/A", "N/A"])
        display_data.append(display_row)

    columns = ["Model", "Fatalities\n(mean)", "Fatalities\n(max)",
               "Injured\n(mean)", "Injured\n(max)",
               "Econ Damage\n(mean)", "Econ Damage\n(max)"]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["large"])
    ax.axis("off")

    table = ax.table(
        cellText=display_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)

    # Header styling
    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#3b528b")
        table[(0, j)].set_text_props(color="white", weight="bold")

    ax.set_title("Impact Summary Statistics by Model", fontsize=14, pad=20)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
