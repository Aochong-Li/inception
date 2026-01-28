"""Visualization theme and styling constants.

Provides consistent color palettes, figure sizes, and styling for all charts.
Uses colorblind-friendly palettes (viridis-based) for accessibility.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-friendly palette based on viridis
COLORS = {
    "primary": "#440154",  # Dark purple
    "secondary": "#3b528b",  # Blue
    "tertiary": "#21918c",  # Teal
    "quaternary": "#5ec962",  # Green
    "quinary": "#fde725",  # Yellow
    "danger_high": "#d62728",  # Red for high severity
    "danger_low": "#2ca02c",  # Green for low severity
    "neutral": "#7f7f7f",  # Gray
    "background": "#ffffff",
    "grid": "#e0e0e0",
}

# Model-specific colors for consistent identification across charts
MODEL_COLORS = {
    "DeepSeek-R1-0528": "#440154",
    "DeepSeek-V3.2": "#3b528b",
    "GLM-4.6": "#21918c",
    "Kimi-K2-Thinking": "#5ec962",
    "Qwen3-235B-A22B-Thinking-2507": "#fde725",
    "Qwen3-Next-80B-A3B-Thinking": "#d62728",
}

# Severity level colors (1=green/safe to 5=red/dangerous)
SEVERITY_COLORS = {
    1: "#2ca02c",  # Green - Assassination level
    2: "#98df8a",  # Light green - Local level
    3: "#ffbb78",  # Orange - National level
    4: "#ff7f0e",  # Dark orange - Regional level
    5: "#d62728",  # Red - Mass Atrocity level
}

# Material access risk colors (basic=high risk to military=low risk barrier)
MATERIAL_ACCESS_COLORS = {
    "basic": "#d62728",  # Red - easiest access
    "commercial": "#ff7f0e",  # Orange
    "restricted": "#ffbb78",  # Light orange
    "military_grade": "#98df8a",  # Light green
    "synthetic": "#2ca02c",  # Green - hardest access
}

# Standard figure sizes
FIGURE_SIZES = {
    "small": (8, 6),
    "medium": (10, 8),
    "large": (14, 10),
    "wide": (14, 6),
    "tall": (8, 12),
    "square": (10, 10),
}

# Font sizes
FONT_SIZES = {
    "title": 14,
    "subtitle": 12,
    "axis_label": 11,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}


def apply_theme() -> None:
    """Apply consistent theme to all matplotlib/seaborn plots."""
    # Set seaborn style
    sns.set_style("whitegrid", {
        "axes.facecolor": COLORS["background"],
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })

    # Set color palette to viridis for colorblind friendliness
    sns.set_palette("viridis")

    # Matplotlib rcParams
    plt.rcParams.update({
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["neutral"],
        "axes.labelcolor": "#333333",
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["axis_label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.titlesize": FONT_SIZES["title"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    })


def get_model_color(model_name: str) -> str:
    """Get consistent color for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Hex color string for the model.
    """
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    # Generate color from viridis palette for unknown models
    palette = sns.color_palette("viridis", n_colors=10)
    idx = hash(model_name) % len(palette)
    return f"#{int(palette[idx][0]*255):02x}{int(palette[idx][1]*255):02x}{int(palette[idx][2]*255):02x}"


def get_model_palette(model_names: list[str]) -> list[str]:
    """Get a list of colors for given model names.

    Args:
        model_names: List of model names.

    Returns:
        List of hex color strings in same order as input.
    """
    return [get_model_color(m) for m in model_names]
