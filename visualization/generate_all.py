#!/usr/bin/env python3
"""Generate all safety evaluation visualizations.

CLI tool to generate all visualization charts and save them to the output directory.

Usage:
    python -m visualization.generate_all
    python -m visualization.generate_all --phase 1
    python -m visualization.generate_all --output-dir ./custom_output
"""
import argparse
from pathlib import Path

from visualization.data_loader import load_all_model_results
from visualization.theme import apply_theme

_script_dir = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = _script_dir / "output"


def generate_phase1(df, output_dir: Path) -> list[Path]:
    """Generate Phase 1 (MVP) visualizations.

    Args:
        df: Loaded DataFrame with all model results.
        output_dir: Directory to save output files.

    Returns:
        List of generated file paths.
    """
    from visualization.severity_charts import (
        severity_by_category,
        severity_comparison_bar,
        severity_heatmap,
    )
    from visualization.distribution_charts import (
        category_breakdown_donuts,
        delivery_method_grouped,
        government_intervention_bar,
        material_access_stacked,
    )

    generated = []

    # Severity charts
    print("  Generating severity_comparison_bar...")
    fig = severity_comparison_bar(df)
    path = output_dir / "severity_comparison_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating severity_heatmap...")
    fig = severity_heatmap(df)
    path = output_dir / "severity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating severity_by_category...")
    fig = severity_by_category(df)
    path = output_dir / "severity_by_category.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    # Distribution charts
    print("  Generating material_access_stacked...")
    fig = material_access_stacked(df)
    path = output_dir / "material_access_stacked.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating delivery_method_grouped...")
    fig = delivery_method_grouped(df)
    path = output_dir / "delivery_method_grouped.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating category_breakdown_donuts...")
    fig = category_breakdown_donuts(df)
    path = output_dir / "category_breakdown_donuts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating government_intervention_bar...")
    fig = government_intervention_bar(df)
    path = output_dir / "government_intervention_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    return generated


def generate_phase2(df, output_dir: Path) -> list[Path]:
    """Generate Phase 2 (advanced comparison) visualizations.

    Args:
        df: Loaded DataFrame with all model results.
        output_dir: Directory to save output files.

    Returns:
        List of generated file paths.
    """
    from visualization.comparison_charts import (
        cost_fatalities_scatter,
        feasibility_bubble,
        impact_radar,
        severity_category_boxplot,
    )

    generated = []

    print("  Generating impact_radar...")
    fig = impact_radar(df)
    path = output_dir / "impact_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating cost_fatalities_scatter...")
    fig = cost_fatalities_scatter(df)
    path = output_dir / "cost_fatalities_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating severity_category_boxplot...")
    fig = severity_category_boxplot(df)
    path = output_dir / "severity_category_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating feasibility_bubble...")
    fig = feasibility_bubble(df)
    path = output_dir / "feasibility_bubble.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    return generated


def generate_phase3(df, output_dir: Path) -> list[Path]:
    """Generate Phase 3 (advanced analysis) visualizations.

    Args:
        df: Loaded DataFrame with all model results.
        output_dir: Directory to save output files.

    Returns:
        List of generated file paths.
    """
    from visualization.advanced_charts import (
        economic_damage_violin,
        impact_summary_stats,
        model_similarity_dendrogram,
        worst_case_table,
    )

    generated = []

    print("  Generating model_similarity_dendrogram...")
    fig = model_similarity_dendrogram(df)
    path = output_dir / "model_similarity_dendrogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating economic_damage_violin...")
    fig = economic_damage_violin(df)
    path = output_dir / "economic_damage_violin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating worst_case_table...")
    fig = worst_case_table(df)
    path = output_dir / "worst_case_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    print("  Generating impact_summary_stats...")
    fig = impact_summary_stats(df)
    path = output_dir / "impact_summary_stats.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    generated.append(path)

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate safety evaluation visualizations"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Generate only specific phase (1, 2, or 3). Default: all phases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated charts. Default: {DEFAULT_OUTPUT_DIR}",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation data...")
    apply_theme()
    df = load_all_model_results()

    if df.empty:
        print("ERROR: No evaluation data found. Run evaluations first.")
        return 1

    print(f"Loaded {len(df)} evaluation rows from {df['model_name'].nunique()} models")
    print(f"Output directory: {output_dir}")
    print()

    all_generated = []

    if args.phase is None or args.phase == 1:
        print("=== Generating Phase 1 visualizations ===")
        generated = generate_phase1(df, output_dir)
        all_generated.extend(generated)
        print(f"  Generated {len(generated)} files")
        print()

    if args.phase is None or args.phase == 2:
        print("=== Generating Phase 2 visualizations ===")
        generated = generate_phase2(df, output_dir)
        all_generated.extend(generated)
        print(f"  Generated {len(generated)} files")
        print()

    if args.phase is None or args.phase == 3:
        print("=== Generating Phase 3 visualizations ===")
        generated = generate_phase3(df, output_dir)
        all_generated.extend(generated)
        print(f"  Generated {len(generated)} files")
        print()

    print("=" * 60)
    print(f"Total: {len(all_generated)} visualizations generated")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for path in all_generated:
        print(f"  - {path.name}")

    return 0


if __name__ == "__main__":
    exit(main())
