#!/usr/bin/env python3
"""
Standalone script to re-evaluate failed safety evaluations across all model directories.

This script:
1. Iterates through all model directories in evaluation-results/
2. For each model, identifies failed evaluations (None or invalid JSON responses)
3. Re-runs safety-judge with rate limiting and validation
4. Backs up original results before overwriting
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import importlib.util

import pandas as pd

# Add parent directory to path
_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.resolve()
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


# Import openaiapi directly to avoid __init__.py triggering vllm import
def _import_module_directly(module_name: str, module_path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_openaiapi = _import_module_directly("openaiapi", _parent_dir / "core" / "openaiapi.py")
validate_safety_response = _openaiapi.validate_safety_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def find_model_directories(results_dir: str) -> List[Path]:
    """Find all model directories in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []

    model_dirs = []
    for item in results_path.iterdir():
        if item.is_dir():
            # Check if directory contains safety_judge.pickle files
            pickle_files = list(item.glob("*_safety_judge*.pickle"))
            if pickle_files:
                model_dirs.append(item)

    logger.info(f"Found {len(model_dirs)} model directories with evaluation results")
    return sorted(model_dirs)


def count_failures(pickle_path: Path, category: str) -> Dict[str, int]:
    """Count the number of failed evaluations in a pickle file."""
    if not pickle_path.exists():
        return {"total": 0, "failed": 0, "success": 0}

    df = pd.read_pickle(pickle_path)
    if df.empty:
        return {"total": 0, "failed": 0, "success": 0}

    total = len(df)
    failed = 0

    for idx, row in df.iterrows():
        response = row.get('response')
        if response is None:
            failed += 1
            continue

        response_str = response[0] if isinstance(response, list) else response
        if not validate_safety_response(response_str, category):
            failed += 1

    return {
        "total": total,
        "failed": failed,
        "success": total - failed
    }


def analyze_model_directory(model_dir: Path) -> Dict[str, Dict[str, int]]:
    """Analyze a model directory and return failure counts by category."""
    categories = ["bioterrorism", "chemical", "cybersecurity"]
    results = {}

    for category in categories:
        pickle_path = model_dir / f"{category}_safety_judge_raw.pickle"
        results[category] = count_failures(pickle_path, category)

    return results


def backup_results(model_dir: Path) -> None:
    """Backup all pickle files in a model directory."""
    backup_dir = model_dir / "backups"
    backup_dir.mkdir(exist_ok=True)

    for pickle_file in model_dir.glob("*.pickle"):
        if "backup" not in pickle_file.name:
            backup_path = backup_dir / f"{pickle_file.stem}_backup.pickle"
            shutil.copy2(pickle_file, backup_path)
            logger.info(f"Backed up {pickle_file.name} to {backup_path}")


def run_reeval_for_model(
    model_dir: Path,
    input_filepath: str,
    eval_model: str,
    client_name: str,
    rate_limit: float,
    max_retries: int,
    dry_run: bool = False
) -> bool:
    """Run re-evaluation for a single model directory."""
    if dry_run:
        logger.info(f"[DRY RUN] Would re-evaluate {model_dir.name}")
        return True

    cmd = [
        sys.executable,
        str(_script_dir / "safety-judge.py"),
        "--input_filepath", input_filepath,
        "--output_dir", str(model_dir),
        "--nick_name", model_dir.name,
        "--eval_model", eval_model,
        "--client_name", client_name,
        "--rate_limit", str(rate_limit),
        "--max_validation_retries", str(max_retries),
        "--reeval_only"
    ]

    logger.info(f"Running re-evaluation for {model_dir.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Re-evaluation completed for {model_dir.name}")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Re-evaluation failed for {model_dir.name}: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def generate_summary_report(
    results_dir: str,
    before_stats: Dict[str, Dict[str, Dict[str, int]]],
    after_stats: Dict[str, Dict[str, Dict[str, int]]]
) -> str:
    """Generate a summary report of the re-evaluation."""
    lines = [
        "=" * 60,
        "Re-evaluation Summary Report",
        "=" * 60,
        ""
    ]

    total_before_failed = 0
    total_after_failed = 0
    total_fixed = 0

    for model_name in sorted(before_stats.keys()):
        lines.append(f"Model: {model_name}")
        lines.append("-" * 40)

        for category in ["bioterrorism", "chemical", "cybersecurity"]:
            before = before_stats[model_name].get(category, {})
            after = after_stats.get(model_name, {}).get(category, {})

            before_failed = before.get("failed", 0)
            after_failed = after.get("failed", 0)
            fixed = before_failed - after_failed

            total_before_failed += before_failed
            total_after_failed += after_failed
            total_fixed += fixed

            if before_failed > 0:
                lines.append(f"  {category}: {before_failed} -> {after_failed} failed ({fixed} fixed)")

        lines.append("")

    lines.extend([
        "=" * 60,
        "Overall Summary",
        "=" * 60,
        f"Total failures before: {total_before_failed}",
        f"Total failures after: {total_after_failed}",
        f"Total fixed: {total_fixed}",
        f"Improvement: {(total_fixed / max(1, total_before_failed)) * 100:.1f}%"
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate failed safety evaluations across all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python reeval_failed.py \\
    --results_dir ./evaluation-results \\
    --input_filepath ./data/preprocessed.pickle \\
    --eval_model gpt-4o-mini \\
    --client_name openai \\
    --rate_limit 5 \\
    --max_retries 3
        """
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing model evaluation results"
    )

    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Path to the input pickle file with preprocessed data"
    )

    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for evaluation (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--client_name",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "togetherai", "openrouter", "deepinfra"],
        help="API client to use (default: openai)"
    )

    parser.add_argument(
        "--rate_limit",
        type=float,
        default=5.0,
        help="Rate limit in requests per second (default: 5.0)"
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for validation failures (default: 3)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Analyze failures without running re-evaluation"
    )

    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Skip backing up original results"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific model directories to process (default: all)"
    )

    args = parser.parse_args()

    # Find model directories
    model_dirs = find_model_directories(args.results_dir)
    if not model_dirs:
        logger.error("No model directories found")
        sys.exit(1)

    # Filter to specific models if requested
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]
        if not model_dirs:
            logger.error(f"No matching model directories found for: {args.models}")
            sys.exit(1)

    # Analyze current state
    logger.info("Analyzing current failure rates...")
    before_stats = {}
    for model_dir in model_dirs:
        before_stats[model_dir.name] = analyze_model_directory(model_dir)
        failures = sum(
            stats["failed"]
            for stats in before_stats[model_dir.name].values()
        )
        if failures > 0:
            logger.info(f"  {model_dir.name}: {failures} total failures")

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN - No changes will be made")
        logger.info("=" * 60)
        for model_name, stats in before_stats.items():
            for category, counts in stats.items():
                if counts["failed"] > 0:
                    logger.info(
                        f"  {model_name}/{category}: "
                        f"{counts['failed']}/{counts['total']} failed "
                        f"({counts['failed']/max(1, counts['total'])*100:.1f}%)"
                    )
        sys.exit(0)

    # Backup and run re-evaluation
    for model_dir in model_dirs:
        stats = before_stats[model_dir.name]
        total_failures = sum(s["failed"] for s in stats.values())

        if total_failures == 0:
            logger.info(f"Skipping {model_dir.name} - no failures to re-evaluate")
            continue

        if not args.no_backup:
            backup_results(model_dir)

        success = run_reeval_for_model(
            model_dir=model_dir,
            input_filepath=args.input_filepath,
            eval_model=args.eval_model,
            client_name=args.client_name,
            rate_limit=args.rate_limit,
            max_retries=args.max_retries,
            dry_run=False
        )

        if not success:
            logger.warning(f"Re-evaluation failed for {model_dir.name}, continuing...")

    # Analyze final state
    logger.info("Analyzing final state...")
    after_stats = {}
    for model_dir in model_dirs:
        after_stats[model_dir.name] = analyze_model_directory(model_dir)

    # Generate and print summary report
    report = generate_summary_report(args.results_dir, before_stats, after_stats)
    print("\n" + report)

    # Save report to file
    report_path = Path(args.results_dir) / "reeval_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
