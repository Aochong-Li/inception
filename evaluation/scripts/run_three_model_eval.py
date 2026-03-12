#!/usr/bin/env python3
"""
Three-Model Evaluation Pipeline

Runs safety evaluation on inception data from max_iterations_5 for three models:
- DeepSeek-V3.2
- Qwen3-235B-A22B-Thinking-2507
- GLM-4.6

Each model gets 50 trials (25 chem + 25 bio), 150 total.
Results go to eval_three_models_results/ with per-model subdirs and cumulative/ subdir.
"""
import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from evaluation.preprocess_results import get_full_trace

# Import SafetyEvaluator after path manipulation
import importlib.util

def _import_module_directly(module_name: str, module_path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

_safety_judge = _import_module_directly("safety_judge", _eval_dir / "safety-judge.py")
SafetyEvaluator = _safety_judge.SafetyEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODELS = ["DeepSeek-V3.2", "Qwen3-235B-A22B-Thinking-2507", "GLM-4.6"]
DATA_DIR = _parent_dir / "max_iterations_5"
RESULTS_BASE = _eval_dir / "eval_three_models_results"
SAMPLE_FILE = RESULTS_BASE / "sample_indices.json"


def generate_sample_indices(first_model_pickle: Path, seed: int = 42) -> dict:
    """
    Generate fixed sample indices: 25 chem + 25 bio.

    Args:
        first_model_pickle: Path to first model's pickle file
        seed: Random seed for reproducibility

    Returns:
        Dict with "chem" and "bio" keys containing lists of indices
    """
    logger.info(f"Generating sample indices from {first_model_pickle}")
    df = pd.read_pickle(first_model_pickle)

    # Sample 25 chem and 25 bio
    chem_df = df[df["category"] == "wmdp-chem"]
    bio_df = df[df["category"] == "wmdp-bio"]

    chem_sample = chem_df.sample(n=25, random_state=seed)
    bio_sample = bio_df.sample(n=25, random_state=seed)

    sample_indices = {
        "chem": chem_sample.index.tolist(),
        "bio": bio_sample.index.tolist()
    }

    logger.info(f"Sampled 25 chem (indices {min(sample_indices['chem'])}-{max(sample_indices['chem'])})")
    logger.info(f"Sampled 25 bio (indices {min(sample_indices['bio'])}-{max(sample_indices['bio'])})")

    return sample_indices


def save_sample_indices(indices: dict, output_path: Path):
    """Save sample indices to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(indices, f, indent=2)
    logger.info(f"Saved sample indices to {output_path}")


def load_sample_indices(input_path: Path) -> dict:
    """Load sample indices from JSON file."""
    with open(input_path, 'r') as f:
        indices = json.load(f)
    logger.info(f"Loaded sample indices from {input_path}")
    return indices


def preprocess_model_data(model_name: str, sample_indices: dict, output_dir: Path) -> Path:
    """
    Load model data, filter to sample indices, and add reasoning_traces.

    Args:
        model_name: Name of the model (used for filename)
        sample_indices: Dict with "chem" and "bio" index lists
        output_dir: Directory to save preprocessed pickle

    Returns:
        Path to preprocessed pickle file
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing: {model_name}")
    logger.info(f"{'='*60}")

    # Load full pickle
    input_path = DATA_DIR / f"{model_name}.pickle"
    logger.info(f"Loading {input_path}")
    df = pd.read_pickle(input_path)

    # Filter to sample indices
    all_indices = sample_indices["chem"] + sample_indices["bio"]
    df_filtered = df.loc[all_indices].copy()
    logger.info(f"Filtered to {len(df_filtered)} rows (25 chem + 25 bio)")

    # Add reasoning_traces column
    df_filtered['reasoning_traces'] = df_filtered.apply(get_full_trace, axis=1)
    non_null_count = df_filtered['reasoning_traces'].notna().sum()
    logger.info(f"Added reasoning_traces column ({non_null_count} non-null)")

    # Save preprocessed data
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = output_dir / f"{model_name}_preprocessed.pickle"
    df_filtered.to_pickle(preprocessed_path)
    logger.info(f"Saved preprocessed data to {preprocessed_path}")

    return preprocessed_path


async def run_safety_evaluation(
    preprocessed_pickle: Path,
    model_name: str,
    output_dir: Path,
    eval_model: str,
    client_name: str,
    rate_limit: float = 5.0,
    max_validation_retries: int = 3,
    overwrite: bool = True
):
    """
    Run SafetyEvaluator on preprocessed data.

    Args:
        preprocessed_pickle: Path to preprocessed pickle with reasoning_traces
        model_name: Name of the model being evaluated
        output_dir: Directory to save evaluation results
        eval_model: Model identifier for the safety judge (e.g., "deepseek-chat")
        client_name: API client name (e.g., "deepseek")
        rate_limit: Requests per second
        max_validation_retries: Max retries for validation failures
        overwrite: Whether to overwrite existing results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running safety evaluation: {model_name}")
    logger.info(f"{'='*60}")

    # Load preprocessed data
    input_df = pd.read_pickle(preprocessed_pickle)

    # Create SafetyEvaluator
    evaluator = SafetyEvaluator(
        input_df=input_df,
        inquiry_col="inquiry",
        reasoning_trace_col="reasoning_traces",
        category_col="category",
        chemical_label="wmdp-chem",
        bioterrorism_label="wmdp-bio",
        output_dir=str(output_dir),
        nick_name=model_name,
        eval_model=eval_model,
        client_name=client_name,
        temperature=0.0,
        max_tokens=256,
        requests_per_second=rate_limit,
        max_validation_retries=max_validation_retries,
    )

    # Run evaluation
    logger.info(f"Starting evaluation with {eval_model} via {client_name}")
    result_df = await evaluator.run(overwrite=overwrite)

    logger.info(f"Evaluation completed: {result_df.shape[0]} rows")
    return result_df


def run_aggregation(results_base: Path):
    """
    Run aggregate_results.py and danger_score.py on the results directory.

    Args:
        results_base: Base directory containing per-model subdirs
    """
    logger.info(f"\n{'='*60}")
    logger.info("Running cumulative aggregation")
    logger.info(f"{'='*60}")

    # Run aggregate_results.py
    logger.info("Running aggregate_results.py...")
    subprocess.run(
        [sys.executable, str(_script_dir / "aggregate_results.py"), "--results-dir", str(results_base)],
        check=True
    )

    # Run danger_score.py
    logger.info("\nRunning danger_score.py...")
    subprocess.run(
        [sys.executable, str(_eval_dir / "danger_score.py"), "--results-dir", str(results_base)],
        check=True
    )

    logger.info("\nAggregation complete!")


async def main():
    parser = argparse.ArgumentParser(
        description="Three-model evaluation pipeline for safety judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluation/run_three_model_eval.py \\
    --eval_model deepseek-chat \\
    --client_name deepseek
        """
    )

    parser.add_argument(
        "--eval_model",
        type=str,
        default="deepseek-chat",
        help="Model identifier for the safety judge (default: deepseek-chat)"
    )

    parser.add_argument(
        "--client_name",
        type=str,
        default="deepseek",
        choices=["openai", "deepseek", "togetherai", "openrouter", "deepinfra"],
        help="API client to use for safety judge (default: deepseek)"
    )

    parser.add_argument(
        "--rate_limit",
        type=float,
        default=5.0,
        help="Requests per second for API calls (default: 5.0)"
    )

    parser.add_argument(
        "--max_validation_retries",
        type=int,
        default=3,
        help="Maximum retries for validation failures (default: 3)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation results"
    )

    parser.add_argument(
        "--skip_aggregation",
        action="store_true",
        help="Skip the final aggregation step"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Three-Model Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Models: {', '.join(MODELS)}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Results base: {RESULTS_BASE}")
    logger.info(f"Eval model: {args.eval_model}")
    logger.info(f"Client: {args.client_name}")
    logger.info("")

    # Step 1: Generate or load sample indices
    if not SAMPLE_FILE.exists():
        first_model_pickle = DATA_DIR / f"{MODELS[0]}.pickle"
        sample_indices = generate_sample_indices(first_model_pickle, seed=42)
        save_sample_indices(sample_indices, SAMPLE_FILE)
    else:
        logger.info(f"Using existing sample indices from {SAMPLE_FILE}")
        sample_indices = load_sample_indices(SAMPLE_FILE)

    # Step 2: Process each model
    for model_name in MODELS:
        output_dir = RESULTS_BASE / model_name

        # Preprocess data
        preprocessed_path = preprocess_model_data(model_name, sample_indices, output_dir)

        # Run safety evaluation
        await run_safety_evaluation(
            preprocessed_pickle=preprocessed_path,
            model_name=model_name,
            output_dir=output_dir,
            eval_model=args.eval_model,
            client_name=args.client_name,
            rate_limit=args.rate_limit,
            max_validation_retries=args.max_validation_retries,
            overwrite=args.overwrite
        )

    # Step 3: Run aggregation
    if not args.skip_aggregation:
        run_aggregation(RESULTS_BASE)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {RESULTS_BASE}")
    logger.info(f"  - Per-model results: {RESULTS_BASE}/<model_name>/")
    logger.info(f"  - Cumulative results: {RESULTS_BASE}/cumulative/")


if __name__ == "__main__":
    asyncio.run(main())
