#!/usr/bin/env python3
"""
Batch evaluation pipeline for all models.

Samples 50 prompts (25 chem, 25 bio), evaluates all models using safety-judge,
and saves per-model results.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.resolve()
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import get_latest_iteration from preprocess_results.py in same directory
import importlib.util
_preprocess_spec = importlib.util.spec_from_file_location("preprocess_results", _script_dir / "preprocess_results.py")
_preprocess_module = importlib.util.module_from_spec(_preprocess_spec)
_preprocess_spec.loader.exec_module(_preprocess_module)
get_full_trace = _preprocess_module.get_full_trace

# Model configurations
MODELS = [
    "DeepSeek-R1-0528",
    "DeepSeek-V3.2",
    "GLM-4.6",
    "Kimi-K2-Thinking",
    "Qwen3-235B-A22B-Thinking-2507",
    "Qwen3-Next-80B-A3B-Thinking",
]

DATA_DIR = Path(_parent_dir) / "max_iterations_5"
RESULTS_DIR = Path(_script_dir) / "results"
SAMPLE_FILE = Path(_script_dir) / "sample_indices.json"


def generate_sample_indices(seed: int = 42) -> dict:
    """Generate fixed sample indices (25 chem, 25 bio) from first model."""
    ref_pickle = DATA_DIR / f"{MODELS[0]}.pickle"
    df = pd.read_pickle(ref_pickle)

    chem_indices = df[df["category"] == "wmdp-chem"].sample(25, random_state=seed).index.tolist()
    bio_indices = df[df["category"] == "wmdp-bio"].sample(25, random_state=seed).index.tolist()

    sample = {"chem": chem_indices, "bio": bio_indices, "seed": seed}

    with open(SAMPLE_FILE, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Generated sample indices: 25 chem, 25 bio (seed={seed})")
    print(f"Saved to: {SAMPLE_FILE}")
    return sample


def load_sample_indices() -> dict:
    """Load existing sample indices or generate new ones."""
    if SAMPLE_FILE.exists():
        with open(SAMPLE_FILE) as f:
            sample = json.load(f)
        print(f"Loaded existing sample indices from {SAMPLE_FILE}")
        return sample
    return generate_sample_indices()


def preprocess_model(model_name: str, sample_indices: list) -> Path:
    """Preprocess a model's pickle file for evaluation."""
    input_path = DATA_DIR / f"{model_name}.pickle"
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_preprocessed.pickle"

    df = pd.read_pickle(input_path)

    # Filter to sample indices
    df = df.loc[sample_indices]

    # Add reasoning_traces column by reconstructing full trace from all iterations
    df["reasoning_traces"] = df.apply(get_full_trace, axis=1)

    df.to_pickle(output_path)
    print(f"Preprocessed {model_name}: {len(df)} rows, {df['reasoning_traces'].notna().sum()} non-null")
    return output_path


def run_safety_judge(model_name: str, preprocessed_path: Path, eval_model: str, client_name: str):
    """Run safety-judge.py on a preprocessed model file using subprocess."""
    output_dir = RESULTS_DIR / model_name

    cmd = [
        sys.executable,
        str(_script_dir / "safety-judge.py"),
        "--input_filepath", str(preprocessed_path),
        "--output_dir", str(output_dir),
        "--nick_name", model_name,
        "--inquiry_col", "inquiry",
        "--reasoning_trace_col", "reasoning_traces",
        "--category_col", "category",
        "--bioterrorism_label", "wmdp-bio",
        "--chemical_label", "wmdp-chem",
        "--cybersecurity_label", "__NONE__",
        "--eval_model", eval_model,
        "--client_name", client_name,
        "--overwrite",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_parent_dir)

    print(f"Running safety-judge for {model_name}...")
    result = subprocess.run(cmd, env=env, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: safety-judge returned non-zero exit code for {model_name}")
    else:
        print(f"Completed evaluation for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation pipeline for all models")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--eval_model", default="gpt-4o-mini", help="Model to use for evaluation")
    parser.add_argument("--client_name", default="openai", help="API client to use")
    parser.add_argument("--regenerate_sample", action="store_true", help="Regenerate sample indices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate or load sample indices
    if args.regenerate_sample or not SAMPLE_FILE.exists():
        sample = generate_sample_indices(args.seed)
    else:
        sample = load_sample_indices()

    all_indices = sample["chem"] + sample["bio"]

    # Process each model
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        # Preprocess
        preprocessed_path = preprocess_model(model_name, all_indices)

        # Run safety evaluation
        run_safety_judge(model_name, preprocessed_path, args.eval_model, args.client_name)

    print(f"\n{'='*60}")
    print("Batch evaluation complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
