#!/usr/bin/env python3
"""
Run reeval for the three-model eval results. Creates preprocessed pickles
and runs safety-judge --reeval_only for each model with failures.

Requires DEEPSEEK_API_KEY in environment or .env for API calls.
"""
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent
sys.path.insert(0, str(_parent_dir))

from evaluation.preprocess_results import get_full_trace

DATA_DIR = _eval_dir / "inception_data" / "max_iterations_5" / "think"
RESULTS_BASE = _eval_dir / "eval_three_models_results"
SAMPLE_FILE = RESULTS_BASE / "sample_indices.json"
MODELS = ["DeepSeek-V3.2", "Qwen3-235B-A22B-Thinking-2507", "GLM-4.6"]


def main():
    with open(SAMPLE_FILE) as f:
        sample_indices = json.load(f)
    all_indices = sample_indices["chem"] + sample_indices["bio"]

    for model_name in MODELS:
        model_dir = RESULTS_BASE / model_name
        if not model_dir.exists():
            continue

        # Create preprocessed pickle
        input_path = DATA_DIR / f"{model_name}.pickle"
        if not input_path.exists():
            print(f"Skipping {model_name}: {input_path} not found")
            continue

        df = pd.read_pickle(input_path)
        df_filtered = df.loc[all_indices].copy()
        df_filtered["reasoning_traces"] = df_filtered.apply(get_full_trace, axis=1)
        preprocessed_path = model_dir / f"{model_name}_preprocessed.pickle"
        df_filtered.to_pickle(preprocessed_path)
        print(f"Created {preprocessed_path} ({len(df_filtered)} rows)")

        # Run safety-judge --reeval_only
        cmd = [
            sys.executable,
            str(_eval_dir / "safety-judge.py"),
            "--input_filepath", str(preprocessed_path),
            "--output_dir", str(model_dir),
            "--nick_name", model_name,
            "--eval_model", "deepseek-chat",
            "--client_name", "deepseek",
            "--inquiry_col", "inquiry",
            "--reasoning_trace_col", "reasoning_traces",
            "--category_col", "category",
            "--chemical_label", "wmdp-chem",
            "--bioterrorism_label", "wmdp-bio",
            "--rate_limit", "5.0",
            "--max_validation_retries", "3",
            "--reeval_only",
        ]
        print(f"Running reeval for {model_name}...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Reeval failed for {model_name} (exit {result.returncode})")
        else:
            print(f"Reeval completed for {model_name}")

    # Run aggregation
    print("\nRunning aggregate_results.py...")
    subprocess.run(
        [sys.executable, str(_script_dir / "aggregate_results.py"), "--results-dir", str(RESULTS_BASE)],
        check=True,
    )
    print("Running danger_score.py...")
    subprocess.run(
        [sys.executable, str(_eval_dir / "danger_score.py"), "--results-dir", str(RESULTS_BASE)],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
