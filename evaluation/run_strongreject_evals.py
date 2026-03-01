#!/usr/bin/env python3
"""
Run StrongREJECT-only evaluation for all model result pickles.

Uses the unified StrongREJECT template (category-agnostic) instead of
bio/chem-specific templates. Evaluates via DeepSeek API.

Requires DEEPSEEK_API_KEY in environment or .env.
"""
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.resolve()
sys.path.insert(0, str(_parent_dir))

from evaluation.preprocess_results import get_full_trace

DATA_DIR = _parent_dir / "max_iterations_5"
RESULTS_BASE = _script_dir / "evaluation-results"

# Discover all model pickles
MODELS = sorted(
    p.stem for p in DATA_DIR.glob("*.pickle")
    if not p.stem.startswith(".")
)


def main():
    if not MODELS:
        print(f"No model pickles found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(MODELS)} models: {MODELS}")

    for model_name in MODELS:
        input_path = DATA_DIR / f"{model_name}.pickle"
        if not input_path.exists():
            print(f"Skipping {model_name}: {input_path} not found")
            continue

        model_dir = RESULTS_BASE / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess: add reasoning_traces column
        df = pd.read_pickle(input_path)
        df["reasoning_traces"] = df.apply(get_full_trace, axis=1)
        preprocessed_path = model_dir / f"{model_name}_preprocessed.pickle"
        df.to_pickle(preprocessed_path)
        print(f"Preprocessed {model_name} ({len(df)} rows)")

        # Run safety-judge with --strongreject_only
        cmd = [
            sys.executable,
            str(_script_dir / "safety-judge.py"),
            "--input_filepath", str(preprocessed_path),
            "--output_dir", str(model_dir),
            "--nick_name", model_name,
            "--eval_model", "deepseek-chat",
            "--client_name", "deepseek",
            "--inquiry_col", "inquiry",
            "--reasoning_trace_col", "reasoning_traces",
            "--category_col", "category",
            "--rate_limit", "5.0",
            "--max_validation_retries", "3",
            "--strongreject_only",
        ]
        print(f"Running StrongREJECT eval for {model_name}...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Eval failed for {model_name} (exit {result.returncode})")
        else:
            print(f"Eval completed for {model_name}")

    # Run danger_score aggregation
    print("\nRunning danger_score.py...")
    subprocess.run(
        [sys.executable, str(_script_dir / "danger_score.py"), "--results-dir", str(RESULTS_BASE)],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
