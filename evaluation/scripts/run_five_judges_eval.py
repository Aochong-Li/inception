#!/usr/bin/env python3
"""
Run safety evaluation on 3 target models using multiple judge models via OpenRouter.

Reuses the same 50-sample subset (25 bio + 25 chem) from eval_three_models_results.
Each judge model evaluates all 3 target models with the full bio/chem template pipeline.

Usage:
    python evaluation/run_five_judges_eval.py                    # Run all judges
    python evaluation/run_five_judges_eval.py --judge gpt-5.2    # Run single judge
"""
import json
import os
import subprocess
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENROUTER_API_KEY"):
    print("ERROR: OPENROUTER_API_KEY not set. Add it to .env and retry.")
    sys.exit(1)

import pandas as pd

_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent
sys.path.insert(0, str(_parent_dir))

from evaluation.preprocess_results import get_full_trace

DATA_DIR = _parent_dir / "max_iterations_5"
EXISTING_RESULTS = _eval_dir / "archived" / "eval_three_models_results"
RESULTS_BASE = _eval_dir / "eval_five_judges"
SAMPLE_FILE = EXISTING_RESULTS / "sample_indices.json"

TARGET_MODELS = [
    "DeepSeek-V3.2",
    "Qwen3-235B-A22B-Thinking-2507",
    "GLM-4.6",
]

# Judge models: nick_name -> OpenRouter model ID
JUDGE_MODELS = {
    "gpt-5.2": "openai/gpt-5.2",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    "glm-5": "z-ai/glm-5",
}


def ensure_preprocessed(model_name: str, all_indices: list) -> Path:
    """Create or reuse preprocessed pickle for a target model."""
    # First check if existing preprocessed file can be reused
    existing = EXISTING_RESULTS / model_name / f"{model_name}_preprocessed.pickle"
    if existing.exists():
        return existing

    # Otherwise create from raw data
    input_path = DATA_DIR / f"{model_name}.pickle"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    df = pd.read_pickle(input_path)
    df_filtered = df.loc[all_indices].copy()
    df_filtered["reasoning_traces"] = df_filtered.apply(get_full_trace, axis=1)

    # Save to a shared location
    out_dir = RESULTS_BASE / "_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_preprocessed.pickle"
    df_filtered.to_pickle(out_path)
    print(f"Created {out_path} ({len(df_filtered)} rows)")
    return out_path


def run_judge_eval(
    judge_nick: str,
    judge_model_id: str,
    target_model: str,
    preprocessed_path: Path,
    overwrite: bool = False,
    max_consecutive_refusals: int = 0,
):
    """Run safety-judge.py for one judge × target combination."""
    output_dir = RESULTS_BASE / judge_nick / target_model
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / f"{target_model}_safety_judge.pickle"
    if combined_path.exists() and not overwrite:
        print(f"  Results exist: {combined_path}, skipping (use --overwrite to force)")
        return True

    cmd = [
        sys.executable,
        str(_eval_dir / "safety-judge.py"),
        "--input_filepath", str(preprocessed_path),
        "--output_dir", str(output_dir),
        "--nick_name", target_model,
        "--eval_model", judge_model_id,
        "--client_name", "openrouter",
        "--inquiry_col", "inquiry",
        "--reasoning_trace_col", "reasoning_traces",
        "--category_col", "category",
        "--chemical_label", "wmdp-chem",
        "--bioterrorism_label", "wmdp-bio",
        "--rate_limit", "3.0",
        "--max_validation_retries", "3",
        "--max_tokens", "16000",
    ]
    if overwrite:
        cmd.append("--overwrite")
    if max_consecutive_refusals > 0:
        cmd.extend(["--max_consecutive_refusals", str(max_consecutive_refusals)])

    print(f"  Running: {judge_nick} -> {target_model}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    print(f"  Done: {target_model}")
    return True


def run_judge_all_targets(
    judge_nick: str,
    judge_model_id: str,
    targets: list,
    preprocessed: dict,
    overwrite: bool,
    max_consecutive_refusals: int = 0,
) -> str:
    """Run all targets for a single judge. Called in a subprocess via ProcessPoolExecutor.

    preprocessed values may be str or Path — both are accepted by run_judge_eval.
    """
    for target in targets:
        run_judge_eval(
            judge_nick=judge_nick,
            judge_model_id=judge_model_id,
            target_model=target,
            preprocessed_path=Path(preprocessed[target]),
            overwrite=overwrite,
            max_consecutive_refusals=max_consecutive_refusals,
        )
    return judge_nick


def run_aggregation(judge_nick: str):
    """Run aggregate_results.py and danger_score.py for a judge's results."""
    results_dir = RESULTS_BASE / judge_nick

    print(f"\nAggregating results for judge: {judge_nick}")
    subprocess.run(
        [
            sys.executable,
            str(_script_dir / "aggregate_results.py"),
            "--results-dir", str(results_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(_eval_dir / "danger_score.py"),
            "--results-dir", str(results_dir),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run safety evaluation with multiple judge models via OpenRouter"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        choices=list(JUDGE_MODELS.keys()),
        help="Run only this judge model (default: all)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=TARGET_MODELS,
        help="Run only this target model (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip running aggregation scripts after evaluation",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run judges in parallel using ProcessPoolExecutor (default: enabled)",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Run judges sequentially instead of in parallel",
    )
    parser.add_argument(
        "--max-consecutive-refusals",
        type=int,
        default=5,
        help="Stop a category after N consecutive judge refusals (0 = disabled, default: 5)",
    )
    args = parser.parse_args()

    # Load sample indices
    with open(SAMPLE_FILE) as f:
        sample_indices = json.load(f)
    all_indices = sample_indices["chem"] + sample_indices["bio"]

    # Copy sample_indices.json to output dir
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    out_sample = RESULTS_BASE / "sample_indices.json"
    if not out_sample.exists():
        import shutil
        shutil.copy2(SAMPLE_FILE, out_sample)

    # Determine which judges and targets to run
    judges = {args.judge: JUDGE_MODELS[args.judge]} if args.judge else JUDGE_MODELS
    targets = [args.target] if args.target else TARGET_MODELS

    # Ensure preprocessed data exists for all targets
    preprocessed = {}
    for target in targets:
        preprocessed[target] = ensure_preprocessed(target, all_indices)

    # Run evaluations
    if args.parallel and len(judges) > 1:
        print(f"\nRunning {len(judges)} judges in parallel (one process per judge).")
        with ProcessPoolExecutor(max_workers=len(judges)) as pool:
            futures = {
                pool.submit(
                    run_judge_all_targets,
                    nick,
                    mid,
                    targets,
                    {t: str(p) for t, p in preprocessed.items()},
                    args.overwrite,
                    args.max_consecutive_refusals,
                ): nick
                for nick, mid in judges.items()
            }
            for fut in as_completed(futures):
                judge_nick = futures[fut]
                try:
                    fut.result()
                    print(f"\nJudge {judge_nick} completed.")
                    if not args.skip_aggregation:
                        run_aggregation(judge_nick)
                except Exception as exc:
                    print(f"\nJudge {judge_nick} FAILED: {exc}")
    else:
        # Sequential fallback (also used when a single judge is selected)
        for judge_nick, judge_model_id in judges.items():
            print(f"\n{'='*60}")
            print(f"Judge: {judge_nick} ({judge_model_id})")
            print(f"{'='*60}")

            all_ok = True
            for target in targets:
                ok = run_judge_eval(
                    judge_nick=judge_nick,
                    judge_model_id=judge_model_id,
                    target_model=target,
                    preprocessed_path=preprocessed[target],
                    overwrite=args.overwrite,
                    max_consecutive_refusals=args.max_consecutive_refusals,
                )
                if not ok:
                    all_ok = False

            if all_ok and not args.skip_aggregation:
                run_aggregation(judge_nick)

    print(f"\n{'='*60}")
    print("All evaluations complete.")
    print(f"Results saved to: {RESULTS_BASE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
