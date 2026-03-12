#!/usr/bin/env python3
"""
Run five-judge evaluation for three target models.

Evaluates DeepSeek-V3.2, Qwen3-235B-A22B-Thinking-2507, and GLM-4.6 using
five judge models via OpenRouter, on the 50 shared sample indices.

Output structure:
  evaluation/eval_five_judges/<judge_safe_name>/<target_model>/  (new runs)
    {target_model}_preprocessed.pickle
    {target_model}_safety_judge.pickle
    bioterrorism_safety_judge_raw.pickle
    chemical_safety_judge_raw.pickle
"""
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load API keys from core/.env explicitly
_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.parent.resolve()
_eval_dir = _script_dir.parent
load_dotenv(_parent_dir / "core" / ".env")

import pandas as pd

sys.path.insert(0, str(_parent_dir))

from evaluation.preprocess_results import get_full_trace

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = _parent_dir / "max_iterations_5"
SAMPLE_FILE = _eval_dir / "archived" / "eval_three_models_results" / "sample_indices.json"
OUTPUT_BASE = _eval_dir / "eval_five_judges"

TARGET_MODELS = [
    "DeepSeek-V3.2",
    "Qwen3-235B-A22B-Thinking-2507",
    "GLM-4.6",
]

# OpenRouter model IDs  →  (safe directory name, max_tokens)
# All judges use 16000 for uniformity and to avoid truncation.
# Thinking models (kimi-k2.5, glm-5) spend tokens on internal chain-of-thought.
JUDGE_MODELS = {
    "openai/gpt-5.2":               ("openai_gpt-5.2",               16000),
    "anthropic/claude-sonnet-4.6":  ("anthropic_claude-sonnet-4.6",  16000),
    "anthropic/claude-sonnet-4.5":  ("anthropic_claude-sonnet-4.5",  16000),
    "moonshotai/kimi-k2.5":         ("moonshotai_kimi-k2.5",         16000),
    "z-ai/glm-5":                   ("z-ai_glm-5",                   16000),
}

SAFETY_JUDGE_SCRIPT = _eval_dir / "safety-judge.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sample_indices() -> list:
    with open(SAMPLE_FILE) as f:
        data = json.load(f)
    return data["chem"] + data["bio"]


def preprocess_model(model_name: str, all_indices: list, output_dir: Path) -> Path:
    """Load model pickle, filter to sample indices, add reasoning_traces, save."""
    input_path = DATA_DIR / f"{model_name}.pickle"
    if not input_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {input_path}")

    df = pd.read_pickle(input_path)
    df_filtered = df.loc[all_indices].copy()
    df_filtered["reasoning_traces"] = df_filtered.apply(get_full_trace, axis=1)

    preprocessed_path = output_dir / f"{model_name}_preprocessed.pickle"
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_pickle(preprocessed_path)
    print(f"  Preprocessed: {preprocessed_path} ({len(df_filtered)} rows)")
    return preprocessed_path


def run_safety_judge(
    preprocessed_path: Path,
    output_dir: Path,
    model_name: str,
    judge_model: str,
    max_tokens: int = 16000,
) -> int:
    """Invoke safety-judge.py as a subprocess. Returns exit code."""
    cmd = [
        sys.executable,
        str(SAFETY_JUDGE_SCRIPT),
        "--input_filepath", str(preprocessed_path),
        "--output_dir", str(output_dir),
        "--nick_name", model_name,
        "--eval_model", judge_model,
        "--client_name", "openrouter",
        "--inquiry_col", "inquiry",
        "--reasoning_trace_col", "reasoning_traces",
        "--category_col", "category",
        "--chemical_label", "wmdp-chem",
        "--bioterrorism_label", "wmdp-bio",
        "--rate_limit", "3.0",
        "--max_validation_retries", "3",
        "--max_consecutive_refusals", "15",
        "--max_tokens", str(max_tokens),
    ]
    print(f"  Running: {judge_model} × {model_name} (max_tokens={max_tokens})")
    result = subprocess.run(cmd)
    return result.returncode


def validate_result(output_dir: Path, model_name: str, judge_safe: str) -> dict:
    """Load the combined safety_judge pickle and print a validation summary."""
    pickle_path = output_dir / f"{model_name}_safety_judge.pickle"
    info = {
        "judge": judge_safe,
        "target": model_name,
        "exists": pickle_path.exists(),
        "n_rows": 0,
        "n_valid": 0,
        "complied_rate": None,
        "cols_ok": False,
    }

    if not pickle_path.exists():
        print(f"  [MISSING] {pickle_path}")
        return info

    try:
        df = pd.read_pickle(pickle_path)
        info["n_rows"] = len(df)

        required_cols = ["complied", "specificity", "convincingness", "raw_response"]
        info["cols_ok"] = all(c in df.columns for c in required_cols)

        if "raw_response" in df.columns:
            info["n_valid"] = int(df["raw_response"].notna().sum())

        if "complied" in df.columns:
            valid_complied = df["complied"].dropna()
            if len(valid_complied) > 0:
                info["complied_rate"] = float(valid_complied.mean())

        status = "OK" if info["cols_ok"] and info["n_rows"] == 50 else "WARN"
        rate_str = f"{info['complied_rate']:.3f}" if info["complied_rate"] is not None else "N/A"
        print(
            f"  [{status}] {judge_safe} x {model_name}: "
            f"{info['n_rows']} rows, {info['n_valid']} valid responses, "
            f"complied_rate={rate_str}"
        )
    except Exception as exc:
        print(f"  [ERROR] Failed to load {pickle_path}: {exc}")

    return info


def run_aggregation(judge_dir: Path) -> None:
    """Run aggregate_results.py for a judge's results directory."""
    agg_script = _script_dir / "aggregate_results.py"
    if not agg_script.exists():
        print(f"  aggregate_results.py not found, skipping aggregation for {judge_dir.name}")
        return
    print(f"  Running aggregation for {judge_dir.name}...")
    result = subprocess.run(
        [sys.executable, str(agg_script), "--results-dir", str(judge_dir)]
    )
    if result.returncode != 0:
        print(f"  [WARN] Aggregation failed for {judge_dir.name} (exit {result.returncode})")
    else:
        print(f"  Aggregation done for {judge_dir.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_indices = load_sample_indices()
    print(f"Loaded {len(all_indices)} sample indices ({len(all_indices)//2} chem + {len(all_indices)//2} bio)\n")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    summary = []

    for judge_model_id, (judge_safe, max_tokens) in JUDGE_MODELS.items():
        judge_dir = OUTPUT_BASE / judge_safe
        judge_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Judge: {judge_model_id}  (dir: {judge_safe}, max_tokens={max_tokens})")
        print(f"{'='*70}")

        for target_model in TARGET_MODELS:
            target_dir = judge_dir / target_model
            target_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Target: {target_model}")

            # Step 1: Preprocess (skip if already done)
            preprocessed_path = target_dir / f"{target_model}_preprocessed.pickle"
            if preprocessed_path.exists():
                print(f"  Preprocessed pickle already exists, reusing: {preprocessed_path}")
            else:
                try:
                    preprocessed_path = preprocess_model(target_model, all_indices, target_dir)
                except FileNotFoundError as exc:
                    print(f"  [SKIP] {exc}")
                    continue

            # Check if result already exists (skip completed runs)
            result_pickle = target_dir / f"{target_model}_safety_judge.pickle"
            if result_pickle.exists():
                print(f"  Result already exists, validating: {result_pickle}")
                info = validate_result(target_dir, target_model, judge_safe)
                summary.append(info)
                continue

            # Step 2: Run safety judge
            exit_code = run_safety_judge(
                preprocessed_path=preprocessed_path,
                output_dir=target_dir,
                model_name=target_model,
                judge_model=judge_model_id,
                max_tokens=max_tokens,
            )
            if exit_code != 0:
                print(f"  [WARN] safety-judge exited with code {exit_code} for {judge_model_id} x {target_model}")

            # Step 3: Validate
            info = validate_result(target_dir, target_model, judge_safe)
            summary.append(info)

        # Step 4: Aggregate per judge
        run_aggregation(judge_dir)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    total = len(summary)
    ok = sum(1 for s in summary if s["exists"] and s["n_rows"] == 50 and s["cols_ok"])
    warn = sum(1 for s in summary if s["exists"] and (s["n_rows"] != 50 or not s["cols_ok"]))
    missing = sum(1 for s in summary if not s["exists"])
    print(f"  Total combinations: {total}")
    print(f"  OK (50 rows + all cols): {ok}")
    print(f"  WARN (exists but issues): {warn}")
    print(f"  MISSING: {missing}")
    print()
    for s in summary:
        status = "OK" if s["exists"] and s["n_rows"] == 50 and s["cols_ok"] else (
            "WARN" if s["exists"] else "MISSING"
        )
        rate = f"{s['complied_rate']:.3f}" if s["complied_rate"] is not None else "N/A"
        print(
            f"  [{status}] {s['judge']} x {s['target']}: "
            f"{s['n_rows']} rows, {s['n_valid']} valid, complied={rate}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
