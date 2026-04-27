"""
Apply the patched _parse_json_responses to every qwen *_safety_judge.pickle
in place. Only the parsed-output columns are overwritten; raw_response,
inputs, and any non-judge columns are left untouched.

Reports per-pickle: parse-fail rows before, after, recovered.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location("safety_judge", EVAL_DIR / "safety-judge.py")
sj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sj)

PARSED_COLS = [
    "raw_response", "analysis", "agent_class", "agent_grade",
    "delivery_method", "exposure_environment", "severity_level", "fatalities",
    "complied", "late_refusal", "specificity", "convincingness", "truncated",
    "threat", "contagiousness", "route", "scale",
]


def make_evaluator():
    stub = pd.DataFrame({"inquiry": ["x"], "reasoning_traces": ["x"], "category": ["wmdp-bio"]})
    return sj.SafetyEvaluator(
        input_df=stub, inquiry_col="inquiry", reasoning_trace_col="reasoning_traces",
        category_col="category", chemical_label="wmdp-chem", bioterrorism_label="wmdp-bio",
        output_dir="/tmp/_reparse_stub", nick_name="stub",
        eval_model="Qwen/Qwen3.5-397B-A17B", client_name="deepinfra",
    )


def reparse_in_place(p: Path, ev) -> dict:
    df = pd.read_pickle(p)
    if "raw_response" not in df.columns:
        return {"path": str(p), "skipped": "no raw_response col"}
    sev_before = int(df["severity_level"].notna().sum()) if "severity_level" in df.columns else 0

    work = df.copy()
    work["response"] = work["raw_response"]  # parser reads 'response'
    if "finish_reason" not in work.columns:
        work["finish_reason"] = None
    parsed = ev._parse_json_responses(work[["response", "finish_reason"]])

    for col in PARSED_COLS:
        if col in parsed.columns:
            df[col] = parsed[col]
    df.to_pickle(p)
    sev_after = int(df["severity_level"].notna().sum())
    return {"path": str(p.relative_to(EVAL_DIR)), "rows": len(df),
            "sev_before": sev_before, "sev_after": sev_after,
            "recovered": sev_after - sev_before}


def main():
    ev = make_evaluator()
    qwen_dir = EVAL_DIR / "eval_qwen397b_judge"
    pickles = sorted(qwen_dir.rglob("*_safety_judge.pickle"))
    print(f"Re-parsing {len(pickles)} pickles in {qwen_dir.relative_to(PROJECT_ROOT)} ...")
    print()
    total_recovered = 0
    for p in pickles:
        r = reparse_in_place(p, ev)
        if "skipped" in r:
            continue
        if r["recovered"] > 0:
            print(f"  +{r['recovered']:>3d}  ({r['sev_before']:>3d}→{r['sev_after']:>3d})  {r['path']}")
            total_recovered += r["recovered"]
    print(f"\n=== Done ===  Recovered {total_recovered} rows across qwen pickles in place")


if __name__ == "__main__":
    main()
