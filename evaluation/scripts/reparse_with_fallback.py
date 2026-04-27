"""
Two operations using the patched safety-judge parser:

  1. Re-parse evaluation/rerun_failures/rerun_results/deepseek_rerun.pickle
     in place (overwrite parsed columns from the same raw_response).

  2. Walk every *_safety_judge.pickle in both judge dirs, find rows with a
     present raw_response but null severity_level (parse-fails), apply the
     patched parser to each, and report how many would recover.
     DOES NOT save anything for the qwen survey — read-only assessment.
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
SafetyEvaluator = sj.SafetyEvaluator


def _make_parser():
    """Cheap stub instance just to access _parse_json_responses."""
    stub = pd.DataFrame({"inquiry": ["x"], "reasoning_traces": ["x"], "category": ["wmdp-bio"]})
    return SafetyEvaluator(
        input_df=stub, inquiry_col="inquiry", reasoning_trace_col="reasoning_traces",
        category_col="category", chemical_label="wmdp-chem", bioterrorism_label="wmdp-bio",
        output_dir="/tmp/_reparse_stub", nick_name="stub",
        eval_model="deepseek-chat", client_name="deepseek",
    )


PARSED_COLS = [
    "raw_response", "analysis", "agent_class", "agent_grade",
    "delivery_method", "exposure_environment", "severity_level", "fatalities",
    "complied", "late_refusal", "specificity", "convincingness", "truncated",
    "threat", "contagiousness", "route", "scale",
]


def reparse_pickle_in_place(pickle_path: Path, ev: SafetyEvaluator) -> dict:
    """Re-run the parser on a pickle's raw_response column; overwrite parsed cols."""
    df = pd.read_pickle(pickle_path)
    if "raw_response" not in df.columns:
        return {"path": str(pickle_path), "skipped": True, "reason": "no raw_response col"}

    work = df.copy()
    # The parser reads from 'response' (raw, before the unwrap). For re-parsing, raw_response
    # is already the unwrapped string, so feed it as 'response' directly.
    work["response"] = work["raw_response"]
    if "finish_reason" not in work.columns:
        work["finish_reason"] = None
    parsed = ev._parse_json_responses(work[["response", "finish_reason"]])

    # Overwrite parsed columns on the original df
    for col in PARSED_COLS:
        if col in parsed.columns:
            df[col] = parsed[col]
    df.to_pickle(pickle_path)

    sev_present = df["severity_level"].notna().sum() if "severity_level" in df.columns else 0
    return {"path": str(pickle_path), "rows": len(df), "sev_present": int(sev_present)}


def survey_parse_fails(judge_dir: Path, ev: SafetyEvaluator) -> dict:
    """For every pickle in judge_dir, count parse-fails and how many would recover."""
    pickles = sorted(judge_dir.rglob("*_safety_judge.pickle"))
    summary = {"pickles": 0, "rows_total": 0, "parse_fails": 0, "would_recover": 0,
               "by_pickle": []}
    for p in pickles:
        df = pd.read_pickle(p)
        if "raw_response" not in df.columns or "severity_level" not in df.columns:
            continue
        rr = df["raw_response"]
        rr_present = rr.notna() & (rr.fillna("").astype(str).str.strip() != "")
        sev_null   = df["severity_level"].isna()
        parse_fail_mask = rr_present & sev_null
        if not parse_fail_mask.any():
            continue
        # Try the patched parser on each parse-fail row
        sub = df.loc[parse_fail_mask, ["raw_response"]].copy()
        sub["response"] = sub["raw_response"]
        sub["finish_reason"] = None
        re_parsed = ev._parse_json_responses(sub[["response", "finish_reason"]])
        recovered = re_parsed["severity_level"].notna().sum() if "severity_level" in re_parsed.columns else 0

        rel = str(p.relative_to(judge_dir))
        summary["pickles"] += 1
        summary["rows_total"] += len(df)
        summary["parse_fails"] += int(parse_fail_mask.sum())
        summary["would_recover"] += int(recovered)
        summary["by_pickle"].append({
            "pickle": rel, "parse_fails": int(parse_fail_mask.sum()),
            "would_recover": int(recovered),
        })
    return summary


def main():
    ev = _make_parser()

    # ── Op 1: re-parse DS rerun results in place ────────────────────────────
    target = EVAL_DIR / "rerun_failures" / "rerun_results" / "deepseek_rerun.pickle"
    print(f"[OP1] Re-parsing {target.relative_to(PROJECT_ROOT)} in place ...")
    res = reparse_pickle_in_place(target, ev)
    print(f"      rows={res['rows']}  severity_level present in {res['sev_present']}/{res['rows']}")
    print()

    # ── Op 2: survey qwen parse-fails (no save) ─────────────────────────────
    for judge_dir in [EVAL_DIR / "eval_qwen397b_judge", EVAL_DIR / "eval_deepseek_chat_judge"]:
        print(f"[OP2] Surveying parse-fails in {judge_dir.relative_to(PROJECT_ROOT)} (no save) ...")
        s = survey_parse_fails(judge_dir, ev)
        print(f"      pickles affected: {s['pickles']}")
        print(f"      total parse-fail rows: {s['parse_fails']}")
        print(f"      would recover with patched parser: {s['would_recover']}/{s['parse_fails']}  "
              f"({100*s['would_recover']/max(1,s['parse_fails']):.1f}%)")
        print(f"      per-pickle breakdown (top 15 by parse-fail count):")
        s["by_pickle"].sort(key=lambda x: -x["parse_fails"])
        for row in s["by_pickle"][:15]:
            print(f"        recover {row['would_recover']:>3d}/{row['parse_fails']:<3d}  {row['pickle']}")
        print()


if __name__ == "__main__":
    main()
