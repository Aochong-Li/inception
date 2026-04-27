"""
Walk every *_safety_judge.pickle under both judge dirs and collect rows that
are broken (raw_response missing OR severity_level missing). Each manifest row
carries source_pickle + source_idx so the rerun output can be patched back.

Outputs:
  evaluation/rerun_manifest/deepseek_failures.pickle  (judge = deepseek-chat)
  evaluation/rerun_manifest/qwen_failures.pickle      (judge = Qwen3.5-397B-A17B)
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "evaluation"
OUT_DIR = EVAL_DIR / "rerun_failures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JUDGES = {
    "DS": EVAL_DIR / "eval_deepseek_chat_judge",
    "QW": EVAL_DIR / "eval_qwen397b_judge",
}

# Columns we need from each row to re-judge it
INPUT_COLS = ["inquiry", "reasoning_traces", "category"]


def collect_failures(judge_label: str, judge_dir: Path):
    rows = []
    pickles = sorted(judge_dir.rglob("*_safety_judge.pickle"))
    pickles_with_failures = 0
    for p in pickles:
        df = pd.read_pickle(p)
        rr = df.get("raw_response")
        sev = df.get("severity_level")
        if rr is None:
            continue
        rr_missing = rr.isna() | (rr.fillna("").astype(str).str.strip() == "")
        sev_missing = sev.isna() if sev is not None else pd.Series([True] * len(df))
        broken_mask = rr_missing | sev_missing
        if not broken_mask.any():
            continue
        pickles_with_failures += 1
        broken = df.loc[broken_mask, INPUT_COLS].copy()
        broken["source_pickle"] = str(p.relative_to(PROJECT_ROOT))
        broken["source_idx"] = broken.index.astype(int)
        broken["fail_kind"] = ""
        broken.loc[rr_missing[broken_mask], "fail_kind"] = "recv_fail"
        broken.loc[sev_missing[broken_mask] & ~rr_missing[broken_mask], "fail_kind"] = "parse_fail"
        rows.append(broken)
    if not rows:
        return pd.DataFrame(columns=INPUT_COLS + ["source_pickle", "source_idx", "fail_kind"])
    out = pd.concat(rows, ignore_index=True)
    print(f"[{judge_label}] {len(out)} broken rows across {pickles_with_failures} pickles")
    return out


def main():
    for label, jdir in JUDGES.items():
        df = collect_failures(label, jdir)
        out_name = "deepseek_failures.pickle" if label == "DS" else "qwen_failures.pickle"
        out_path = OUT_DIR / out_name
        df.to_pickle(out_path)
        # Per-source-pickle breakdown
        if len(df):
            by_src = df.groupby("source_pickle").size().sort_values(ascending=False)
            print(f"  saved -> {out_path.relative_to(PROJECT_ROOT)}")
            print(f"  fail_kind: {dict(df['fail_kind'].value_counts())}")
            print(f"  top 5 source pickles:")
            for src, n in by_src.head(5).items():
                print(f"    {n:4d}  {src}")
        else:
            print(f"  (nothing to rerun for {label})")
        print()


if __name__ == "__main__":
    main()
