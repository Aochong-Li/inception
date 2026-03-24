#!/usr/bin/env python3
"""
DeepSeek V3.2 judge token counts and cost analysis for eval_deepseek_judge outputs.

**Offline only:** reads existing pickles under `eval_deepseek_judge/` (and optionally
`inception_data/` for iteration stats). Uses `tiktoken` locally. **Does not call any
LLM API and does not rerun the safety judge evaluation.**

- Reconstructs judge prompts (same as SafetyEvaluator) and counts input/output tokens.
- Billed tokens: full text as sent/received (cl100k_base approximation).
- Content-only tokens: after stripping thinking/XML-style tags (for reporting).
- Optional: architect vs target tokens per iteration from inception_data pickles.

Outputs under evaluation/eval_deepseek_judge/analysis/token_cost/:
  - judge_tokens_by_run.csv
  - judge_tokens_aggregate.json
  - inception_iteration_tokens.csv (if inception data available)
  - COST_REPORT.md

Usage:
  uv run python evaluation/scripts/token_cost_analysis.py
  uv run python evaluation/scripts/token_cost_analysis.py --eval-root evaluation/eval_deepseek_judge
  uv run python evaluation/scripts/token_cost_analysis.py --no-inception
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.resolve()
_EVAL_DIR = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.prompts.loader import PromptLoader  # noqa: E402

# Official DeepSeek V3.2 (deepseek-chat) pricing, USD per 1M tokens (cache miss)
PRICE_INPUT_PER_1M = 0.28
PRICE_OUTPUT_PER_1M = 0.42

ENCODING_NAME = "cl100k_base"


# ---------------------------------------------------------------------------
# Tag stripping (content-only token stats)
# ---------------------------------------------------------------------------

# HTML-style thinking wrappers (encoding_dsv32.py; openaiapi uses `</think>` between CoT and content)
_THINK_START = b"\x3cthink\x3e".decode("ascii")
_THINK_END = b"\x3c/think\x3e".decode("ascii")
_THINK_BLOCK = re.compile(
    re.escape(_THINK_START) + r"[\s\S]*?" + re.escape(_THINK_END),
    re.DOTALL,
)
_THINK_TOKENS = re.compile("|".join(map(re.escape, (_THINK_START, _THINK_END))))
_XMLISH = re.compile(
    r"</?(?:reasoning|analysis|reasoning_trace|think|thought|user_query|scenario|"
    r"classification|example|input|output_format|rubric|tips|system)\b[^>]*>",
    re.IGNORECASE,
)


def strip_special_tags(text: str) -> str:
    """Remove common thinking / XML-style tags for content-only token counts."""
    if not text or not isinstance(text, str):
        return ""
    s = _THINK_BLOCK.sub("", text)
    s = _THINK_TOKENS.sub("", s)
    s = _XMLISH.sub("", s)
    return s


def unwrap_response(val: Any) -> Optional[str]:
    """Match SafetyEvaluator._unwrap_response."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        if len(val) == 0:
            return None
        val = val[0]
    if isinstance(val, float) and pd.isna(val):
        return None
    return str(val) if val is not None else None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class ModelRun:
    """One evaluated model directory under eval_deepseek_judge."""

    branch: str  # max_iterations_5 | ablation | simple_inject
    label: str  # e.g. max_iterations_5/think, ablation/think/tokens_512
    model_name: str
    model_dir: Path


def discover_model_runs(eval_root: Path) -> list[ModelRun]:
    runs: list[ModelRun] = []
    for branch, subpaths in (
        ("max_iterations_5", [("think",), ("instruct",)]),
        ("simple_inject", [("think",), ("instruct",)]),
    ):
        for (subtype,) in subpaths:
            base = eval_root / branch / subtype
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                if d.is_dir():
                    runs.append(
                        ModelRun(branch, f"{branch}/{subtype}", d.name, d)
                    )

    abl = eval_root / "ablation" / "think"
    if abl.is_dir():
        for tok_dir in sorted(abl.iterdir()):
            if not tok_dir.is_dir() or not tok_dir.name.startswith("tokens_"):
                continue
            for d in sorted(tok_dir.iterdir()):
                if d.is_dir():
                    runs.append(
                        ModelRun(
                            "ablation",
                            f"ablation/think/{tok_dir.name}",
                            d.name,
                            d,
                        )
                    )
    return runs


def inception_source_pickle(run: ModelRun, inception_data: Path) -> Optional[Path]:
    """Map eval output dir back to inception_data source pickle."""
    parts = run.label.split("/")
    if run.branch == "max_iterations_5":
        # max_iterations_5/think or instruct
        sub = parts[1]
        p = inception_data / "max_iterations_5" / sub / f"{run.model_name}.pickle"
    elif run.branch == "simple_inject":
        sub = parts[1]
        p = inception_data / "simple_inject" / sub / f"{run.model_name}.pickle"
    elif run.branch == "ablation":
        # ablation/think/tokens_512
        tok = parts[2]  # tokens_512
        n = tok.replace("tokens_", "")
        p = (
            inception_data
            / "max_iterations_1"
            / "think"
            / f"architect_initial_max_tokens_{n}"
            / f"{run.model_name}.pickle"
        )
    else:
        return None
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Prompt rendering (match SafetyEvaluator)
# ---------------------------------------------------------------------------


def render_judge_prompt(
    loader: PromptLoader,
    category: str,
    reasoning_trace: Any,
    user_query: str,
) -> str:
    if category == "bioterrorism":
        return loader.render_biosecurity_prompt(
            reasoning_trace=reasoning_trace or "",
            user_query=user_query or "",
        )
    if category == "chemical":
        return loader.render_chemical_prompt(
            reasoning_trace=reasoning_trace or "",
            user_query=user_query or "",
        )
    raise ValueError(f"Unknown category: {category}")


def effective_reasoning_trace(row: pd.Series) -> str:
    """Reasoning text fed to the judge (simple_inject may use `response`)."""
    rt = row.get("reasoning_traces")
    if rt is not None and not (isinstance(rt, float) and pd.isna(rt)):
        s = str(rt).strip()
        if s:
            return str(rt)
    resp = row.get("response")
    if resp is not None and not (isinstance(resp, float) and pd.isna(resp)):
        s = str(resp).strip()
        if s:
            return str(resp)
    return ""


# ---------------------------------------------------------------------------
# Architect / target iteration token stats
# ---------------------------------------------------------------------------


def row_iteration_text(row: pd.Series, i: int, role: str) -> str:
    if role == "architect":
        for c in (f"architect_iteration_{i}", f"architect_{i}"):
            if c in row.index:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    return str(v)
    else:
        for c in (f"target_iteration_{i}", f"target_{i}"):
            if c in row.index:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    return str(v)
    return ""


def analyze_inception_iterations(
    df: pd.DataFrame, enc: tiktoken.Encoding, run: ModelRun
) -> list[dict[str, Any]]:
    """Per-row iteration token counts (content-stripped)."""
    rows_out: list[dict[str, Any]] = []
    has_arch = any(
        f"architect_iteration_{i}" in df.columns or f"architect_{i}" in df.columns
        for i in range(5)
    )
    if not has_arch:
        # simple_inject: single response as target-only
        if "response" not in df.columns:
            return rows_out
        for idx, row in df.iterrows():
            raw = row.get("response")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            text = str(raw)
            stripped = strip_special_tags(text)
            rows_out.append(
                {
                    "run_label": run.label,
                    "model_name": run.model_name,
                    "branch": run.branch,
                    "df_index": idx,
                    "category": row.get("category", ""),
                    "iteration": 0,
                    "role": "target_simple_inject",
                    "tokens_full": len(enc.encode(text)),
                    "tokens_content": len(enc.encode(stripped)),
                }
            )
        return rows_out

    for idx, row in df.iterrows():
        cat = row.get("category", "")
        for i in range(5):
            for role in ("architect", "target"):
                text = row_iteration_text(row, i, role)
                if not text:
                    continue
                stripped = strip_special_tags(text)
                rows_out.append(
                    {
                        "run_label": run.label,
                        "model_name": run.model_name,
                        "branch": run.branch,
                        "df_index": idx,
                        "category": cat,
                        "iteration": i,
                        "role": role,
                        "tokens_full": len(enc.encode(text)),
                        "tokens_content": len(enc.encode(stripped)),
                    }
                )
    return rows_out


# ---------------------------------------------------------------------------
# Main judge token pass
# ---------------------------------------------------------------------------


@dataclass
class JudgeRowAgg:
    input_tokens_full: int = 0
    input_tokens_content: int = 0
    output_tokens_full: int = 0
    output_tokens_content: int = 0
    n_calls: int = 0


def process_run(
    run: ModelRun,
    loader: PromptLoader,
    enc: tiktoken.Encoding,
    *,
    collect_records: bool = False,
) -> tuple[list[dict[str, Any]], JudgeRowAgg]:
    """Return per-call records (if collect_records) and aggregate for one model directory."""
    pre = run.model_dir / f"{run.model_name}_preprocessed.pickle"
    if not pre.exists():
        return [], JudgeRowAgg()

    pdf = pd.read_pickle(pre)
    records: list[dict[str, Any]] = []
    agg = JudgeRowAgg()

    for category, fname, api_cat in (
        ("bioterrorism", "bioterrorism_safety_judge_raw.pickle", "bioterrorism"),
        ("chemical", "chemical_safety_judge_raw.pickle", "chemical"),
    ):
        raw_path = run.model_dir / fname
        if not raw_path.exists():
            continue
        raw = pd.read_pickle(raw_path)

        for idx, rraw in raw.iterrows():
            err = rraw.get("error")
            if err is not None and not (isinstance(err, float) and pd.isna(err)):
                continue
            resp_s = unwrap_response(rraw.get("response"))
            if not resp_s:
                continue

            lookup = (
                rraw["idx"]
                if "idx" in rraw.index and pd.notna(rraw.get("idx"))
                else idx
            )
            try:
                row_data = pdf.loc[lookup]
            except (KeyError, TypeError, ValueError):
                try:
                    row_data = pdf.iloc[int(lookup)]
                except (ValueError, IndexError, KeyError, TypeError):
                    continue

            if isinstance(row_data, pd.DataFrame):
                row_data = row_data.iloc[0]

            trace = effective_reasoning_trace(row_data)
            inquiry = row_data.get("inquiry", "") or ""
            if isinstance(inquiry, float) and pd.isna(inquiry):
                inquiry = ""

            prompt = render_judge_prompt(
                loader, api_cat, trace, str(inquiry) if inquiry else ""
            )
            prompt_stripped = strip_special_tags(prompt)
            out_stripped = strip_special_tags(resp_s)

            in_full = len(enc.encode(prompt))
            in_content = len(enc.encode(prompt_stripped))
            out_full = len(enc.encode(resp_s))
            out_content = len(enc.encode(out_stripped))

            if collect_records:
                records.append(
                    {
                        "run_label": run.label,
                        "model_name": run.model_name,
                        "branch": run.branch,
                        "judge_category": category,
                        "row_idx": lookup,
                        "input_tokens_full": in_full,
                        "input_tokens_content": in_content,
                        "output_tokens_full": out_full,
                        "output_tokens_content": out_content,
                    }
                )
            agg.input_tokens_full += in_full
            agg.input_tokens_content += in_content
            agg.output_tokens_full += out_full
            agg.output_tokens_content += out_content
            agg.n_calls += 1

    return records, agg


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_INPUT_PER_1M + (
        output_tokens / 1e6
    ) * PRICE_OUTPUT_PER_1M


def write_report(
    out_dir: Path,
    per_call_df: pd.DataFrame,
    by_run_df: pd.DataFrame,
    inception_df: Optional[pd.DataFrame],
    grand: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeepSeek V3.2 Judge — Token & Cost Report",
        "",
        "## Methodology",
        "",
        "- **Data source**: Existing artifacts only (`*_preprocessed.pickle`, `*_safety_judge_raw.pickle`). No API calls; no re-evaluation.",
        "- **Tokenizer**: `tiktoken` encoding `cl100k_base` (approximation; DeepSeek uses a different tokenizer; expect small systematic bias).",
        "- **Billed tokens** (`*_full`): full prompt and response text as reconstructed from pickles (matches what is sent/received).",
        "- **Content tokens** (`*_content`): same text after stripping `...` blocks, `<reasoning>`, `<analysis>`, and similar XML-style tags.",
        "- **Pricing** (DeepSeek API, `deepseek-chat` / V3.2, cache miss): input **$0.28 / 1M**, output **$0.42 / 1M** ([pricing](https://api-docs.deepseek.com/quick_start/pricing/)).",
        "- **Cost in this report** uses **full** (billed) token counts.",
        "",
        "## Grand totals",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Stored judge completions counted | {grand['n_calls']:,} |",
        f"| Input tokens (full) | {grand['input_tokens_full']:,} |",
        f"| Output tokens (full) | {grand['output_tokens_full']:,} |",
        f"| Input tokens (content-stripped) | {grand['input_tokens_content']:,} |",
        f"| Output tokens (content-stripped) | {grand['output_tokens_content']:,} |",
        f"| **Estimated cost (USD, full tokens)** | **${grand['cost_usd_full']:.2f}** |",
        f"| Estimated cost (USD, content-only) | ${grand['cost_usd_content']:.2f} |",
        "",
        "## By branch",
        "",
    ]

    if not by_run_df.empty and "branch" in by_run_df.columns:
        br = (
            by_run_df.groupby("branch")
            .agg(
                n_calls=("n_calls", "sum"),
                input_full=("input_tokens_full", "sum"),
                output_full=("output_tokens_full", "sum"),
            )
            .reset_index()
        )
        br["cost_usd"] = br.apply(
            lambda r: cost_usd(int(r["input_full"]), int(r["output_full"])), axis=1
        )
        lines.append("| Branch | Calls | Input (full) | Output (full) | Cost USD |")
        lines.append("|--------|-------|-------------|---------------|----------|")
        for _, r in br.iterrows():
            lines.append(
                f"| {r['branch']} | {int(r['n_calls']):,} | {int(r['input_full']):,} | "
                f"{int(r['output_full']):,} | ${r['cost_usd']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `judge_tokens_by_run.csv` — one row per (branch label, model).",
            "- `judge_tokens_aggregate.json` — machine-readable grand totals + by-branch.",
            "- `judge_per_call_tokens.csv` — optional large file; one row per stored completion (`--per-call`).",
            "",
        ]
    )

    if inception_df is not None and not inception_df.empty:
        lines.append("## Inception iteration tokens (content-stripped)")
        lines.append("")
        lines.append(
            "See `inception_iteration_tokens.csv` for architect/target tokens per iteration."
        )
        lines.append("")

    (out_dir / "COST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=_EVAL_DIR / "eval_deepseek_judge",
        help="Root of eval_deepseek_judge output",
    )
    parser.add_argument(
        "--inception-data",
        type=Path,
        default=_EVAL_DIR / "inception_data",
        help="Path to inception_data (for iteration-level stats)",
    )
    parser.add_argument(
        "--no-inception",
        action="store_true",
        help="Skip loading inception_data for iteration stats",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: eval_root/analysis/token_cost)",
    )
    parser.add_argument(
        "--per-call",
        action="store_true",
        help="Write judge_per_call_tokens.csv (large; default: skip for speed)",
    )
    args = parser.parse_args()

    eval_root = args.eval_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (eval_root / "analysis" / "token_cost")
    )

    enc = tiktoken.get_encoding(ENCODING_NAME)
    loader = PromptLoader()

    runs = discover_model_runs(eval_root)
    if not runs:
        print(f"No model runs found under {eval_root}", file=sys.stderr)
        sys.exit(1)

    all_records: list[dict[str, Any]] = []
    run_aggs: list[dict[str, Any]] = []

    for run in tqdm(runs, desc="Model runs"):
        recs, agg = process_run(run, loader, enc, collect_records=args.per_call)
        if args.per_call:
            all_records.extend(recs)
        if agg.n_calls:
            run_aggs.append(
                {
                    "branch": run.branch,
                    "run_label": run.label,
                    "model_name": run.model_name,
                    "n_calls": agg.n_calls,
                    "input_tokens_full": agg.input_tokens_full,
                    "input_tokens_content": agg.input_tokens_content,
                    "output_tokens_full": agg.output_tokens_full,
                    "output_tokens_content": agg.output_tokens_content,
                    "cost_usd_full": cost_usd(
                        agg.input_tokens_full, agg.output_tokens_full
                    ),
                }
            )

    by_run_df = pd.DataFrame(run_aggs)

    if args.per_call:
        per_call_df = pd.DataFrame(all_records)
    else:
        per_call_df = pd.DataFrame()

    if not by_run_df.empty:
        grand = {
            "n_calls": int(by_run_df["n_calls"].sum()),
            "input_tokens_full": int(by_run_df["input_tokens_full"].sum()),
            "output_tokens_full": int(by_run_df["output_tokens_full"].sum()),
            "input_tokens_content": int(by_run_df["input_tokens_content"].sum()),
            "output_tokens_content": int(by_run_df["output_tokens_content"].sum()),
        }
    else:
        grand = {
            "n_calls": 0,
            "input_tokens_full": 0,
            "output_tokens_full": 0,
            "input_tokens_content": 0,
            "output_tokens_content": 0,
        }
    grand["cost_usd_full"] = cost_usd(
        grand["input_tokens_full"], grand["output_tokens_full"]
    )
    grand["cost_usd_content"] = cost_usd(
        grand["input_tokens_content"], grand["output_tokens_content"]
    )

    inception_rows: list[dict[str, Any]] = []
    if not args.no_inception and args.inception_data.is_dir():
        for run in runs:
            src = inception_source_pickle(run, args.inception_data)
            if src is None:
                continue
            try:
                idf = pd.read_pickle(src)
            except Exception as e:
                print(f"Warning: could not load {src}: {e}", file=sys.stderr)
                continue
            inception_rows.extend(analyze_inception_iterations(idf, enc, run))

    inception_df = pd.DataFrame(inception_rows) if inception_rows else None

    out_dir.mkdir(parents=True, exist_ok=True)
    by_run_df.to_csv(out_dir / "judge_tokens_by_run.csv", index=False)
    if args.per_call and not per_call_df.empty:
        per_call_df.to_csv(out_dir / "judge_per_call_tokens.csv", index=False)

    by_branch = (
        by_run_df.groupby("branch")
        .agg(
            n_calls=("n_calls", "sum"),
            input_tokens_full=("input_tokens_full", "sum"),
            output_tokens_full=("output_tokens_full", "sum"),
        )
        .reset_index()
    )
    by_branch["cost_usd_full"] = by_branch.apply(
        lambda r: cost_usd(int(r["input_tokens_full"]), int(r["output_tokens_full"])),
        axis=1,
    )

    aggregate = {
        "encoding": ENCODING_NAME,
        "pricing_usd_per_1m": {
            "input_cache_miss": PRICE_INPUT_PER_1M,
            "output": PRICE_OUTPUT_PER_1M,
        },
        "grand_totals": grand,
        "by_branch": by_branch.to_dict(orient="records"),
    }
    (out_dir / "judge_tokens_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    if inception_df is not None and not inception_df.empty:
        inception_df.to_csv(out_dir / "inception_iteration_tokens.csv", index=False)
        summary = (
            inception_df.groupby(["branch", "role", "iteration"])
            .agg(
                mean_content=("tokens_content", "mean"),
                mean_full=("tokens_full", "mean"),
                n=("tokens_content", "count"),
            )
            .reset_index()
        )
        summary.to_csv(
            out_dir / "inception_iteration_summary.csv", index=False
        )

    write_report(out_dir, per_call_df, by_run_df, inception_df, grand)
    print(f"Wrote analysis to {out_dir}")
    print(
        f"Grand total: {grand['n_calls']} calls, "
        f"cost ${grand['cost_usd_full']:.2f} (full tokens)"
    )


if __name__ == "__main__":
    main()
