#!/usr/bin/env python3
"""
Temporary diagnostic script: analyze DeepSeek judge parse failures and categorize
failure reasons (API errors, token limits, malformed JSON, etc.).

Useful for:
- Determining if any failed rows can be salvaged by re-parsing (no API cost)
- Informing fallback/retry logic for larger pipeline runs

Usage:
    python evaluation/scripts/diagnose_parse_failures.py
    # Reads from evaluation/archived/eval_five_judges/
"""
import json
import re
import codecs
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

# Paths relative to evaluation/
SCRIPT_DIR = Path(__file__).parent.resolve()
_eval_dir = SCRIPT_DIR.parent
BASE = _eval_dir / "archived" / "eval_five_judges"
JUDGE_DIR = "deepseek_deepseek-chat"
TARGETS = [
    "DeepSeek-V3.2",
    "GLM-4.6",
    "Qwen3-235B-A22B-Thinking-2507",
]
RAW_CATEGORIES = ["bioterrorism", "chemical"]

# Approximate chars per token for truncation heuristic (~4 for English)
CHARS_PER_TOKEN = 4
MAX_TOKENS_DEFAULT = 3072  # From safety-judge / run_five_judges_eval
TRUNCATION_CHAR_THRESHOLD = (MAX_TOKENS_DEFAULT * CHARS_PER_TOKEN) * 0.85  # ~10.4k chars


def unwrap_response(val):
    """Mirror safety-judge._unwrap_response."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        if len(val) == 0:
            return None
        val = val[0]
    if isinstance(val, float) and pd.isna(val):
        return None
    return str(val) if val is not None else None


def try_extract_json(text: str) -> dict | None:
    """
    Try to extract and parse a JSON object from text.
    Searches for first { to last } (handles nested braces loosely).
    Returns parsed dict or None.
    """
    if not text or "{" not in text or "}" not in text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    # Find matching closing brace (simple balance count)
    depth = 0
    end = -1
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end <= start:
        return None
    json_str = text[start:end]
    try:
        data = json.loads(json_str)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def try_extract_json_with_escape(text: str) -> dict | None:
    """Try json extraction, with unicode_escape fallback like safety-judge."""
    result = try_extract_json(text)
    if result is not None:
        return result
    # Try after removing <analysis>...</analysis> to avoid false matches
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL).strip()
    result = try_extract_json(cleaned)
    if result is not None:
        return result
    try:
        decoded = codecs.decode(cleaned, "unicode_escape")
        return try_extract_json(decoded)
    except (UnicodeDecodeError, ValueError):
        pass
    return None


@dataclass
class FailureRecord:
    idx: int
    target: str
    category: str
    reason: str
    detail: str = ""
    raw_len: int = 0
    has_analysis_open: bool = False
    has_analysis_close: bool = False
    has_json_brace: bool = False
    salvageable: bool = False
    salvage_attempted: bool = False
    salvage_success: bool = False


def categorize_failure(row, idx: int, target: str, category: str) -> FailureRecord:
    """
    Categorize why a row failed to parse. Returns a FailureRecord.
    """
    raw = row.get("raw_response") or row.get("response")
    raw_str = unwrap_response(raw)
    error_val = row.get("error")
    has_error = error_val is not None and (not (isinstance(error_val, float) and pd.isna(error_val)))

    # 1. API error
    if has_error:
        return FailureRecord(
            idx=idx,
            target=target,
            category=category,
            reason="api_error",
            detail=str(error_val)[:200],
            raw_len=len(raw_str) if raw_str else 0,
        )

    # 2. Empty / no response
    if raw_str is None or (isinstance(raw_str, str) and len(raw_str.strip()) == 0):
        return FailureRecord(
            idx=idx,
            target=target,
            category=category,
            reason="empty_response",
            detail="raw_response is null or empty",
            raw_len=0,
        )

    raw_str = str(raw_str).strip()
    raw_len = len(raw_str)
    has_open = "<analysis>" in raw_str
    has_close = "</analysis>" in raw_str
    has_brace = "{" in raw_str

    # 3. Truncation (token limit): has content but no JSON, often no </analysis>
    # Strong signal: has <analysis> but no </analysis> and no {
    if has_open and not has_close and not has_brace:
        return FailureRecord(
            idx=idx,
            target=target,
            category=category,
            reason="truncation_token_limit",
            detail="<analysis> present, </analysis> and JSON block missing; likely hit max_tokens",
            raw_len=raw_len,
            has_analysis_open=True,
            has_analysis_close=False,
            has_json_brace=False,
        )

    # 4. Truncation (no JSON): has </analysis> but no { (JSON never emitted)
    if has_close and not has_brace:
        return FailureRecord(
            idx=idx,
            target=target,
            category=category,
            reason="truncation_no_json",
            detail="</analysis> present but no JSON block; response may be truncated after analysis",
            raw_len=raw_len,
            has_analysis_open=True,
            has_analysis_close=True,
            has_json_brace=False,
        )

    # 5. Long response without JSON - likely truncation before JSON
    if not has_brace and raw_len >= TRUNCATION_CHAR_THRESHOLD:
        return FailureRecord(
            idx=idx,
            target=target,
            category=category,
            reason="truncation_likely",
            detail=f"Long response ({raw_len} chars) without JSON; likely hit token limit",
            raw_len=raw_len,
            has_analysis_open=has_open,
            has_analysis_close=has_close,
            has_json_brace=False,
        )

    # 6. JSON present - try to salvage
    if has_brace:
        parsed = try_extract_json_with_escape(raw_str)
        if parsed and any(k in parsed for k in ("complied", "specificity", "convincingness")):
            return FailureRecord(
                idx=idx,
                target=target,
                category=category,
                reason="json_salvageable",
                detail=f"Extracted keys: {list(parsed.keys())}",
                raw_len=raw_len,
                has_analysis_open=has_open,
                has_analysis_close=has_close,
                has_json_brace=True,
                salvageable=True,
                salvage_attempted=True,
                salvage_success=True,
            )
        else:
            return FailureRecord(
                idx=idx,
                target=target,
                category=category,
                reason="json_present_parse_failed",
                detail="Has { but extraction failed or missing required keys",
                raw_len=raw_len,
                has_analysis_open=has_open,
                has_analysis_close=has_close,
                has_json_brace=True,
                salvageable=False,
                salvage_attempted=True,
                salvage_success=False,
            )

    # 7. Other
    return FailureRecord(
        idx=idx,
        target=target,
        category=category,
        reason="other",
        detail="Unclear; no API error, has content, but no analysis/JSON structure",
        raw_len=raw_len,
        has_analysis_open=has_open,
        has_analysis_close=has_close,
        has_json_brace=has_brace,
    )


def load_raw_pickle(judge_dir: Path, target: str, category: str) -> pd.DataFrame | None:
    path = judge_dir / target / f"{category}_safety_judge_raw.pickle"
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def main():
    judge_path = BASE / JUDGE_DIR
    if not judge_path.exists():
        print(f"Judge directory not found: {judge_path}")
        print("Ensure evaluation/archived/eval_five_judges/ contains eval data.")
        return 1

    all_records: list[FailureRecord] = []
    parse_failed_mask = None  # Will be a function of columns

    for target in TARGETS:
        for cat in RAW_CATEGORIES:
            df = load_raw_pickle(judge_path, target, cat)
            if df is None or df.empty:
                continue

            # Identify parse-failed rows: raw_response exists but structured fields missing
            has_raw = df["raw_response"].notna() if "raw_response" in df.columns else df["response"].notna()
            if "raw_response" not in df.columns and "response" in df.columns:
                df = df.copy()
                df["raw_response"] = df["response"].apply(unwrap_response)
                has_raw = df["raw_response"].notna()

            missing_structured = pd.Series(True, index=df.index)
            if "complied" in df.columns:
                missing_structured = df["complied"].isna()
            elif "specificity" in df.columns:
                missing_structured = df["specificity"].isna()
            elif "analysis" in df.columns:
                missing_structured = df["analysis"].isna()

            # Also include API failures (error or no raw_response)
            has_error = df["error"].notna() if "error" in df.columns else pd.Series(False, index=df.index)
            no_raw = ~has_raw
            failed = (has_raw & missing_structured) | has_error | no_raw

            for idx in df.index[failed]:
                row = df.loc[idx]
                rec = categorize_failure(row, int(idx) if isinstance(idx, (int, float)) else idx, target, cat)
                all_records.append(rec)

    # Aggregate by reason
    by_reason = defaultdict(list)
    for r in all_records:
        by_reason[r.reason].append(r)

    # Print report
    print("=" * 70)
    print("DeepSeek Judge Parse Failure Diagnostic")
    print("=" * 70)
    print(f"Judge dir: {judge_path}")
    print(f"Total failed rows: {len(all_records)}")
    print()

    print("--- By failure reason ---")
    for reason in sorted(by_reason.keys(), key=lambda r: (-len(by_reason[r]), r)):
        recs = by_reason[reason]
        print(f"\n  {reason}: {len(recs)} rows")
        for r in recs[:3]:
            print(f"    - idx={r.idx} target={r.target} cat={r.category} len={r.raw_len}")
        if len(recs) > 3:
            print(f"    ... and {len(recs) - 3} more")
        if recs:
            print(f"    Detail: {recs[0].detail[:100]}")

    salvageable = [r for r in all_records if r.salvageable and r.salvage_success]
    truncation = [r for r in all_records if "truncation" in r.reason]
    api_errors = [r for r in all_records if r.reason == "api_error"]
    empty = [r for r in all_records if r.reason == "empty_response"]

    print()
    print("--- Summary for fallback/retry design ---")
    print(f"  Salvageable (re-parse, no API): {len(salvageable)}")
    print(f"  Truncation / token limit:       {len(truncation)}  <- increase max_tokens, re-eval")
    print(f"  API errors:                     {len(api_errors)}  <- retry with backoff")
    print(f"  Empty response:                 {len(empty)}")
    print()

    if truncation:
        print("--- Truncation indices (for targeted re-eval) ---")
        by_target = defaultdict(list)
        for r in truncation:
            by_target[r.target].append(r.idx)
        for t, idxs in sorted(by_target.items()):
            print(f"  {t}: {sorted(idxs)}")
        lens = [r.raw_len for r in truncation]
        print(f"  Response length range: {min(lens)}–{max(lens)} chars (~{min(lens)//4}–{max(lens)//4} tokens)")

    if salvageable:
        print()
        print("--- Salvageable rows (can fix without API) ---")
        for r in salvageable:
            print(f"  idx={r.idx} target={r.target} cat={r.category}")

    print()
    print("--- Fallback recommendations (for pipeline integration) ---")
    print("  truncation_token_limit / truncation_no_json / truncation_likely:")
    print("    -> Increase max_tokens for judge (e.g. 6144/8192 for DeepSeek)")
    print("    -> Extend _identify_failed_rows to flag parse-failed rows; re-eval those only")
    print("  api_error:")
    print("    -> Retry with exponential backoff; consider batch API for large runs")
    print("  empty_response:")
    print("    -> Retry once; if persistent, log and skip or flag for manual review")
    print("  json_present_parse_failed:")
    print("    -> Add lenient JSON parser (extract partial keys); or re-eval")
    print()

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
