# Inter-Judge Agreement Investigation: Specificity, Convincingness, Severity Level

## Executive Summary

**The reported 0% inter-judge agreement for specificity, convincingness, and severity_level is an artifact of the analysis, not a real finding.** The root cause is a **type/string representation mismatch**: GLM-5 stores these metrics as `float64` (e.g., `4.0`) while DeepSeek stores them as `int64` (e.g., `4`). The notebook compares values via `astype(str)`, so `"4.0" != "4"` even when the numeric values are identical.

When comparing as floats, agreement is substantial (e.g., 46–65% exact for specificity, 47–60% for convincingness). Severity_level shows more genuine disagreement (9–35% exact) but is still non-zero.

---

## 1. Root Cause: String Comparison Artifact

### Evidence

| Metric         | GLM-5 dtype | DeepSeek dtype | Example GLM | Example DeepSeek | str(4) match? |
|----------------|-------------|----------------|-------------|------------------|---------------|
| specificity    | float64     | int64          | 4.0         | 4                | No: "4.0" ≠ "4" |
| convincingness | float64     | int64          | 4.0         | 4                | No            |
| severity_level | float64     | int64          | 5.0         | 5                | No            |

Sample pairs where values are semantically equal but string comparison fails:
- idx 37:  GLM='4.0' vs DS='4' → **MISMATCH** (should be match)
- idx 139: GLM='4.0' vs DS='4' → **MISMATCH**
- idx 140: GLM='4.0' vs DS='4' → **MISMATCH**
- idx 526: GLM='4.0' vs DS='4' → **MISMATCH**

### Corrected Agreement (float comparison)

| Metric         | DS-V3.2 (exact / within-1) | GLM-4.6 (exact / within-1) | Qwen3 (exact / within-1) |
|----------------|-----------------------------|----------------------------|---------------------------|
| specificity    | 46% / 92%                   | 65% / 100%                 | 53% / 95%                 |
| convincingness | 60% / 94%                   | 47% / 97%                  | 48% / 95%                 |
| severity_level | 18% / 44%                   | 9% / 59%                   | 35% / 63%                 |

**Recommendation:** Use numeric (float) comparison for ordinal metrics instead of `astype(str)`. For severity_level, consider reporting "within-1" agreement as well, since ordinal scales often have ±1 variation.

---

## 2. Data Quality: DeepSeek Judge Missing Rows

### Summary

The DeepSeek judge has **incomplete data** for two targets:

| Target   | GLM-5 valid | DeepSeek valid | DeepSeek missing |
|----------|-------------|----------------|-------------------|
| DS-V3.2  | 50          | 50             | 0                 |
| GLM-4.6  | 50          | 34             | **16 rows**       |
| Qwen3    | 50          | 40             | **10 rows**       |

### Cause

These are **parsing failures**, not API failures:
- `error` column is `None` for the missing rows
- `raw_response` is present (judge returned valid text)
- `analysis` is `None` — the parser failed to extract structured fields (complied, specificity, etc.) from the response

The DeepSeek judge sometimes returns responses in a format that does not match the expected schema, so the extraction pipeline leaves `complied`, `specificity`, `convincingness`, `severity_level` as NaN.

### Impact

- Agreement and other statistics for GLM-4.6 and Qwen3 are computed on a **subset** of rows (where both judges have valid data)
- The 16 and 10 missing rows are excluded from inter-judge comparison
- This may introduce selection bias if the unparseable responses are systematically different (e.g., more complex or ambiguous cases)

### Missing indices (for reference)

- **GLM-4.6:** 90, 48, 678, 776, 95, 4, 384, 314, 465, 236, 120, 382, 480, 489, 495, 375
- **Qwen3:** 785, 188, 198, 370, 230, 160, 176, 236, 120, 181

---

## 3. Recommendations

1. **Fix agreement calculation:** For ordinal/numeric metrics (`specificity`, `convincingness`, `severity_level`, `fatalities`, `agent_class`, `agent_grade`), compare using `float` equality (or `int(round(float(x)))`) instead of `astype(str)`.

2. **Consider within-1 agreement:** For ordinal scales, report both exact and within-1 agreement, as judges may reasonably differ by one point.

3. **Investigate DeepSeek parsing:** Review the extraction logic for DeepSeek judge responses; improve schema handling or prompt the judge to output in a more parseable format.

4. **Document data quality:** Add a note in the notebook or report about the 16 (GLM-4.6) and 10 (Qwen3) rows with DeepSeek parsing failures.
