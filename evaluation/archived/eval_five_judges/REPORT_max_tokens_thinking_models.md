# Report: max_tokens for Thinking Models in Safety Evaluation Judge Pipelines

**Date:** March 4, 2025  
**Scope:** Evaluation of 16,000 vs 3,072 tokens as default `max_tokens` for thinking models (GLM-5, Kimi K2, Claude extended thinking, Gemini thinking) in safety judge pipelines.

---

## Executive Summary

**Recommendation: Use 16,000 tokens as the default for thinking models in safety evaluation judge pipelines.**

Evidence from provider documentation, truncation analysis, and best practices supports 16,000 over 3,072. The 3,072 limit causes frequent truncation when judges output long analysis plus structured JSON, leading to parse failures and lost evaluations.

---

## 1. max_tokens / max_completion_tokens Limits by Model

### GLM-5 (Z.ai)

| Parameter | Limit |
|-----------|-------|
| Max output tokens | Up to **128K** |
| Context window | 200K |
| Thinking | Enabled by default; unified hybrid reasoning architecture |

- Thinking modes: Interleaved (default), Turn-level, Preserved
- No separate thinking budget parameter; thinking shares output token budget
- **Implication:** `max_tokens` must accommodate both reasoning and final response

### Kimi K2 / K2.5 (Moonshot AI)

| Parameter | Limit |
|-----------|-------|
| Context window | 131K (K2 Thinking) / 256K (256K version) |
| API parameter | `max_tokens` (not `max_completion_tokens`) |
| Known issues | Truncation with `stop_reason: length` even when `max_output_tokens=16384` |

- **GitHub Issue #83:** Users report truncated replies with `stop_reason: length` despite requesting 16,384 output tokens; response ~4,304 chars
- Some frameworks cap output at **1,024 tokens** regardless of specified value (LangGraph compatibility)
- Moonshot recommends: estimate input tokens → subtract from model max → use result as `max_tokens`

### Claude Extended Thinking (Anthropic)

| Parameter | Limit |
|-----------|-------|
| `budget_tokens` (thinking) | Min 1,024; max 128,000 |
| `max_tokens` (total output) | Must be **strictly greater** than reasoning budget |
| Opus 4.6 | Up to 128K output tokens |
| Earlier models | Up to 64K output tokens |
| Streaming | Required when `max_tokens` > 21,333 |

**Recommended budget_tokens by complexity:**
- Simple: 3,000
- Medium: 5,000
- High: 10,000
- Maximum: 20,000

**Official Claude docs use `max_tokens=16000` with `budget_tokens=10000`** in all extended-thinking examples.

### Gemini Thinking (Google)

| Model | thinking_budget range | Default |
|-------|----------------------|---------|
| Gemini 2.5 Pro | 128 – 32,768 | 8,192 |
| Gemini 2.5 Flash | 0 – 24,576 | dynamic (-1) |
| Gemini 2.5 Flash-Lite | 512 – 24,576 | disabled (0) |
| Gemini 3 | `thinking_level`: LOW / HIGH (no token budget) | — |

**Recommended:** 8,000–16,000 tokens for complex reasoning; 1,024–4,096 for simpler tasks.

---

## 2. OpenRouter and Provider Documentation

### OpenRouter Reasoning Tokens

- **`reasoning.max_tokens`** (Anthropic-style): Supported by Gemini thinking, Anthropic reasoning, some Alibaba Qwen models
- **`reasoning.effort`** (OpenAI-style): Supported by OpenAI o-series, o3, Grok
- Reasoning tokens count as **output tokens** for billing

**Anthropic models via OpenRouter:**
- `budget_tokens = max(min(max_tokens * effort_ratio, 128000), 1024)`
- **Important:** `max_tokens` must be strictly higher than the reasoning budget so tokens remain for the final response

**Example from OpenRouter docs:**
```json
{
  "reasoning": { "max_tokens": 8000 },
  "max_tokens": 10000
}
```

### General max_tokens

- OpenRouter: `max_tokens` in range `[1, context_length)` per model
- No single recommended value; depends on model and use case

---

## 3. Truncation Issues with 3,072 Tokens

### Why 3,072 Is Problematic

1. **Output structure:** Safety judges produce:
   - `<analysis>...</analysis>` (long prose)
   - JSON block with schema fields (e.g., `complied`, `specificity`, `convincingness`, `severity_level`, etc.)

2. **Token budget:** At ~4 chars/token:
   - 3,072 tokens ≈ 12,288 characters
   - Truncation heuristic in `diagnose_parse_failures.py`: 85% of 3,072 × 4 ≈ 10,445 chars

3. **Observed failure modes** (from `diagnose_parse_failures.py`):
   - **truncation_token_limit:** `<analysis>` present, `</analysis>` and JSON missing → likely hit `max_tokens`
   - **truncation_no_json:** `</analysis>` present but no JSON → truncated after analysis
   - **truncation_likely:** Long response without JSON, above char threshold

4. **finish_reason: "length"** indicates truncation; responses end mid-sentence or mid-JSON.

### Evidence from Codebase

- `safety-judge.py`: `max_tokens: int = 16000  # Sufficient for long analysis + JSON; avoids truncation`
- `run_five_judges.py`: All judge models use 16,000 for uniformity and to avoid truncation
- `diagnose_parse_failures.py`: References 3,072 as prior default; recommends increasing to 6,144/8,192 for DeepSeek

---

## 4. Best Practices for Safety Evaluation Judge Pipelines

From MLflow, Langfuse, Pydantic, and MetaEvaluator:

1. **Request reasoning:** Ask judges to explain verdicts for debugging and rubric iteration.
2. **Use case-specific evaluators:** Tailor judges to biosecurity, chemical, etc.
3. **Combine deterministic and semantic checks:** Validate format/type first, then use LLM for nuanced assessment.
4. **Choose models with structured output:** Reliable JSON/parsing is critical.
5. **Validate judges:** Compare to human annotations before deployment.
6. **Account for bias:** Be aware of sycophancy and self-favoring.

**Implication for max_tokens:** Judges that produce reasoning plus structured output need enough headroom for both. Truncation breaks structured output and undermines evaluation reliability.

---

## 5. Evaluation: 16,000 vs 3,072

| Criterion | 3,072 | 16,000 |
|-----------|-------|--------|
| **Claude extended thinking** | Too low for budget_tokens + response | Matches official examples (16k max, 10k budget) |
| **Gemini complex reasoning** | Below recommended 8k–16k | Within recommended range |
| **GLM-5** | Risk of truncation | Adequate for reasoning + JSON |
| **Kimi K2** | High truncation risk | Reduces truncation (provider limits may still apply) |
| **Safety judge output** | Often truncates before JSON | Typically fits analysis + JSON |
| **Parse failure rate** | Higher (truncation categories) | Lower |
| **Cost** | Lower per request | Higher per request |
| **Reliability** | Poor (lost evaluations) | Better (complete responses) |

### Conclusion

**16,000 tokens is a better default** because:

1. **Provider alignment:** Claude and Gemini docs use or recommend 8k–16k for complex reasoning.
2. **Truncation reduction:** 3,072 frequently truncates before the JSON block; 16,000 usually allows full output.
3. **Thinking models:** Reasoning and final answer both consume output tokens; 16,000 leaves room for both.
4. **Uniformity:** One default across judges simplifies configuration and comparison.
5. **Cost vs reliability:** For safety evaluation, complete and parseable responses outweigh marginal cost savings.

### When to Use Lower Limits

- **StrongREJECT-only mode:** 512 tokens is sufficient for a 3-field JSON response.
- **Simple classification:** If the rubric is minimal and no long analysis is required, 4,096–6,144 may be acceptable after validation.

---

## References

- [OpenRouter Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Claude Extended Thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Kimi K2 Issue #83 – Truncation](https://github.com/MoonshotAI/Kimi-K2/issues/83)
- [Gemini Thinking Mode](https://ai.google.dev/gemini-api/docs/thinking)
- [LLM-as-a-Judge Best Practices (Pydantic)](https://pydantic.dev/articles/llm-as-a-judge)
- Inception-eval: `safety-judge.py`, `run_five_judges.py`, `diagnose_parse_failures.py`
