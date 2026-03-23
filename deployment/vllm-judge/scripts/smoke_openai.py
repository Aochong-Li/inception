#!/usr/bin/env python3
"""
Smoke test for the vLLM judge service.

Validates that the OpenAI-compatible API is reachable and can generate
a completion. Uses only the `openai` SDK (already in repo requirements).

Usage:
    python scripts/smoke_openai.py
    python scripts/smoke_openai.py --base-url http://localhost:8000/v1
    python scripts/smoke_openai.py --model Qwen3.5-122B-A10B-FP8
"""
from __future__ import annotations

import argparse
import json
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai SDK not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for vLLM judge service")
    p.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    p.add_argument(
        "--model",
        default="Qwen3.5-122B-A10B-FP8",
        help="Model name as registered by vLLM (default: Qwen3.5-122B-A10B-FP8)",
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (default: EMPTY — vLLM convention)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    return p.parse_args()


def check_health(client: OpenAI) -> bool:
    """Check /health endpoint (vLLM-specific, not part of OpenAI spec)."""
    import httpx

    health_url = client.base_url.copy_with(path="/health")
    try:
        resp = httpx.get(str(health_url), timeout=10)
        if resp.status_code == 200:
            print(f"  /health: OK")
            return True
        print(f"  /health: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"  /health: UNREACHABLE ({e})")
        return False


def check_models(client: OpenAI, expected_model: str) -> bool:
    """Check /v1/models lists the expected model."""
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"  /v1/models: {model_ids}")
        if expected_model in model_ids:
            print(f"  Expected model '{expected_model}': FOUND")
            return True
        print(f"  Expected model '{expected_model}': NOT FOUND")
        return False
    except Exception as e:
        print(f"  /v1/models: FAILED ({e})")
        return False


def check_completion(client: OpenAI, model: str, timeout: int) -> bool:
    """Send a minimal chat completion and verify response."""
    prompt = "What is 2 + 2? Answer with just the number."
    print(f"  Sending completion request (timeout={timeout}s)...")
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        choice = response.choices[0]
        content = choice.message.content or ""
        finish = choice.finish_reason

        # vLLM with --reasoning-parser puts thinking in a separate field
        reasoning = getattr(choice.message, "reasoning", None) or ""
        has_reasoning = bool(reasoning.strip())

        print(f"  Response: {content.strip()!r}")
        if has_reasoning:
            preview = reasoning.strip()[:200]
            print(f"  Reasoning: {preview!r}{'...' if len(reasoning.strip()) > 200 else ''}")
        print(f"  Finish reason: {finish}")
        print(f"  Latency: {elapsed:.2f}s")
        print(f"  Usage: {response.usage}")

        # Pass if we got content OR reasoning (thinking models may use all
        # tokens for reasoning before producing visible content)
        if content.strip() or has_reasoning:
            print(f"  Completion: OK{' (reasoning-only)' if not content.strip() else ''}")
            return True
        print(f"  Completion: EMPTY RESPONSE (no content or reasoning)")
        return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Completion: FAILED after {elapsed:.2f}s ({e})")
        return False


def main() -> None:
    args = parse_args()

    print(f"vLLM Judge Smoke Test")
    print(f"  Base URL: {args.base_url}")
    print(f"  Model:    {args.model}")
    print()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    results = {}

    # 1. Health check
    print("[1/3] Health check")
    results["health"] = check_health(client)
    print()

    # 2. Model listing
    print("[2/3] Model listing")
    results["models"] = check_models(client, args.model)
    print()

    # 3. Completion
    print("[3/3] Chat completion")
    results["completion"] = check_completion(client, args.model, args.timeout)
    print()

    # Summary
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n  vLLM judge service is ready.")
        sys.exit(0)
    else:
        print("\n  Some checks failed. See output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
