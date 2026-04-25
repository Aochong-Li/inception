"""
openaiapi.py

A production-ready client wrapper for calling OpenAI and third-party LLM APIs.
Provides functions for single and parallel chat completions with support for configurable models,
rate limiting, and efficient client reuse.

Author: Aochong Oliver Li
Date: 2025-07-14
"""

from __future__ import annotations

import json, logging, math, os, random, time
from dotenv import load_dotenv

load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import (OpenAI, APIError, APIConnectionError, RateLimitError,
                    APITimeoutError)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# logging setup (inherits root config from caller if present)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Provider registry – add new providers in one place
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai":         {"env": "OPENAI_API_KEY",     "base_url": None},
    "deepseek":       {"env": "DEEPSEEK_API_KEY",   "base_url": "https://api.deepseek.com"},
    # DeepSeek requires the /beta base_url for the raw /v1/completions endpoint.
    # Use "deepseek_beta" for mode="completions" with DeepSeek models.
    "deepseek_beta":  {"env": "DEEPSEEK_API_KEY",   "base_url": "https://api.deepseek.com/beta"},
    "togetherai":     {"env": "TOGETHERAI_API_KEY", "base_url": "https://api.together.xyz/v1"},
    "openrouter":     {"env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
    "deepinfra":      {"env": "DEEPINFRA_API_KEY",  "base_url": "https://api.deepinfra.com/v1/openai"},
    "vllm_local":     {"env": "VLLM_API_KEY",       "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")},
    # Local architect server (vLLM/SGLang). Keyless by convention; use the
    # ARCHITECT_BASE_URL env var to point elsewhere (default localhost:8001).
    "local_architect": {"env": None,                 "base_url": os.environ.get("ARCHITECT_BASE_URL", "http://localhost:8001/v1")},
}

RETRYABLE = (RateLimitError, APIError, APIConnectionError, APITimeoutError)

# ---------------------------------------------------------------------------
# Thread-safe token bucket rate limiter
# ---------------------------------------------------------------------------
class TokenBucketRateLimiter:
    """Thread-safe token bucket for rate limiting API calls.

    Workers call ``acquire()`` before each API request.  On ``RateLimitError``
    they call ``throttle()`` to halve the effective rate; on success they call
    ``restore()`` to gradually recover toward the original rate.
    """

    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate                   # tokens per second (current effective)
        self._target_rate = rate           # original / ceiling rate for restore
        self.burst = burst                 # max tokens (burst capacity)
        self.tokens = float(burst)         # current token count
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._min_rate = 0.5              # floor to prevent complete stall

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time. Must be called with lock held."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, timeout: float = 30.0) -> bool:
        """Block until a token is available (or timeout expires).

        Returns True if a token was acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            # Sleep for at most the time needed to accumulate one token
            wait = min(1.0 / max(self.rate, self._min_rate), deadline - time.monotonic())
            if wait <= 0:
                return False
            time.sleep(wait)

    def throttle(self, factor: float = 0.5) -> None:
        """Reduce rate on 429 — called from retry logic."""
        with self._lock:
            self.rate = max(self._min_rate, self.rate * factor)
            logger.warning("Rate limiter throttled to %.2f req/s", self.rate)

    def restore(self) -> None:
        """Gradually recover rate after successful requests (10% step toward target)."""
        with self._lock:
            if self.rate < self._target_rate:
                self.rate = min(self._target_rate, self.rate * 1.1)


# ---------------------------------------------------------------------------
# Client factory – cached per‑process, per provider
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def create_client(client_name: str) -> OpenAI:
    cfg = PROVIDERS.get(client_name)
    if cfg is None:
        raise ValueError(f"Unknown provider '{client_name}'.")
    if cfg["env"] is None:
        # Keyless provider (e.g. local vLLM/SGLang server) — SDK still requires
        # an api_key field, so pass a placeholder.
        api_key = "EMPTY"
    else:
        api_key = os.getenv(cfg["env"])
        if not api_key:
            raise RuntimeError(f"Environment variable {cfg['env']} not set.")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    if client_name == "openai":
        kwargs.update(
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT_ID"),
        )
    return OpenAI(**kwargs)

# ---------------------------------------------------------------------------
# Wrapper for chat completions and completions
# ---------------------------------------------------------------------------
def generate_chat_completions(
    *,
    input_prompt: str,
    developer_message: str = "You are a helpful assistant",
    model: str = "gpt-4o",
    client_name: str = "openai",
    temperature: float = 0.6,
    max_tokens: int = 4096,
    n: int = 1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[List[str]] = None,
    max_attempts: int = 3,
    rate_limiter: Optional["TokenBucketRateLimiter"] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[str]], Optional[str], List[str], int]:
    """Returns (content, finish_reason, errors, attempt). finish_reason is 'length' when truncated.

    ``extra_body`` is forwarded as-is to the OpenAI SDK (e.g. provider-specific
    ``{"thinking": {"type": "enabled"}}`` or
    ``{"chat_template_kwargs": {"enable_thinking": true}}``).
    """
    client = create_client(client_name)
    messages = [
        {"role": "system", "content": developer_message},
        {"role": "user", "content": input_prompt},
    ]

    errors: List[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: Dict[str, Any] = dict(model=model, messages=messages, n=n)
            if model == "deepseek-reasoner":
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs.update(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            # Restore rate after a successful call
            if rate_limiter is not None:
                rate_limiter.restore()
            c0 = resp.choices[0]
            finish_reason = getattr(c0, "finish_reason", None) or getattr(c0, "stop_reason", None)
            if model == "deepseek-reasoner":
                content = [
                    f"{c.message.reasoning_content}\n</think>\n{c.message.content}" for c in resp.choices
                ]
            else:
                # Generalized: if the provider returned a reasoning_content
                # field (e.g. DeepSeek V4-Pro chat endpoint), synthesize the
                # <think>{reasoning}</think>{content} envelope so downstream
                # code that expects `</think>` as a delimiter keeps working.
                def _assemble(c):
                    rc = getattr(c.message, "reasoning_content", None)
                    ct = c.message.content or ""
                    if rc:
                        return f"<think>{rc}</think>{ct}"
                    return ct
                content = [_assemble(c) for c in resp.choices]
            return content, finish_reason, errors, attempt
        except RateLimitError as exc:
            errors.append(repr(exc))
            # Signal the shared limiter to back off
            if rate_limiter is not None:
                rate_limiter.throttle(0.5)
            if attempt == max_attempts:
                logger.error("%s – final failure", exc)
                return None, None, errors, attempt
            # header‑aware back‑off
            hdr_delay = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        hdr_delay = float(retry_after)
                    except ValueError:
                        hdr_delay = None
            delay = hdr_delay if hdr_delay else min(30, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            logger.warning("%s – retry %d/%d in %.1fs", exc, attempt, max_attempts, delay)
            time.sleep(delay)
        except RETRYABLE as exc:
            errors.append(repr(exc))
            if attempt == max_attempts:
                logger.error("%s – final failure", exc)
                return None, None, errors, attempt
            # header‑aware back‑off
            hdr_delay = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        hdr_delay = float(retry_after)
                    except ValueError:
                        hdr_delay = None
            delay = hdr_delay if hdr_delay else min(30, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            logger.warning("%s – retry %d/%d in %.1fs", exc, attempt, max_attempts, delay)
            time.sleep(delay)
        except Exception as exc:
            errors.append(repr(exc))
            logger.error("Non‑retryable error: %s", exc)
            return None, None, errors, attempt

    return None, None, errors, max_attempts

def generate_completions(
    *,
    input_prompt: str,
    model: str = "gpt-4o",
    client_name: str = "openai",
    temperature: float = 0.6,
    max_tokens: int = 4096,
    n: int = 1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[List[str]] = None,
    max_attempts: int = 3,
    rate_limiter: Optional["TokenBucketRateLimiter"] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[str]], Optional[str], List[str], int]:
    """Returns (content, finish_reason, errors, attempt). finish_reason is 'length' when truncated.

    ``extra_body`` is forwarded as-is to the OpenAI SDK on the /v1/completions
    endpoint.
    """
    client = create_client(client_name)

    errors: List[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: Dict[str, Any] = dict(model=model, prompt=input_prompt, n=n)
            kwargs.update(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            resp = client.completions.create(**kwargs)
            # Restore rate after a successful call
            if rate_limiter is not None:
                rate_limiter.restore()
            c0 = resp.choices[0]
            finish_reason = getattr(c0, "finish_reason", None) or getattr(c0, "stop_reason", None)
            if model == "deepseek-reasoner":
                content = [
                    f"{c.message.reasoning_content}\n</think>\n{c.message.content}" for c in resp.choices
                ]
            else:
                content = [c.text for c in resp.choices]
            return content, finish_reason, errors, attempt
        except RateLimitError as exc:
            errors.append(repr(exc))
            if rate_limiter is not None:
                rate_limiter.throttle(0.5)
            if attempt == max_attempts:
                logger.error("%s – final failure", exc)
                return None, None, errors, attempt
            # header‑aware back‑off
            hdr_delay = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        hdr_delay = float(retry_after)
                    except ValueError:
                        hdr_delay = None
            delay = hdr_delay if hdr_delay else min(30, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            logger.warning("%s – retry %d/%d in %.1fs", exc, attempt, max_attempts, delay)
            time.sleep(delay)
        except RETRYABLE as exc:
            errors.append(repr(exc))
            if attempt == max_attempts:
                logger.error("%s – final failure", exc)
                return None, None, errors, attempt
            # header‑aware back‑off
            hdr_delay = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        hdr_delay = float(retry_after)
                    except ValueError:
                        hdr_delay = None
            delay = hdr_delay if hdr_delay else min(30, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            logger.warning("%s – retry %d/%d in %.1fs", exc, attempt, max_attempts, delay)
            time.sleep(delay)
        except Exception as exc:
            errors.append(repr(exc))
            logger.error("Non‑retryable error: %s", exc)
            return None, None, errors, attempt

    return None, None, errors, max_attempts
# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------
ResultRow = Tuple[int, Optional[List[str]], Optional[List[str]], int, Optional[str]]  # idx, response, error, retries, finish_reason

def _process(
    idx: int,
    req: Dict[str, Any],
    func_name: str,
    rate_limiter: Optional[TokenBucketRateLimiter] = None,
    max_api_attempts: int = 3,
) -> ResultRow:
    """
    Process a single request with API-level retries.

    Args:
        idx: Request index
        req: Request dictionary containing body and client_name
        func_name: "chat_completions" or "completions"
        rate_limiter: Optional shared TokenBucketRateLimiter; acquire() is called before each API call.
        max_api_attempts: Max API-level retries per call (default 3).

    Returns:
        Tuple of (idx, response, errors, retries, finish_reason)
    """
    body = req["body"]
    all_errors: List[str] = []
    effective_temperature = body.get("temperature", 0.0)

    # Acquire a token from the rate limiter before calling the API
    if rate_limiter is not None:
        acquired = rate_limiter.acquire()
        if not acquired:
            logger.warning(
                "Rate limiter acquire() timed out for idx %d, proceeding unthrottled",
                idx,
            )

    if func_name == "chat_completions":
        _msgs = body.get("messages") or []
        _dev = _msgs[0]["content"] if (_msgs and _msgs[0].get("role") in ("system", "developer")) else ""
        _user = _msgs[1]["content"] if len(_msgs) > 1 else (_msgs[0]["content"] if _msgs else "")
        response, finish_reason, errs, tries = generate_chat_completions(
            input_prompt=_user,
            developer_message=_dev,
            model=body["model"],
            client_name=req["client_name"],
            temperature=effective_temperature,
            max_tokens=body.get("max_tokens", 1024),
            n=body.get("n", 1),
            top_p=body.get("top_p", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
            max_attempts=max_api_attempts,
            rate_limiter=rate_limiter,
            extra_body=body.get("extra_body"),
        )
    elif func_name == "completions":
        response, finish_reason, errs, tries = generate_completions(
            input_prompt=body["prompt"],
            model=body["model"],
            client_name=req["client_name"],
            temperature=effective_temperature,
            max_tokens=body.get("max_tokens", 1024),
            n=body.get("n", 1),
            top_p=body.get("top_p", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
            max_attempts=max_api_attempts,
            rate_limiter=rate_limiter,
            extra_body=body.get("extra_body"),
        )
    else:
        raise ValueError(f"Unknown function name: {func_name}")

    if errs:
        all_errors.extend(errs)

    return idx, response, all_errors if all_errors else None, tries, finish_reason

def generate_parallel_completions(
    *,
    input_filepath: str,
    cache_filepath: str,
    num_workers: int = 20,
    checkpoint_every: int = 100,
    func_name: str = "chat_completions",
    requests_per_second: float = 0.0,
    max_api_attempts: int = 3,
) -> None:
    """
    Run chat completions with a thread pool and checkpoint progress.

    Args:
        input_filepath: Path to JSONL file with requests
        cache_filepath: Path to pickle file for caching results
        num_workers: Number of parallel workers (default 20)
        checkpoint_every: Save checkpoint every N results (default 100)
        func_name: "chat_completions" or "completions"
        requests_per_second: Rate limit (0 = no limit, default 0). If set, limits request rate.
        max_api_attempts: Max API-level retries per call passed to _process (default 3)
    """
    with open(input_filepath) as fh:
        requests_all = [json.loads(line) for line in fh]

    # resume cache
    done_results: List[ResultRow] = []
    done_idx: set[int] = set()
    if os.path.exists(cache_filepath):
        df_prev = pd.read_pickle(cache_filepath)
        # Handle both old format (response column) and new format (raw_response column)
        response_col = 'raw_response' if 'raw_response' in df_prev.columns else 'response'
        # Backward compat: old pickles lack finish_reason column
        finish_reason_col = 'finish_reason' if 'finish_reason' in df_prev.columns else None
        for r in df_prev.itertuples():
            cached_response = getattr(r, response_col, None)
            if cached_response is not None:
                fr = getattr(r, finish_reason_col, None) if finish_reason_col else None
                done_results.append((int(r.idx), cached_response, r.error, r.retries, fr))
        done_idx = {r[0] for r in done_results}
        logger.info("Loaded %d prior successes", len(done_idx))

    pending = [req for req in requests_all if int(req["custom_id"].split("_")[1]) not in done_idx]
    if not pending:
        logger.info("Nothing left to process; exiting.")
        return

    args_list: List[Tuple[int, Dict[str, Any]]] = [
        (int(req["custom_id"].split("_")[1]), req) for req in pending
    ]

    # Build a token bucket rate limiter when a rate limit is requested.
    # Workers call limiter.acquire() internally (inside _process), so the
    # submission loop no longer needs to sleep and can use all num_workers.
    limiter: Optional[TokenBucketRateLimiter] = None
    if requests_per_second > 0:
        burst = max(3, int(requests_per_second))
        limiter = TokenBucketRateLimiter(rate=requests_per_second, burst=burst)
        logger.info(
            "Token bucket rate limiter: %.2f req/s, burst=%d, workers=%d",
            requests_per_second, burst, num_workers,
        )

    results = done_results.copy()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _process, idx, req, func_name, limiter, max_api_attempts,
            ): idx
            for idx, req in args_list
        }

        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Requests")):
            try:
                result_row = fut.result()
                results.append(result_row)
            except Exception as exc:
                idx = futures[fut]
                logger.error("Worker crashed on idx %s: %s", idx, exc)
                results.append((idx, None, [repr(exc)], 0, None))

            if (i + 1) % checkpoint_every == 0:
                _save(cache_filepath, results)

    _save(cache_filepath, results)
    logger.info("Finished %d total results.", len(results))


def _save(path: str, rows: List[ResultRow]):
    df = pd.DataFrame({
        "idx": [r[0] for r in rows],
        "response": [r[1] for r in rows],
        "error": [r[2] for r in rows],
        "retries": [r[3] for r in rows],
        "finish_reason": [r[4] if len(r) > 4 else None for r in rows],
    })
    df.sort_values("idx", inplace=True)
    df.to_pickle(path)

def batch_completions_template(
    input_prompt: str,
    model: str = 'gpt-4o',
    client_name: str = '',
    custom_id: str = '',
    temperature: float = 0.0,
    max_tokens: int = 32768,
    n: int = 1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[list[str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
):
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "prompt": input_prompt,
        "max_tokens": max_tokens,
        "n": n,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
    }
    if extra_body is not None:
        body["extra_body"] = extra_body
    query_template = {
        "custom_id": custom_id,
        "client_name": client_name,
        "method": "POST",
        "url": "/v1/completions",
        "body": body,
    }
    return query_template

def batch_chat_completions_template(
    input_prompt: str,
    developer_message: str = 'You are a helpful assistant',
    model: str = 'gpt-4o',
    client_name: str = '',
    custom_id: str = '',
    temperature: float = 0.0,
    max_tokens: int = 32768,
    n: int = 1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[list[str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
):
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "developer", "content": developer_message},
            {"role": "user", "content": input_prompt},
        ],
        "max_tokens": max_tokens,
        "n": n,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
    }
    if extra_body is not None:
        body["extra_body"] = extra_body
    query_template = {
        "custom_id": custom_id,
        "client_name": client_name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return query_template

def cache_batch_query(filepath: str, query: dict):
    with open(filepath, 'a') as f:
        f.write(json.dumps(query) + '\n')
