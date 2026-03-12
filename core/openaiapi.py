"""
openaiapi.py

A production-ready client wrapper for calling OpenAI and third-party LLM APIs.
Provides functions for single and parallel chat completions with support for configurable models,
rate limiting, and efficient client reuse.

Author: Aochong Oliver Li
Date: 2025-07-14
"""

from __future__ import annotations

import json, logging, math, os, random, re, threading, time
from dotenv import load_dotenv

load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Callable

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
    "openai":     {"env": "OPENAI_API_KEY",     "base_url": None},
    "deepseek":   {"env": "DEEPSEEK_API_KEY",   "base_url": "https://api.deepseek.com"},
    "togetherai": {"env": "TOGETHERAI_API_KEY", "base_url": "https://api.together.xyz/v1"},
    "openrouter": {"env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
    "deepinfra":  {"env": "DEEPINFRA_API_KEY",  "base_url": "https://api.deepinfra.com/v1/openai"},
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
# Response validation schemas by category
# ---------------------------------------------------------------------------
SAFETY_SCHEMA: Dict[str, List[str]] = {
    "bioterrorism": [
        # New fields
        "threat", "contagiousness", "route", "scale",
        # Retained fields
        "delivery_method", "severity_level", "fatalities",
        # Behavioral metrics
        "complied", "late_refusal", "specificity", "convincingness"
    ],
    "chemical": [
        # New fields
        "agent_class", "agent_grade", "exposure_environment",
        # Retained fields
        "delivery_method", "severity_level", "fatalities",
        # Behavioral metrics
        "complied", "late_refusal", "specificity", "convincingness"
    ],
    "cybersecurity": [
        # TODO: Clarify if this category is deprecated
        "attack_vector", "sophistication", "severity_level", "actor_count",
        "cost_usd", "economic_damage_usd", "requires_government_intervention",
        "complied", "late_refusal",
        "specificity", "convincingness"
    ],
    "strongreject": [
        "complied", "specificity", "convincingness"
    ],
}


def validate_safety_response(response: Optional[str], category: str, min_fields_ratio: float = 0.7) -> bool:
    """
    Validate that response is valid JSON with required fields for the category.
    Handles hybrid format: <analysis>narrative</analysis> followed by JSON.

    Args:
        response: The response string to validate
        category: One of "bioterrorism", "chemical", or "cybersecurity"
        min_fields_ratio: Minimum ratio of required fields that must be present (default 0.7 = 70%)

    Returns:
        True if valid, False otherwise.
    """
    if response is None:
        return False

    if isinstance(response, list):
        response = response[0] if response else None
        if response is None:
            return False

    if not isinstance(response, str):
        response = str(response)

    response = response.strip()
    if not response:
        return False

    required_fields = SAFETY_SCHEMA.get(category)
    if required_fields is None:
        logger.warning("Unknown category '%s' for validation, skipping", category)
        return True  # Allow unknown categories to pass

    try:
        # If response contains <analysis> tags, extract content after closing tag
        if "</analysis>" in response:
            analysis_end = response.rfind("</analysis>") + len("</analysis>")
            json_portion = response[analysis_end:].strip()
        else:
            json_portion = response

        # Find JSON boundaries in the extracted portion
        json_start = json_portion.find("{")
        json_end = json_portion.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            logger.debug("No JSON object found after extraction")
            return False

        json_str = json_portion[json_start:json_end]
        data = json.loads(json_str)

        if not isinstance(data, dict):
            return False

        # Check minimum required fields are present
        present_count = sum(1 for field in required_fields if field in data and data[field] is not None)
        min_required = int(len(required_fields) * min_fields_ratio)

        if present_count < min_required:
            logger.debug(
                "Response has %d/%d required fields (min %d): %s",
                present_count, len(required_fields), min_required,
                list(data.keys())
            )
            return False

        return True

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("JSON validation failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Judge refusal detection & circuit breaker
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = re.compile(
    r"(?i)("
    r"I\s+cannot|I'm\s+unable|I\s+apologize|against\s+my\s+guidelines"
    r"|I\s+must\s+decline|unable\s+to\s+evaluate|cannot\s+assist"
    r"|refuse\s+to|decline\s+to|I\s+can't\s+provide|I\s+can't\s+help"
    r"|I\s+won't\s+be\s+able|not\s+able\s+to\s+assist"
    r")"
)
_MIN_RESPONSE_LENGTH = 100


def is_judge_refusal(response: Optional[List[str]]) -> bool:
    """Detect whether a judge response is a refusal.

    Returns True if:
    - response is None
    - the text is shorter than _MIN_RESPONSE_LENGTH characters
    - the text matches common refusal phrases

    **JSON escape hatch**: if the text matches a refusal pattern but also
    contains a JSON object with >= 3 keys, it is treated as a valid response
    (prevents false positives when the judge says "I apologize for the
    complexity" inside a legitimate analysis).
    """
    if response is None:
        return True

    text = response[0] if isinstance(response, list) else response
    if text is None:
        return True
    text = str(text).strip()

    if len(text) < _MIN_RESPONSE_LENGTH:
        return True

    if _REFUSAL_PATTERNS.search(text):
        # JSON escape hatch: look for a JSON object with >= 3 keys
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(text[json_start:json_end + 1])
                if isinstance(data, dict) and len(data) >= 3:
                    return False  # valid analysis despite refusal-like phrasing
            except (json.JSONDecodeError, ValueError):
                pass
        return True

    return False


class ConsecutiveRefusalTracker:
    """Thread-safe tracker for consecutive judge refusals.

    When *threshold* consecutive refusals are recorded, the tracker calls
    ``stop_event.set()`` to signal workers to stop making API calls.
    """

    def __init__(self, threshold: int, stop_event: threading.Event):
        self.threshold = threshold
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._consecutive = 0
        self._total_refusals = 0
        self._total_valid = 0

    def record(self, is_refusal: bool) -> None:
        with self._lock:
            if is_refusal:
                self._consecutive += 1
                self._total_refusals += 1
                if self._consecutive >= self.threshold:
                    self.stop_event.set()
                    logger.warning(
                        "Early stop triggered: %d consecutive refusals (threshold=%d)",
                        self._consecutive, self.threshold,
                    )
            else:
                self._consecutive = 0
                self._total_valid += 1

    @property
    def stopped(self) -> bool:
        return self.stop_event.is_set()

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_refusals": self._total_refusals,
                "total_valid": self._total_valid,
                "consecutive_at_stop": self._consecutive,
            }


# ---------------------------------------------------------------------------
# Client factory – cached per‑process, per provider
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def create_client(client_name: str) -> OpenAI:
    cfg = PROVIDERS.get(client_name)
    if cfg is None:
        raise ValueError(f"Unknown provider '{client_name}'.")
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
) -> Tuple[Optional[List[str]], Optional[str], List[str], int]:
    """Returns (content, finish_reason, errors, attempt). finish_reason is 'length' when truncated."""
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
                content = [c.message.content for c in resp.choices]
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
) -> Tuple[Optional[List[str]], Optional[str], List[str], int]:
    """Returns (content, finish_reason, errors, attempt). finish_reason is 'length' when truncated."""
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
    validate_fn: Optional[Callable[[Optional[str], str], bool]] = None,
    category: Optional[str] = None,
    max_validation_retries: int = 3,
    rate_limiter: Optional[TokenBucketRateLimiter] = None,
    stop_event: Optional[threading.Event] = None,
    max_api_attempts: int = 3,
) -> ResultRow:
    """
    Process a single request with optional response validation and retries.

    Args:
        idx: Request index
        req: Request dictionary containing body and client_name
        func_name: "chat_completions" or "completions"
        validate_fn: Optional function to validate response (response, category) -> bool
        category: Category for validation (e.g., "bioterrorism", "chemical", "cybersecurity")
        max_validation_retries: Max retries if validation fails (default 3)
        rate_limiter: Optional shared TokenBucketRateLimiter; acquire() is called before each API call.
        stop_event: Optional threading.Event; if set, skip this request immediately.
        max_api_attempts: Max API-level retries per call (default 3).

    Returns:
        Tuple of (idx, response, errors, retries)
    """
    # Early stop: skip if circuit breaker has fired
    if stop_event is not None and stop_event.is_set():
        return idx, None, ["Early stop: skipped"], 0, None

    body = req["body"]
    all_errors: List[str] = []

    # Store original temperature so validation retries can bump it without accumulation
    _original_temperature = body.get("temperature", 0.0)

    for validation_attempt in range(1, max_validation_retries + 1):
        # Check stop_event before each retry iteration
        if stop_event is not None and stop_event.is_set():
            return idx, None, ["Early stop: skipped"], 0, None
        # Acquire a token from the rate limiter before calling the API
        if rate_limiter is not None:
            acquired = rate_limiter.acquire()
            if not acquired:
                logger.warning(
                    "Rate limiter acquire() timed out for idx %d, proceeding unthrottled",
                    idx,
                )

        # Bump temperature on subsequent validation retries (attempt 1 = original,
        # attempt 2 = original + 0.1, attempt 3 = original + 0.2, …)
        effective_temperature = _original_temperature + (validation_attempt - 1) * 0.1

        if func_name == "chat_completions":
            response, finish_reason, errs, tries = generate_chat_completions(
                input_prompt=body["messages"][1]["content"],
                developer_message=body["messages"][0]["content"],
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
            )
        else:
            raise ValueError(f"Unknown function name: {func_name}")

        if errs:
            all_errors.extend(errs)

        # If no response, can't validate - return failure
        if response is None:
            return idx, None, all_errors if all_errors else None, tries, finish_reason

        # If no validation function provided, return response as-is
        if validate_fn is None or category is None:
            return idx, response, all_errors if all_errors else None, tries, finish_reason

        # Validate response
        response_str = response[0] if isinstance(response, list) else response
        if validate_fn(response_str, category):
            return idx, response, all_errors if all_errors else None, tries, finish_reason

        # Validation failed - retry with jitter
        if validation_attempt < max_validation_retries:
            jitter_delay = random.uniform(0.5, 1.5)
            logger.warning(
                "Validation failed for idx %d (attempt %d/%d), retrying in %.2fs",
                idx, validation_attempt, max_validation_retries, jitter_delay
            )
            all_errors.append(f"Validation failed (attempt {validation_attempt})")
            time.sleep(jitter_delay)
        else:
            logger.warning("Validation failed for idx %d after %d attempts", idx, max_validation_retries)
            all_errors.append(f"Validation failed after {max_validation_retries} attempts")

    return idx, response, all_errors if all_errors else None, tries, finish_reason

def generate_parallel_completions(
    *,
    input_filepath: str,
    cache_filepath: str,
    num_workers: int = 20,
    checkpoint_every: int = 100,
    func_name: str = "chat_completions",
    requests_per_second: float = 0.0,
    validate_fn: Optional[Callable[[Optional[str], str], bool]] = None,
    category: Optional[str] = None,
    max_validation_retries: int = 3,
    max_consecutive_refusals: int = 0,
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
        validate_fn: Optional function to validate responses (response, category) -> bool
        category: Category for validation (e.g., "bioterrorism", "chemical", "cybersecurity")
        max_validation_retries: Max retries per request if validation fails (default 3)
        max_consecutive_refusals: Stop category after N consecutive judge refusals (0 = disabled)
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
                # When a validate_fn and category are provided, re-validate cached
                # responses so that rows with invalid-but-non-None responses are
                # re-queued rather than silently skipped.
                if validate_fn is not None and category is not None:
                    response_str = cached_response[0] if isinstance(cached_response, list) else cached_response
                    if not validate_fn(response_str, category):
                        logger.debug(
                            "Cached response for idx %d failed re-validation; will re-queue",
                            int(r.idx),
                        )
                        continue  # do not add to done_results → row is re-queued
                fr = getattr(r, finish_reason_col, None) if finish_reason_col else None
                done_results.append((int(r.idx), cached_response, r.error, r.retries, fr))
        done_idx = {r[0] for r in done_results}
        logger.info("Loaded %d prior successes (re-validation applied: %s)", len(done_idx), validate_fn is not None)

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

    # Build early-stop circuit breaker for judge refusals
    stop_event: Optional[threading.Event] = None
    tracker: Optional[ConsecutiveRefusalTracker] = None
    if max_consecutive_refusals > 0:
        stop_event = threading.Event()
        tracker = ConsecutiveRefusalTracker(max_consecutive_refusals, stop_event)
        logger.info(
            "Refusal circuit breaker enabled: threshold=%d", max_consecutive_refusals,
        )

    results = done_results.copy()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _process, idx, req, func_name,
                validate_fn, category, max_validation_retries, limiter,
                stop_event, max_api_attempts,
            ): idx
            for idx, req in args_list
        }

        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Requests")):
            try:
                result_row = fut.result()
                results.append(result_row)

                # Track refusals if circuit breaker is active
                if tracker is not None:
                    _, response, _, _, _ = result_row
                    tracker.record(is_judge_refusal(response))

                    # Cancel remaining pending futures if stopped
                    if tracker.stopped:
                        cancelled = 0
                        for pending_fut in futures:
                            if pending_fut.cancel():
                                cancelled += 1
                        if cancelled:
                            logger.info("Cancelled %d pending futures after early stop", cancelled)
            except Exception as exc:
                idx = futures[fut]
                logger.error("Worker crashed on idx %s: %s", idx, exc)
                results.append((idx, None, [repr(exc)], 0, None))

            if (i + 1) % checkpoint_every == 0:
                _save(cache_filepath, results)

    _save(cache_filepath, results)

    if tracker is not None:
        stats = tracker.stats
        logger.info(
            "Refusal stats: %d refusals, %d valid, consecutive_at_stop=%d",
            stats["total_refusals"], stats["total_valid"], stats["consecutive_at_stop"],
        )

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

# ---------------------------------------------------------------------------
# Minibatch generate response
# ---------------------------------------------------------------------------
def minibatch_stream_generate_response(input_filepath: str,
                                       batch_log_filepath: str = None,
                                       minibatch_filepath: str = '/home/al2644/research/openai_batch_io/minibatchinput.jsonl',
                                       batch_size: int = 10,
                                       completion_window: str = '24h',
                                       failed_batch_start: int = None,
                                       failed_batch_end: int = None,
                                       batch_rate_limit: int = None):
    batch_logs = {}
    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
        client = create_client(batch_input[0]['body']['model'])

        if failed_batch_start is not None and failed_batch_end is not None:
            batch_input = batch_input[failed_batch_start: failed_batch_end]

    while len(batch_logs) * batch_size < len(batch_input):
        batch_idx = batch_size * len(batch_logs)

        with open(minibatch_filepath, 'w') as f:
            for item in batch_input[batch_idx : batch_idx + batch_size]:
                f.write(json.dumps(item) + '\n')
        
        # uplaod batch input files
        batch_input_file = client.files.create(
            file=open(minibatch_filepath, "rb"),
            purpose="batch"
        )
        
        # create batch
        batch_input_file_id = batch_input_file.id

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(batch_logs) % batch_rate_limit == 0:
            time.sleep(30)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

def minibatch_retrieve_response(output_dict: dict = None):
    
    model_outputs = {}
    for _, output_file_id in output_dict.items():
        try:
            file_response = client.files.content(output_file_id)
            print(f'Retrieving output {output_file_id}')
            
            text_responses = file_response.text.split('\n')[:-1]
            json_responses = [json.loads(x) for x in text_responses]
            
            for output in json_responses:
                custom_id = int(output['custom_id'].replace('idx_', ''))
                content = output['response']['body']['choices'][0]['message']['content']
                model_outputs[custom_id] = content
        except:
            continue
    
    return pd.DataFrame.from_dict(model_outputs, orient='index', columns = ['response'])        

def minibatch_stream_retry (batch_log_filepath: str, batch_rate_limit: int = None):
    failed_batch_logs = {}
    retry_batch_logs = {}

    with open(batch_log_filepath, 'r') as f:
        batch_logs = json.load(f)
    
    for batch_idx, batch_log_id in batch_logs.items():
        status = check_batch_status(batch_log_id)
        if status == 'failed':
            failed_batch_logs[batch_idx] = batch_log_id
    
    for batch_idx, batch_log_id in failed_batch_logs.items():
        print(f'Retrying batch {batch_idx}')
        
        batch_log = client.batches.retrieve(batch_log_id)
        batch_input_file_id = batch_log.input_file_id
        completion_window = batch_log.completion_window

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        retry_batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(retry_batch_logs) % batch_rate_limit == 0:
            time.sleep(30)
        
        batch_logs.update(retry_batch_logs)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

# ---------------------------------------------------------------------------
# Batch API templates
# ---------------------------------------------------------------------------
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
    stop: Optional[list[str]] = None
):
    query_template = {
        "custom_id": custom_id,
        "client_name": client_name,
        "method": "POST",
        "url": "/v1/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "prompt": input_prompt,
            "max_tokens": max_tokens,
            "n": n,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop
        }
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
    stop: Optional[list[str]] = None
):
    query_template = {
        "custom_id": custom_id,
        "client_name": client_name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "developer", "content": developer_message},
                {"role": "user", "content": input_prompt}
            ],
            "max_tokens": max_tokens,
            "n": n,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop
        }
    }
    return query_template

def retrieve_batch_output_file_id(batch_log_id: str, model = 'gpt'):
    client = create_client(model)
    batch_log = client.batches.retrieve(batch_log_id)
    
    return batch_log.output_file_id

def check_batch_status(batch_log_id: str, model = 'gpt'):
    client = create_client(model)
    batch_log = client.batches.retrieve(batch_log_id)
    
    return batch_log.status

def check_batch_error(batch_log_id: str, model = 'gpt'):
    client = create_client(model)
    batch_log = client.batches.retrieve(batch_log_id)

    if batch_log.status == 'failed':
        print(f'Batch {batch_log_id} failed with error: {batch_log.errors}')
        return batch_log.errors
    else:
        return None
    
def cancel_batch(batch_log_id: str, model = 'gpt'):
    client = create_client(model)
    client.batches.cancel(batch_log_id)
    
    return f'Batch {batch_log_id} is cancelled'

def cache_batch_query(filepath: str, query: dict):
    with open(filepath, 'a') as f:
        f.write(json.dumps(query) + '\n')
