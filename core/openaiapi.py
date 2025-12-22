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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
from openai import (OpenAI, APIError, APIConnectionError, RateLimitError,
                    Timeout)
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

RETRYABLE = (RateLimitError, APIError, APIConnectionError, Timeout)

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

# This is the old version. The new version is under test. Delete this later.
def generate_chat_completions(
    *,
    input_prompt: str,
    developer_message: str = "",
    system_message: str = "",
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
) -> Tuple[Optional[List[str]], List[str], int]:
    client = create_client(client_name)
    messages = [
        {"role": "user", "content": input_prompt},
    ]
    if system_message != "":
        messages.insert(0, {"role": "system", "content": system_message})
    elif developer_message != "":
        messages.insert(0, {"role": "developer", "content": developer_message})

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
            if model == "deepseek-reasoner":
                return [
                    f"{c.message.reasoning_content}\n</think>\n{c.message.content}" for c in resp.choices
                ], errors, attempt
            return [c.message.content for c in resp.choices], errors, attempt
        except RETRYABLE as exc:
            errors.append(exc)
            if attempt == max_attempts:
                logger.error("%s – final failure", exc)
                return None, errors, attempt
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
            errors.append(exc)
            logger.error("Non‑retryable error: %s", exc)
            return None, errors, attempt

    return None, errors, max_attempts

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
) -> Tuple[Optional[List[str]], List[str], int]:
    client = create_client(client_name)
    raise Exception("This is not useable yet. Still under test.")
    
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
            if model == "deepseek-reasoner":
                return [
                    f"{c.message.reasoning_content}\n</think>\n{c.message.content}" for c in resp.choices
                ], errors, attempt
            return [c.text for c in resp.choices], errors, attempt
        except RETRYABLE as exc:
            errors.append(repr(exc))
            if attempt == max_attempts:
                logger.error("%s – final failure", exc)
                return None, errors, attempt
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
            return None, errors, attempt

    return None, errors, max_attempts

# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------
ResultRow = Tuple[int, Optional[List[str]], List[str], int]

def _process(idx: int, req: Dict[str, Any], func_name: str) -> ResultRow:
    body = req["body"]
    if func_name == "chat_completions":
        
        messages = {}
        for msg in body["messages"]:
            if msg["role"] == "user":
                messages["user"] = msg["content"]
            elif msg["role"] == "system":
                messages["system"] = msg["content"]
            elif msg["role"] == "developer":
                messages["developer"] = msg["content"]
        
        response, errs, tries = generate_chat_completions(
            input_prompt=messages["user"],
            system_message=messages.get("system", ""),
            developer_message=messages.get("developer", ""),
            model=body["model"],
            client_name=req["client_name"],
            temperature=body.get("temperature", 0.0),
            max_tokens=body.get("max_tokens", 1024),
            n=body.get("n", 1),
            top_p=body.get("top_p", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
        )
    elif func_name == "completions":
        response, errs, tries = generate_completions(
            input_prompt=body["prompt"],
            model=body["model"],
            client_name=req["client_name"],
            temperature=body.get("temperature", 0.0),
            max_tokens=body.get("max_tokens", 1024),
            n=body.get("n", 1),
            top_p=body.get("top_p", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
        )
    else:
        raise ValueError(f"Unknown function name: {func_name}")

    return idx, response, errs if errs else None, tries

def generate_parallel_completions(
    *,
    input_filepath: str,
    cache_filepath: str,
    num_workers: int = 20,
    checkpoint_every: int = 100,
    func_name: str = "chat_completions"
) -> None:
    """Run chat completions with a thread pool and checkpoint progress."""
    with open(input_filepath) as fh:
        requests_all = [json.loads(line) for line in fh]

    # resume cache
    done_results: List[ResultRow] = []
    done_idx: set[int] = set()
    if os.path.exists(cache_filepath):
        df_prev = pd.read_pickle(cache_filepath)
        done_results.extend(
            [
                (int(r.idx), r.response, r.error, r.retries)
                for r in df_prev.itertuples()
                if r.response is not None
            ]
        )
        done_idx = {r[0] for r in done_results}
        logger.info("Loaded %d prior successes", len(done_idx))

    pending = [req for req in requests_all if int(req["custom_id"].split("_")[1]) not in done_idx]
    if not pending:
        logger.info("Nothing left to process; exiting.")
        return

    args_list: List[Tuple[int, Dict[str, Any]]] = [
        (int(req["custom_id"].split("_")[1]), req) for req in pending
    ]

    results = done_results.copy()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process, idx, req, func_name): idx for idx, req in args_list}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Requests")):
            try:
                results.append(fut.result())
            except Exception as exc:
                idx = futures[fut]
                logger.error("Worker crashed on idx %s: %s", idx, exc)
                results.append((idx, None, [repr(exc)], 0))

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
    system_message: str = '',
    developer_message: str = '',
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
    messages = [
        {"role": "user", "content": input_prompt}
    ]
    if system_message != '':
        messages.insert(0, {"role": "system", "content": system_message})
    elif developer_message != '':
        messages.insert(0, {"role": "developer", "content": developer_message})
    
    query_template = {
        "custom_id": custom_id,
        "client_name": client_name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "messages": messages,
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
