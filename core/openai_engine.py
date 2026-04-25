"""Module for managing batch processing of GPT model queries with caching and parallel execution."""

import os
from typing import Any, Dict, Optional

try:
    from . import openaiapi
except ImportError:
    import openaiapi
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenAI_Engine():
    def __init__(
        self,
        input_df: pd.DataFrame,
        prompt_template: str = "",
        developer_message: str = "",
        template_map: dict[str, str] = {},
        nick_name: str = "gpt_engine",
        batch_io_root: str = "/home/al2644/research/openai_batch_io/reasoning",
        cache_filepath: str = "",
        model: str = "deepseek-chat",
        client_name: str = "openai",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        n: int = 1,
        mode: str = "chat_completions",
        requests_per_second: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.input_df = input_df
        self.prompt_template = prompt_template
        self.developer_message = developer_message
        self.template_map = template_map

        root = Path(batch_io_root) if batch_io_root else Path(os.environ.get("BATCH_IO_ROOT", ""))
        self.input_filepath = root / f"{nick_name}_input.jsonl"
        self.cache_filepath = cache_filepath if cache_filepath else root / f"{nick_name}_cache.pickle"

        self.model = model
        self.client_name = client_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n
        self.mode = mode
        self.requests_per_second = requests_per_second
        # Provider-specific reasoning controls forwarded per-request into the
        # JSONL body so the workers send them on every call.
        self.extra_body = extra_body

    def prepare_chat_completions_input(self):
        """Prepare batch input file with prompts formatted from the input dataframe."""
        assert self.input_filepath is not None, 'input_filepath is required'

        if self.input_filepath.exists():
            self.input_filepath.unlink()

        for idx, row in tqdm(self.input_df.iterrows(), total=len(self.input_df)):
            if self.template_map:
                properties = {
                    k: getattr(row, v) if v in self.input_df.columns else v
                    for k, v in self.template_map.items()
                }
            input_prompt = self.prompt_template.format(**properties)

            query = openaiapi.batch_chat_completions_template(
                input_prompt=input_prompt,
                developer_message=self.developer_message,
                model=self.model,
                client_name=self.client_name,
                custom_id=f'idx_{idx}',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
                top_p=self.top_p,
                extra_body=self.extra_body,
            )

            openaiapi.cache_batch_query(self.input_filepath, query)

        logger.info(f'Batch input prepared and stored at {self.input_filepath}')

    def prepare_chat_completions_prefill_input(self):
        """Prepare chat-completions input using assistant-prefill messages.

        Each row must provide ``inquiry`` and ``reasoning`` columns. The built
        body has messages=[{user: inquiry}, {assistant: "<think>"+reasoning}],
        which lets providers (e.g. DeepSeek V4-Pro) that don't emit `</think>`
        on /v1/completions still accept the architect's partial reasoning as a
        continuation seed.
        """
        assert self.input_filepath is not None, 'input_filepath is required'
        assert 'inquiry' in self.input_df.columns, "prefill mode requires 'inquiry' column"
        assert 'reasoning' in self.input_df.columns, "prefill mode requires 'reasoning' column"

        if self.input_filepath.exists():
            self.input_filepath.unlink()

        for idx, row in tqdm(self.input_df.iterrows(), total=len(self.input_df)):
            query = openaiapi.batch_chat_completions_prefill_template(
                inquiry=row['inquiry'],
                reasoning=row['reasoning'],
                model=self.model,
                client_name=self.client_name,
                custom_id=f'idx_{idx}',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
                top_p=self.top_p,
                extra_body=self.extra_body,
            )
            openaiapi.cache_batch_query(self.input_filepath, query)

        logger.info(f'Prefill chat-completions input prepared at {self.input_filepath}')

    def prepare_completions_input(self):
        """Prepare batch input file with prompts formatted from the input dataframe."""
        assert self.input_filepath is not None, 'input_filepath is required'

        if self.input_filepath.exists():
            self.input_filepath.unlink()

        for idx, row in tqdm(self.input_df.iterrows(), total=len(self.input_df)):
            input_prompt = row['prompt']

            query = openaiapi.batch_completions_template(
                input_prompt=input_prompt,
                model=self.model,
                client_name=self.client_name,
                custom_id=f'idx_{idx}',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
                top_p=self.top_p,
                extra_body=self.extra_body,
            )

            openaiapi.cache_batch_query(self.input_filepath, query)

        logger.info(f'Batch input prepared and stored at {self.input_filepath}')

    def run_model(self, overwrite=False, num_workers=20):
        """Run batch generation, optionally overwriting existing cached results."""
        if self.mode == 'chat_completions':
            self.prepare_chat_completions_input()
            func_name = "chat_completions"
        elif self.mode == 'completions':
            self.prepare_completions_input()
            func_name = "completions"
        elif self.mode == 'chat_completions_prefill':
            # Assistant-prefill chat mode: bypass template formatting and send
            # [{user: inquiry}, {assistant: "<think>"+reasoning}] as messages.
            # Required for providers (e.g. DeepSeek V4-Pro) that only surface
            # reasoning via /v1/chat/completions's reasoning_content field.
            self.prepare_chat_completions_prefill_input()
            func_name = "chat_completions"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if overwrite and Path(self.cache_filepath).exists():
            raise ValueError(
                f'The cache file {self.cache_filepath} already exists. '
                'Please manually delete this file for security reasons.'
            )

        openaiapi.generate_parallel_completions(
            input_filepath=self.input_filepath,
            cache_filepath=self.cache_filepath,
            num_workers=num_workers,
            func_name=func_name,
            requests_per_second=self.requests_per_second,
        )
        logger.info(f'Results are generated and stored at {self.cache_filepath}')

    def retrieve_outputs(self, overwrite=False):
        """Retrieve generated outputs from the cache pickle."""
        if self.cache_filepath and Path(self.cache_filepath).exists():
            logger.info(f'Results are retrieved from {self.cache_filepath}')
            return pd.read_pickle(self.cache_filepath)
        raise ValueError(
            f'The cache file {self.cache_filepath} does not exist. Please run the model first.'
        )
