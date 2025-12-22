"""Module for managing batch processing of GPT model queries with caching and parallel execution."""

import os
import json

from core import openaiapi
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
        system_message: str = "",
        developer_message: str = "",
        template_map: dict[str, str] = {},
        nick_name: str = "gpt_engine",
        batch_io_root: str = "../openai_batch_io",
        cache_filepath: str = "",
        model: str = "deepseek-chat",
        client_name: str = "openai",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        n: int = 1,
        batch_size: int = 20,
        mode: str = "chat_completions",
        batch_rate_limit: int = 10,
    ):
        self.input_df = input_df
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.developer_message = developer_message
        self.template_map = template_map

        root = Path(batch_io_root) if batch_io_root else Path(os.environ.get("BATCH_IO_ROOT", ""))
        self.input_filepath = root / f"{nick_name}_input.jsonl"
        self.batch_log_filepath = root / f"{nick_name}_batch_log.json"
        self.cache_filepath = cache_filepath if cache_filepath else root / f"{nick_name}_cache.pickle"

        self.model = model
        self.client_name = client_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n
        self.batch_size = batch_size
        self.mode = mode
        self.batch_rate_limit = batch_rate_limit

        assert self.n == 1 or self.temperature > 0.0, "When n > 1, temperature must be greater than 0.0"

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
                system_message=self.system_message,
                developer_message=self.developer_message,
                model=self.model,
                client_name=self.client_name,
                custom_id=f'idx_{idx}',
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
                top_p=self.top_p
            )

            openaiapi.cache_batch_query(self.input_filepath, query)

        logger.info(f'Batch input prepared and stored at {self.input_filepath}')

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
                top_p=self.top_p
            )

            openaiapi.cache_batch_query(self.input_filepath, query)

        logger.info(f'Batch input prepared and stored at {self.input_filepath}')

    def run_model(self, overwrite=False, num_workers=20):
        """Run the GPT model batch generation, optionally overwriting existing results."""
        if self.model == 'gpt-4o' and self.batch_rate_limit is None:
            self.batch_rate_limit = 20

        if self.mode == 'chat_completions' or not self.batch_log_filepath.exists():
            '''Prepare batch input'''
            self.prepare_chat_completions_input()

            if self.mode == 'chat_completions':
                if overwrite and Path(self.cache_filepath).exists():
                    raise ValueError(f'The cache file {self.cache_filepath} already exists. Please manually delete this file for security reasons.')
                openaiapi.generate_parallel_completions(input_filepath=self.input_filepath,
                                                    cache_filepath=self.cache_filepath,
                                                    num_workers=num_workers,
                                                    func_name="chat_completions"
                                                    )
                logger.info(f'Results are generated and stored at {self.cache_filepath}')

            elif self.mode == 'batch_chat_completions':
                openaiapi.minibatch_stream_generate_response(input_filepath=self.input_filepath,
                                                             batch_log_filepath=self.batch_log_filepath,
                                                             batch_size=self.batch_size,
                                                             batch_rate_limit=self.batch_rate_limit)
        elif self.mode == 'completions':
            self.prepare_completions_input()
            openaiapi.generate_parallel_completions(input_filepath=self.input_filepath,
                                                    cache_filepath=self.cache_filepath,
                                                    num_workers=num_workers,
                                                    func_name="completions"
                                                    )
        
        logger.info(f'Results are generated and check {self.batch_log_filepath}')

    def retrieve_outputs(self, overwrite=False, cancel_in_progress_jobs: bool = False):
        """Retrieve generated outputs from cache or batch logs."""
        if self.cache_filepath and Path(self.cache_filepath).exists() \
            and (self.mode == 'chat_completions' or self.mode == 'batch_chat_completions'):
            logger.info(f'Results are retrieved from {self.cache_filepath}')
            output_df = pd.read_pickle(self.cache_filepath)
        
        elif self.mode == 'batch_chat_completions' and overwrite:
            with open(self.batch_log_filepath) as f:
                batch_logs = json.load(f)
            output_dict = {}
            for idx, batch_log_id in tqdm(batch_logs.items()):
                status = openaiapi.check_batch_status(batch_log_id)
                if status == 'completed':
                    output_file_id = openaiapi.retrieve_batch_output_file_id(batch_log_id)
                    output_dict[idx] = output_file_id
                elif cancel_in_progress_jobs:
                    logger.info(f'Batch {batch_log_id} at {idx} failed. Cancel {batch_log_id}')
                    openaiapi.cancel_batch(batch_log_id)
                else:
                    logger.info(f'Batch {batch_log_id} at {idx} failed')

            output_df = openaiapi.minibatch_retrieve_response(output_dict=output_dict)
            output_df.to_pickle(self.cache_filepath)
            logger.info(f'Results are retrieved and stored at {self.cache_filepath}')
        else:
            raise ValueError(f'The cache file {self.cache_filepath} does not exist. Please run the model first.')

        return output_df