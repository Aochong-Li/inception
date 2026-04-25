import os
import pandas as pd
from transformers import AutoModelForCausalLM
from core.llm_engine import *
from core.openai_engine import *

import argparse
from datasets import load_dataset, load_from_disk, Dataset

import numpy as np
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False

class BenchmarkEval(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_name_or_path: str = 'aochongoliverli/wmdp_biochem_inquiries_800',
                 subset_name: str = None,
                 split_name: str = 'test',
                 sample_size: int = None,
                 output_dir: str = './results/benchmark',
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 system_prompt: str = None,
                 max_tokens: int = 32768,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 top_k: int = -1,
                 sample_k: int = 1,
                 max_num_batched_tokens: int = 8192,
                 overwrite: bool = False,
                 client_name: str = '',
                 filename_suffix: str = '',
                 requests_per_second: float = 0.0,
                 mode: str = 'chat_completions',
                 extra_body: dict | None = None,
                 api_model_name: str | None = None,
                 ):

        self.model_name = model_name
        self.api_model_name = api_model_name or model_name
        self.nick_name = nick_name
        self.mode = mode
        self.extra_body = extra_body
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_k = sample_k
        self.overwrite = overwrite
        self.max_num_batched_tokens = max_num_batched_tokens
        self.client_name = client_name
        self.filename_suffix = filename_suffix
        self.system_prompt = system_prompt
        self.requests_per_second = requests_per_second

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filepath = os.path.join(
            self.output_dir, f"{self.nick_name}{self.filename_suffix if self.filename_suffix else ''}.pickle"
            )
        if not self.overwrite:
            if os.path.exists(self.output_filepath):
                print(f"Results already exist for {self.nick_name}")
                exit()

        # Load dataset
        self.load_dataset(
            dataset_name_or_path,
            subset_name,
            split_name,
            sample_size
        )

        if self.client_name == '':
            # Local inference via vLLM
            config = ModelConfig(
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                n = self.sample_k,
                max_num_batched_tokens=self.max_num_batched_tokens
            )
            _ = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            super().__init__(config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Start evaluating {self.nick_name} on dataset: {dataset_name_or_path} | subset: {subset_name} | split: {split_name} | avg@{self.sample_k}")

    def load_dataset(self, dataset_name: str, subset_name: str, split_name: str, sample_size: int) -> None:
        try:
            dataset = load_from_disk(dataset_name)[split_name]
        except Exception as e:
            print(f"Error loading from disk: {e}")
            try:
                dataset = load_dataset(dataset_name, subset_name)[split_name]
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}")
        if sample_size:
            self.df = pd.DataFrame(dataset).sample(n=sample_size, random_state=45).reset_index(drop = True)
        else:
            self.df = pd.DataFrame(dataset)

    def apply_chat_template (self, question: str):
        chat_history = [
            {'role': 'user', 'content': question}
        ]
        if self.system_prompt:
            chat_history.insert(0, {'role': 'system', 'content': self.system_prompt})

        tokenized_prompt = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize = False,
            add_generation_prompt = True
        )
        
        return tokenized_prompt
    
    def eval(self) -> None:
        if self.client_name == '':
            self.local_eval()
        else:
            self.api_eval()

        self.df = self.df.loc[np.repeat(self.df.index, self.sample_k)].reset_index(drop=True)
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)
        self.df.to_pickle(self.output_filepath)

        # Post-process
        self.df['thinking'] = self.df['response'].apply(lambda x: x.split('</think>')[0].strip() if x and '</think>' in x else x)
        self.df.to_pickle(self.output_filepath)
        
    def api_eval(self) -> None:
        os.makedirs(self.output_dir + "/api", exist_ok=True)
        self.df["prompt"] = self.df["inquiry"]

        engine = OpenAI_Engine(
            input_df=self.df,
            prompt_template="{prompt}",
            developer_message=self.system_prompt or "",
            template_map={"prompt": "prompt"},
            nick_name=f"benchmark_eval_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/inception-eval/benchmark/batch_io",
            cache_filepath=self.output_dir + f"/api/{self.nick_name}_api_responses.pickle",
            model=self.api_model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.sample_k,
            mode=self.mode,
            extra_body=self.extra_body,
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs(overwrite=self.overwrite)
        self.response = self.response.set_index('idx').explode(['response']).reset_index(drop=True)

    def local_eval(self) -> None:
        prompts = self.df['inquiry'].apply(self.apply_chat_template)
        self.response = self.generate(prompts=prompts)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Benchmark: direct query evaluation (no jailbreak)")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, default="aochongoliverli/wmdp_biochem_inquiries_800")
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--split_name", type=str, default='test')
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default='./results/benchmark')
    parser.add_argument("--filename_suffix", type=str, default="")

    # Local inference settings
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)

    # Generation settings
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--sample_k", type=int, default=1)

    # API settings
    parser.add_argument("--client_name", type=str, default='',
                        help="API provider (empty=local vLLM, deepinfra, togetherai, etc.)")
    parser.add_argument("--requests_per_second", type=float, default=0.0,
                        help="Rate limit for API calls (0=no limit)")

    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default="chat_completions",
                        help="OpenAI_Engine mode (chat_completions / completions / chat_completions_prefill)")
    parser.add_argument("--extra_body", type=str, default=None,
                        help="JSON string passed as extra_body to the API request")
    args = parser.parse_args()

    import json as _json
    from src.main import TARGET_MODEL_OVERRIDES
    args_dict = vars(args)
    _overrides = TARGET_MODEL_OVERRIDES.get(args_dict['model_name'], {})
    if 'client_name' in _overrides and not args_dict.get('client_name'):
        args_dict['client_name'] = _overrides['client_name']
    if 'mode' in _overrides:
        args_dict['mode'] = _overrides['mode']
    if 'extra_body' in _overrides:
        args_dict['extra_body'] = _overrides['extra_body']
    elif args_dict.get('extra_body'):
        args_dict['extra_body'] = _json.loads(args_dict['extra_body'])
    if 'api_model_name' in _overrides:
        args_dict['api_model_name'] = _overrides['api_model_name']

    engine = BenchmarkEval(**args_dict)
    engine.eval()