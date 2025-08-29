import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from core.llm_engine import *
from core.openai_engine import *

from utils.chunk_r import equal_chunk
from utils.corrupt_num import *
from more_itertools import chunked

TARGET_METAINFO ={
    "DeepSeek-R1-0528": {
        "architect": "OpenThinker2-7B-DeepSeek",
        "prompt_template": '''<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n{reasoning}'''
    },
    "Qwen3-235B-A22B-Thinking-2507": {
        "architect": "OpenThinker3-7B-Qwen",
        "prompt_template": '''<|im_start|>user\{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}'''
    },
    "GPT-OSS-120B": {
        "architect": "OpenThinker3-7B-Qwen",
        "prompt_template": '''<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-08

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{question}<|end|><|start|>assistant<|channel|>analysis<|message|>{reasoning}'''
    }
}


class Inception(OpenLMEngine):
    def __init__(
        self,
        model_name: str,
        nick_name: str,
        tokenizer_name: str,
        results_dir: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        max_num_batched_tokens: int = 8192,
        mini_batch_size: int = None,
        sample_size: int = None,
        granularity: int = 30,
        unit: float = 0.2,
        overwrite: bool = False,
        client_name: str = "",
        **kwargs,
    ):
        # Initialize attributes first
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.sample_size = sample_size
        self.granularity = granularity
        self.unit = unit
        self.overwrite = overwrite
        self.client_name = client_name

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "inception")
        os.makedirs(self.output_dir, exist_ok=True)
        
        out_pickle = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Stress test (Inception) already exists: {self.nick_name}")
            exit()
        
        cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_position_embeddings = cfg.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if self.client_name == '':    
            # Initialize model config
            config = ModelConfig(
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_position_embeddings
            )

            # Initialize parent class
            super().__init__(config=config)
        
        self.load_dataset()
    
        print(f"Start stress testing: {self.nick_name} on inception")

    def load_dataset(self) -> None:
        self.dataset_path = os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle")
        self.df = pd.read_pickle(self.dataset_path)[["inquiry", "response", "thinking"]] \
            .rename(
                columns={"inquiry": "inquiry",
                "response": "original_response",
                "thinking": "original_reasoning"}
                )

        self.architect = TARGET_METAINFO[self.nick_name]["architect"]
        self.prompt_template = TARGET_METAINFO[self.nick_name]["prompt_template"]
        
        self.architect_df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{self.architect}.pickle"))
        self.architect_df = self.architect_df[["inquiry", "response", "thinking"]] \
            .rename(
                columns={"inquiry": "inquiry",
                "response": "architect_response",
                "thinking": "architect_reasoning"}
                )
        self.architect_df = self.architect_df.drop_duplicates(subset=["inquiry"])
        self.df = self.df.merge(self.architect_df, on="inquiry", how="left").reset_index(drop=True)
        if self.sample_size is not None:
            self.df = self.df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        
    def architect_inception(self, row):
        architect_ratio = row["architect_ratio"]
        architect_reasoning = row["architect_reasoning"].replace("<think>", "").strip("\n")

        architect_chunks = equal_chunk(architect_reasoning, self.granularity)
        n_architect = int(len(architect_chunks) * architect_ratio)
        architect_reasoning = "".join(architect_chunks[:n_architect])

        return architect_reasoning

    def inception(self):
        windows = [self.unit]
        self.df["architect_ratio"] = len(self.df) * [windows]
        self.df = self.df.explode("architect_ratio", ignore_index = True)

        self.df["inception"] = self.df.apply(self.architect_inception, axis = 1)
    
    def apply_chat_template(self, row):
        question = row["inquiry"]
        reasoning = row["inception"]
        prompt = self.prompt_template.format(question=question, reasoning=reasoning)

        return prompt

    def local_eval(self) -> None:
        self.responses = []
        self.mini_batch_size = self.df.shape[0] if self.mini_batch_size is None else self.mini_batch_size
        for batch in chunked(list(self.df["prompt"]), self.mini_batch_size):
            new_sampling_params = [
                {
                    "max_tokens": min(self.max_tokens, self.max_position_embeddings - len(self.tokenizer.encode(prompt)) - 1)
                }
                for prompt in batch
            ]
            out = self.generate(prompts=batch, new_sampling_params=new_sampling_params)
            self.responses.append(out)
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'incepted_response'})
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def api_eval(self) -> None:
        os.makedirs(self.output_dir + f"/api", exist_ok=True)

        engine = OpenAI_Engine(
            input_df=self.df,
            nick_name=f"inception_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/wmdp",
            cache_filepath=os.path.join(self.output_dir, f"api/{self.nick_name}_api_responses.pickle"),
            model=self.model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode="completions"
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs()
        self.response = self.response.explode(['response']).set_index('idx').rename(columns={'response': 'incepted_response'})
        self.df = self.df.merge(self.response[["incepted_response"]], left_index=True, right_index=True)

    def eval(self) -> None:
        self.df["prompt"] = self.df.apply(self.apply_chat_template, axis = 1)
        
        if self.client_name == '':
            self.local_eval()
        else:
            self.api_eval()
        
        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        self.df.to_pickle(output_path)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
    
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192,
                        help="Maximum number of tokens in a batch")
    
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini batch size for generation")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of problems to sample for the stress test")
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--unit", type=float, default=0.2,
                        help="Unit of the thinking chunks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")

    args = parser.parse_args()
    engine = Inception(
        **vars(args),
    )
    engine.inception()
    engine.eval()