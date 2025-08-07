import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
import json
import random
import math
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM

from core.llm_engine import *
from core.openai_engine import *

from reward_score.math500 import math_if_boxed
from utils.chunk_r import equal_chunk
from utils.corrupt_num import *
from more_itertools import chunked

BUFFER_TOKENS = 8192

class CorruptNumbers(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 results_dir: str,
                 sample_size: int = 250,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 32768,
                 temperature: float = 0.6,
                 top_p: float = 1.0,
                 top_k: int = -1,
                 max_num_batched_tokens: int = 8192,
                 mini_batch_size: int = None,
                 granularity: int = 30,
                 overwrite: bool = False,
                 how: str = "fixed",
                 client_name: str = "",
                 **kwargs
                 ):
        # Initialize attributes first
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.nick_name = nick_name
        self.results_dir = results_dir
        self.sample_size = sample_size
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.granularity = granularity
        self.how = how
        self.client_name = client_name
        self.overwrite = overwrite

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, f"corrupt_numbers_{self.how}_window")
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}.pickle")) and not self.overwrite:
            print(f"Stress test (Corrupt Numbers) already exists: {self.nick_name}")
            exit()

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.max_position_embeddings = model.config.max_position_embeddings

        if self.client_name == '':
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

        # Load dataset from pickle file
        self.load_dataset()

        print(f"Start stress testing: {self.nick_name} on Number Corruption")

    def load_dataset(self) -> None:
        self.dataset_path = os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle")
        self.df = pd.read_pickle(self.dataset_path)
        
        self.df['response_tokens'] = self.df.apply(lambda x: len(self.tokenizer.encode(x['response'])) 
                                                    if x['model_is_correct'] else self.max_position_embeddings,
                                                    axis = 1)
      
        rate = self.df.groupby('problem').agg({'model_is_correct': 'sum', 'response_tokens': 'min'}) \
            .reset_index().rename(columns = {"model_is_correct": "solve_n", "response_tokens": "min_tokens"})

        prob_df = rate[(rate['solve_n'] > 0) & (rate['min_tokens'] < self.max_position_embeddings - BUFFER_TOKENS)] # choose problems that have at least one correct response and the response tokens are less than 8192

        inv_counts = (
            prob_df.groupby('solve_n')['problem']
            .transform('count')
            .rdiv(1.0)
        )
        prob_df['w'] = inv_counts / inv_counts.sum()

        chosen = prob_df.sample(n=self.sample_size, weights='w', random_state=42)['problem'].tolist()
        self.df = self.df[
            (self.df.problem.isin(chosen)) &
             (self.df.model_is_correct == 1) &
             (self.df.response_tokens < self.max_position_embeddings - BUFFER_TOKENS)
             ].reset_index(drop=True)
        self.df = self.df.drop_duplicates(subset = 'problem').reset_index(drop = True)

        self.df = self.df[['problem', 'solution', 'source', 'response']].rename(columns = {'response': 'original_response'})
        self.df = self.df.merge(prob_df[['problem', 'solve_n']], on = 'problem').reset_index(drop = True)

    def corrupt_thinking(self, how: str = "fixed", unit = 0.2, max_start_pos: float = 1.0, num_corrupt_sample: int = 4, seed: int = 42) -> None:
        rng = random.Random(seed)
        self.df["reasoning_chunks"] = self.df["original_response"].apply(self.process_response)
        self.df = self.df.dropna(how='any').reset_index(drop = True)
        
        def sample_fixed_window (row):
            # generate all possible start and end positions
            pos = np.arange(0, 1, unit)
            windows = [(round(p, 2), round(p + unit, 2)) for p in pos]

            chunks = row["reasoning_chunks"]
            n = len(chunks)
            
            res = {k: [] for k in ["start_pos",
                                   "end_pos",
                                   "prefix_reasoning",
                                   "corrupted_reasoning",
                                   "num_prefix_tokens",
                                   "num_corrupted_tokens"]}
            
            for start, end in windows:
                start_chunk_idx = max(0, int(start * n))
                end_chunk_idx = max(start_chunk_idx + 1, int(end * n))
                end_chunk_idx = min(end_chunk_idx, n)

                prefix_txt = '\n\n'.join(chunks[:start_chunk_idx])
                corrupted_txt = '\n\n'.join(chunks[start_chunk_idx:end_chunk_idx])
                num_prefix_tokens = len(self.tokenizer.encode(prefix_txt, add_special_tokens=False))
                num_corrupted_tokens = len(self.tokenizer.encode(corrupted_txt, add_special_tokens=False))
                
                res["start_pos"].append(start)
                res["end_pos"].append(end)
                res["prefix_reasoning"].append(prefix_txt)
                res["corrupted_reasoning"].append(corrupted_txt)
                res["num_prefix_tokens"].append(num_prefix_tokens)
                res["num_corrupted_tokens"].append(num_corrupted_tokens)

            return pd.Series(res, dtype = object)

        def sample_sliding_window (row):
            chunks = row["reasoning_chunks"]
            n = len(chunks)
            
            res = {k: [] for k in ["start_pos",
                                   "end_pos",
                                   "prefix_reasoning",
                                   "corrupted_reasoning",
                                   "num_prefix_tokens",
                                   "num_corrupted_tokens"]}

            for _ in range(num_corrupt_sample):
                start_chunk_idx = random.randint(0, int(n * max_start_pos))
                end_chunk_idx = start_chunk_idx + math.ceil(n * unit)
                end_chunk_idx = min(end_chunk_idx, n)
            
                start_pos, end_pos = round(start_chunk_idx / n, 2), round(end_chunk_idx / n, 2)
            
                prefix_txt = '\n\n'.join(chunks[:start_chunk_idx])
                corrupted_txt = '\n\n'.join(chunks[start_chunk_idx:end_chunk_idx])
                num_prefix_tokens = len(self.tokenizer.encode(prefix_txt, add_special_tokens=False))
                num_corrupted_tokens = len(self.tokenizer.encode(corrupted_txt, add_special_tokens=False))
            
                res["start_pos"].append(start_pos)
                res["end_pos"].append(end_pos)
                res["prefix_reasoning"].append(prefix_txt)
                res["corrupted_reasoning"].append(corrupted_txt)
                res["num_prefix_tokens"].append(num_prefix_tokens)
                res["num_corrupted_tokens"].append(num_corrupted_tokens)
            
            return pd.Series(res, dtype = object)

        self.df[["start_pos", "end_pos",
                 "prefix_reasoning", "corrupted_reasoning",
                 "num_prefix_tokens", "num_corrupted_tokens"]] = self.df.apply(sample_fixed_window 
                                                                                if how == "fixed" 
                                                                                else sample_sliding_window,
                                                                                axis=1)
        self.df = self.df.explode(column=["start_pos", "end_pos","prefix_reasoning", "corrupted_reasoning",
                                            "num_prefix_tokens", "num_corrupted_tokens"],
                                            ignore_index = True)
        
        self.corrupt_number(rng)
    
    def process_response(self, response: str) -> None:
        if "</think>" in response:
            reasoning = response.split('</think>')[0] + "</think>\n"
        else:
            reasoning = response

        return equal_chunk(reasoning, self.granularity)
    
    def corrupt_number(self, rng: random.Random) -> None:
        def math_corrupt_number(row):
            corrupted_reasoning = row["corrupted_reasoning"]
            numbers = extract_number(corrupted_reasoning)
            replacement = {number: perturb_number(number, rng) for number in list(numbers)}
            
            corrupted_reasoning = replace_number(corrupted_reasoning, replacement)
            
            return corrupted_reasoning, replacement
        
        self.df["corrupted_reasoning"], self.df["replacement"] = zip(*self.df.apply(math_corrupt_number, axis=1))

    def local_eval(self) -> None:
        self.responses = []
        for batch in chunked(list(self.df["prompt"]), self.mini_batch_size if self.mini_batch_size is not None else len(self.df)):
            new_sampling_params = [
                {
                    "max_tokens": min(self.max_tokens, self.max_position_embeddings - len(self.tokenizer.encode(prompt)) - 1)
                }
                for prompt in batch
            ]
            out = self.generate(prompts=batch, new_sampling_params=new_sampling_params)
            self.responses.append(out)
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'post_corruption_response'})

        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def api_eval(self) -> None:
        os.makedirs(self.output_dir + f"api", exist_ok=True)
        
        engine = OpenAI_Engine(
            input_df=self.df,
            nick_name=f"corrupt_numbers_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/reasoning",
            cache_filepath=self.output_dir + f"api/{self.nick_name}_api_responses.pickle",
            model=self.model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode="completions"
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs(overwrite=self.overwrite)

        self.response = self.response.explode(['response']).set_index('idx').rename(columns = {'response': 'post_corruption_response'})
        self.df = self.df.merge(self.response, left_index=True, right_index=True)

    def eval(self) -> None:
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True).\
            split("HANDLE")
        
        self.df["prompt"] = template_prefix + self.df["problem"] + template_suffix + \
            self.df['prefix_reasoning'] + self.df['corrupted_reasoning']

        if self.client_name == '':
            self.local_eval()
        else:
            self.api_eval()
        
        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        self.df.to_pickle(output_path)

        self.result_df = self.df.copy()
        self.result_df['pred'] = self.result_df['post_corruption_response'].apply(lambda x: x.split('</think>')[-1].strip() if '</think>' in x else x)
        self.result_df['ground_truth'] = self.result_df['solution']
        self.result_df['if_boxed'] = self.result_df['pred'].apply(math_if_boxed)

        self.result_df.to_pickle(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of problems to sample")

    parser.add_argument("--client_name", type=str, default="", help="Name of api client")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
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
                        help="Maximum number of tokens to batch")

    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini batch size for local evaluation")
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the reasoning chunks")
    parser.add_argument("--how", type=str, default="fixed",
                        help="How to corrupt the reasoning chunks")
    parser.add_argument("--unit", type=float, default=0.25,
                        help="Unit of the reasoning chunks")
    parser.add_argument("--max_start_pos", type=float, default=1.0,
                        help="Maximum start position of the reasoning chunks")
    parser.add_argument("--num_corrupt_sample", type=int, default=4,
                        help="Sample size of the reasoning chunks")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    
    args = parser.parse_args()
    
    engine = CorruptNumbers(
        **vars(args),
    )
    
    engine.corrupt_thinking(
        how=args.how,
        unit=args.unit,
        max_start_pos=args.max_start_pos,
        num_corrupt_sample=args.num_corrupt_sample,
        seed=args.seed
    )
    engine.eval()