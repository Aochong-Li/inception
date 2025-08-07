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
from multiprocessing import Pool

from core.llm_engine import *

from reward_score.countdown import compute_score, extract_solution

def truncate_to_tokens(text: str, tokenizer, limit: int = 2048) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:limit])

class InjectRandomThoughts(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_path: str,
                 results_dir: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 sample_size: int = 50000,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 max_cot_tokens: int = 1024,
                 max_num_batched_tokens: int = 16384,
                 **kwargs
                 ):
        self.nick_name = nick_name
        self.results_dir = results_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.dataset_path = dataset_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.sample_size = sample_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_cot_tokens = max_cot_tokens
        self.max_num_batched_tokens = max_num_batched_tokens

        self.output_dir = os.path.join(self.results_dir, "inject_thoughts")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize model config
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
            max_num_batched_tokens=self.max_num_batched_tokens
        )

        super().__init__(config=config)
        self.load_dataset(sample_size = self.sample_size)

        print(f"Start generating random thoughts: {self.nick_name}")
    
    def load_dataset(self, sample_size: int = None) -> None:
        self.df = pd.read_pickle(self.dataset_path).rename(columns = {'response': 'original_response'})
        self.df = self.df[(self.df["correct"] == 1.) & self.df["original_response"].str.contains("</think>")]
        if sample_size is not None:
            self.df = self.df.sample(n = sample_size, random_state = 42).reset_index(drop = True)

        self.thoughts = pd.read_pickle(os.path.join(self.results_dir, "random_thoughts", f"{self.nick_name}.pickle"))
        self.thoughts = self.thoughts[self.thoughts["long_cot"]].reset_index(drop = True)
        self.thoughts["random_thought_trimmed"] = self.thoughts["random_thought"].apply(
            lambda t: truncate_to_tokens(t, self.tokenizer, self.max_cot_tokens)
        )
    
    def inject_thoughts(self) -> None:
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'HANDLE_ME'}], tokenize=False, add_generation_prompt=True).split("HANDLE_ME")
        
        def inject(row):
            problem = row["problem"]
            reasoning, _ = row["original_response"].split("</think>")
            reasoning = truncate_to_tokens(reasoning, self.tokenizer, self.max_cot_tokens)
            random_thought = self.thoughts.sample(n = 1).iloc[0]["random_thought_trimmed"]
            
            inject_type = random.choice(["full", "half"])
            
            if inject_type == "half":
                reasoning, random_thought = reasoning[:self.max_cot_tokens // 2], random_thought[:self.max_cot_tokens // 2]
                prompt = f"{template_prefix}{problem}{template_suffix}{reasoning}\n{random_thought}"
            elif inject_type == "full":
                prompt = f"{template_prefix}{problem}{template_suffix}<think>\n{random_thought}"
            
            return prompt, inject_type
        self.df[["prompt", "inject_type"]] = self.df.apply(inject, axis=1, result_type='expand')
    
    def generate_response(self) -> None:
        self.response = self.generate(prompts=self.df["prompt"]).rename(columns = {'response': 'post_inject_response'})
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)
        
        correctness = []
        if_answer = []
        for _, row in self.df.iterrows():
            solution = row['post_inject_response']
            
            # Clean solution if it contains thinking steps
            if "</think>" in solution:
                solution = solution.split("</think>")[-1]
            
            numbers, target = row['nums'], row['target']
            score = compute_score(solution, numbers, target)
            if_answer.append(extract_solution(solution) != None)

            correctness.append(score)
        
        # Combine results
        self.df['still_correct'] = correctness
        
        if if_answer:
            self.df['if_answer'] = if_answer

        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        self.df.to_pickle(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to evaluate on")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_cot_tokens", type=int, default=1024,
                        help="Maximum number of tokens for COT")
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384,
                        help="Maximum number of tokens to batch")
    parser.add_argument("--sample_size", type=int, default=50000,
                        help="Number of samples to use")
                        
    args = parser.parse_args()
    engine = InjectRandomThoughts(**vars(args))
    engine.inject_thoughts()
    engine.generate_response()