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

class RandomThoughtGenerator(OpenLMEngine):
    def __init__(self,
                 task: str,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_size: int = 1000,
                 results_dir: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 max_num_batched_tokens: int = 16384,
                 **kwargs
                 ):
        self.nick_name = nick_name
        self.task = task
        self.results_dir = results_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.dataset_size = dataset_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens

        self.output_dir = os.path.join(self.results_dir, "random_thoughts")
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

        print(f"Start generating random thoughts: {self.nick_name}")

        self.output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        if os.path.exists(self.output_path):
            self.existing_data = pd.read_pickle(self.output_path)
        else:
            self.existing_data = pd.DataFrame()
    
    def generate_prompts(self) -> list:
        if self.task == "math":
            topics = [
                "Algebra", "Calculus", "Applied Mathematics", "Geometry", "Number Theory", "Discrete Mathematics", "Differential Equations",
                "Abstract Algebra", "Algebraic Expressions", "Algorithms", "Category Theory", "Combinatorics", "Congruences", "Differential Calculus",
                "Differential Geometry", "Field Theory", "Geodesics", "Graph Theory", "Group Theory", "Hyperbolic Geometry", "Integral Calculus",
                "Lie Algebras", "Manifolds", "Mathematical Statistics", "Non-Euclidean Geometry", "Ring Theory"
                ]
            prompts = []
            for topic in topics:
                prompt = self.tokenizer.apply_chat_template(

                    [
                        {"role": "system", "content": "You are the smartest math student."},
                        {"role": "user", "content": f"You need to solve a very difficult {topic} problem. You need to think extensively about this problem."}
                    
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
        elif self.task == "countdown":
            raise NotImplementedError("Countdown task not supported yet")
        else:
            raise ValueError(f"Task {self.task} not supported")
        
        return prompts

    def count_think_tokens (self, prompt):
        if "</think>" in prompt:
            thinking = prompt.split("</think>")[0]
        else:
            thinking = prompt
            
        return len(self.tokenizer.encode(thinking))
    
    def generate_random_thoughts(self) -> None:
        try:
            prompts = self.generate_prompts()
            dataset_size_per_template = self.dataset_size // len(prompts)
            prompts = prompts * dataset_size_per_template
            
            self.response = self.generate(prompts=prompts).rename(columns = {'response': 'random_thought'})
            self.response["prompt"] = prompts

            self.df = pd.concat([self.existing_data[['prompt', 'random_thought']], self.response[['prompt', 'random_thought']]])
            self.df["think_tokens"] = self.df["random_thought"].apply(lambda x: self.count_think_tokens(x))

            # Save results
            self.df.to_pickle(self.output_path)
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--task", type=str, required=True, help="Name of the task to generate prompts for")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of random thoughts to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384,
                        help="Maximum number of tokens to batch")
    
    args = parser.parse_args()
    
    engine = RandomThoughtGenerator(**vars(args))
    engine.generate_random_thoughts()