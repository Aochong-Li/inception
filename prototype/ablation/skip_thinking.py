import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from llm_engine import *
import argparse
from reward_score.math import compute_score
import logging
from typing import List


class SkipThinking(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 results_dir: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 4096,
                 temperature: float = 0.6,
                 top_p: float = 1.0,
                 top_k: int = -1,
                 pass_at_k: int = 3,
                 overwrite: bool = False,
                 **kwargs
                 ):
        # Initialize all arguments as instance attributes
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
        self.pass_at_k = pass_at_k
        self.overwrite = overwrite

        self.output_dir = os.path.join(self.results_dir, f"skip_thinking")
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}.pickle")) and not overwrite:
            print(f"Dataset already exists: {self.nick_name}")
            exit()

        # Initialize the LLM engine
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
            n = self.pass_at_k
        )

        super().__init__(config=config)

        # Load dataset
        self.load_dataset()

        # Fill thinking
        self.skip_thinking()
    
    def load_dataset(self):
        thinking_df = pd.read_pickle(os.path.join(self.results_dir, f"benchmark/{self.nick_name}.pickle"))
        nonthinking_df = pd.read_pickle(os.path.join(self.results_dir, f"benchmark/{self.nick_name}_nothinking.pickle"))

        problems = set(thinking_df[thinking_df["correct"] == 1]["problem"]).difference(set(nonthinking_df[nonthinking_df["correct"] == 1]["problem"]))

        self.df = thinking_df[thinking_df["problem"].isin(problems)].drop(columns=["response", "correct"]).drop_duplicates(subset=["problem"], ignore_index=True)
    
    def skip_thinking(self):
        def prepare_prompt (row):
            problem = row["problem"]
           
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            if "<think>" in prompt:
                prompt = prompt.split("<think>")[0]
            prompt += "<think>\nOkay, I finished my thinking and I'm ready to answer the question.\n</think>\n"

            return prompt
        
        self.df["prompt"] = self.df.apply(prepare_prompt, axis=1)

    def eval(self):
        try:
            self.response = self.generate(prompts=self.df["prompt"]).rename(columns = {'response': 'post_corruption_response'})
            
            self.df = self.df.loc[np.repeat(self.df.index, self.pass_at_k)].reset_index(drop=True)
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            correctness = []
            for _, row in self.df.iterrows():
                solution = row['post_corruption_response']
                ground_truth = row['solution']
                
                # Clean solution if it contains thinking steps
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                score = compute_score(solution, ground_truth)
                correctness.append(score)
            
            # Combine results
            self.df['still_correct'] = correctness
            
            # Save results
            output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
            self.df.to_pickle(output_path)

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--nick_name", type=str, default="Qwen3-14B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--results_dir", type=str, default="./results/benchmark")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--pass_at_k", type=int, default=3)
    parser.add_argument("--overwrite", type=bool, default=False)
    args = parser.parse_args()

    engine = SkipThinking(**vars(args))
    engine.eval()