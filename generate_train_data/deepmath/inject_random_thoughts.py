import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import math

from core.llm_engine import *
from reward_score.math import math_compute_score

def truncate_to_tokens(text: str, tokenizer, limit: int = 2048) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:limit])

class InjectRandomThoughts(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_path: str,
                 distractor_dataset_path: str,
                 results_dir: str,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 8192,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 max_cot_tokens: int = 2048,
                 max_num_batched_tokens: int = 32768,
                 **kwargs
                 ):
        self.nick_name = nick_name
        self.results_dir = results_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.dataset_path = dataset_path
        self.distractor_dataset_path = distractor_dataset_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
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
            pipeline_parallel_size=self.pipeline_parallel_size,
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

        self.load_dataset()

    
    def load_dataset(self) -> None:
        self.df = pd.read_parquet(self.dataset_path)
        self.distractor_df = pd.read_pickle(self.distractor_dataset_path)

        self.df = self.df[~self.df['zero_success']].reset_index(drop = True)
        self.distractor_df = self.distractor_df[self.distractor_df['think_tokens'] > self.max_cot_tokens].reset_index(drop = True)
        self.distractor_df["trimmed_random_thoughts"] = self.distractor_df["random_thought"].apply(
            lambda x: truncate_to_tokens(x, self.tokenizer, self.max_cot_tokens)
        )

        problems = self.df["prompt"].apply(lambda x: x[0]['content']).tolist()
        ground_truths = self.df['reward_model'].apply(lambda x: x['ground_truth']).tolist()
        self.result_df = pd.DataFrame()
        self.prompts = pd.DataFrame(
            {
                "problem": problems,
                "ground_truth": ground_truths,
            }
        )

    def inject_thoughts(self) -> None:
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'HANDLE_ME'}], tokenize=False, add_generation_prompt=True).split("HANDLE_ME")
        
        def inject(row):
            problem = row["problem"]
            random_thought = self.distractor_df.sample(n = 1).iloc[0]["trimmed_random_thoughts"]            
            
            return f"{template_prefix}{problem}{template_suffix}{random_thought}", random_thought

        self.prompts['prompt'], self.prompts['random_thought'] = zip(*self.prompts.apply(inject, axis=1))

    def generate_response(self) -> None:
        while len(self.prompts) > 0:
            self.inject_thoughts()

            self.response = self.generate(prompts=list(self.prompts["prompt"])).rename(columns = {'response': 'post_inject_response'})
            self.response.index = self.prompts.index
            self.prompts = pd.concat([self.prompts[['problem', 'ground_truth', 'prompt', 'random_thought']], self.response], axis=1)
            
            correctness = []
            for _, row in self.prompts.iterrows():
                solution = row['post_inject_response']
                ground_truth = row['ground_truth']
                
                score = math_compute_score(solution, ground_truth)
                correctness.append(score)
            
            # Combine results
            self.prompts['still_correct'] = correctness
            
            self.result_df = pd.concat([self.result_df, self.prompts[self.prompts['still_correct'] == 0.]], axis=0)
            self.prompts = self.prompts[self.prompts['still_correct'] == 1.].drop(columns = ['post_inject_response'])

            output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
            self.result_df.to_pickle(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to evaluate on")
    parser.add_argument("--distractor_dataset_path", type=str, required=True, help="Path to the distractor dataset")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory")
    
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Number of GPUs for pipeline parallelism")
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
    parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                        help="Maximum number of tokens to batch")
                        
    args = parser.parse_args()
    engine = InjectRandomThoughts(**vars(args))
    engine.generate_response()