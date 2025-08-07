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

from reward_score.math import compute_score as math_compute_score
from reward_score.countdown import compute_score as countdown_compute_score, extract_solution

from utils.chunk_r import equal_chunk
from utils.corrupt_num import *
from tqdm.auto import tqdm

def process_single_row_sliding_window(args):
    row_idx, reasoning_chunks, unit, max_start_pos, sample_size, seed = args    
    chunks = reasoning_chunks
    n = len(chunks)
    
    rng = random.Random(seed + row_idx)
    results = []
    for _ in range(sample_size):
        start_chunk_idx = rng.randint(0, int(n * max_start_pos))
        end_chunk_idx = start_chunk_idx + math.ceil(n * unit)
        end_chunk_idx = min(end_chunk_idx, n)
        
        start_pos, end_pos = round(start_chunk_idx / n, 2), round(end_chunk_idx / n, 2)
        prefix_txt = '\n\n'.join(chunks[:start_chunk_idx])
        corrupted_txt = '\n\n'.join(chunks[start_chunk_idx:end_chunk_idx])
        
        results.append({
            'index': row_idx,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'prefix_reasoning': prefix_txt,
            'corrupted_reasoning': corrupted_txt
        })
    
    return pd.DataFrame(results)

def math_corrupt_number(args):
    idx, reasoning, seed = args
    rng = random.Random(seed +idx)
    numbers = extract_number(reasoning)
    replacement = {number: perturb_number(number, rng) for number in list(numbers)}
    
    corrupted_reasoning = replace_number(reasoning, replacement)
    
    return corrupted_reasoning, replacement

def countdown_corrupt_number (args):
    idx, reasoning, target, seed = args
    rng = random.Random(seed +idx)
    numbers = extract_number(reasoning)

    replacement = {number: perturb_number(number, rng) for number in list(numbers) if number != target}
    
    corrupted_reasoning = replace_number(reasoning, replacement)
    
    return corrupted_reasoning, replacement

class CorruptNumbers(OpenLMEngine):
    def __init__(self,
                 task: str,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 results_dir: str = None,
                 question_ids_fname: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 max_num_batched_tokens: int = 16384,
                 granularity: int = 20,
                 overwrite: bool = False,
                 generate_data_only: bool = False,
                 **kwargs
                 ):
        # Initialize attributes first
        self.task = task
        self.nick_name = nick_name
        self.results_dir = results_dir
        self.question_ids_fname = question_ids_fname
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.granularity = granularity

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "corrupt_numbers")
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}.pickle")) and not overwrite:
            print(f"Stress test (Corrupt Numbers) already exists: {self.nick_name}")
            exit()
        
        # Load dataset from pickle file
        self.load_dataset()
    
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

        if generate_data_only:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            super().__init__(config=config)

        print(f"Start stress testing: {self.nick_name} on corrupted digit")

    def load_dataset(self) -> None:
        """Load dataset from pickle file.
        
        Args:
            dataset_path: Path to the pickle file containing the dataset
        """
        self.dataset_path = os.path.join(self.results_dir, "stage2_rl_gen_seq", f"{self.nick_name}.pickle")
        self.df = pd.read_pickle(self.dataset_path)
        self.df = self.df[self.df["correct"] == 1.].reset_index(drop = True)
        self.df = self.df[(self.df["response"].str.contains("</think>"))]
        
        if self.question_ids_fname is not None:
            question_ids = json.load(open(os.path.join(self.results_dir, self.question_ids_fname)))
        else:
            question_ids = None
        
        try:
            if self.task == "math":
                self.compute_score = math_compute_score
                self.df = self.df[self.df['unique_id'].isin(question_ids)].reset_index(drop = True) if question_ids is not None else self.df
                self.df = self.df.drop_duplicates(subset = ["unique_id"]).reset_index(drop = True)
            elif self.task == "countdown":
                self.compute_score = countdown_compute_score
                self.df = self.df[self.df['problem'].isin(question_ids)].reset_index(drop = True) if question_ids is not None else self.df
            self.df = self.df.drop(columns = ['correct']).rename(columns = {'response': 'original_response'})
            
            self.df = self.df.sample(n = 5000).reset_index(drop = True) # TODO: testing
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")
    
    def corrupt_thinking_fast(self, unit: float = 0.2, max_start_pos: float = 1.0, sample_size: int = 4, seed: int = 42, n_processes: int = 4) -> None:
        """
        Optimized version using multiprocessing for large DataFrames (100k+ rows)
        
        Args:
            how: "fixed" or "sliding" window sampling
            unit: Size of the corruption window as a fraction
            max_start_pos: Maximum starting position for sliding window (fraction)
            sample_size: Number of samples for sliding window
            seed: Random seed
            n_processes: Number of processes to use (default: CPU count)
        """        
        print(f"Processing {len(self.df)} rows with {n_processes} processes...")
        
        # Process responses to get reasoning chunks
        self.df["reasoning_chunks"] = self.df["original_response"].apply(self.process_response)
        self.df = self.df.dropna(how='any').reset_index(drop=True)
        
        args_list = [
            (idx, row["reasoning_chunks"], unit, max_start_pos, sample_size, seed)
            for idx, row in self.df.iterrows()
        ]
        process_func = process_single_row_sliding_window
        # Process in parallel
        with Pool(n_processes) as pool:
            parts = list(tqdm(
                pool.imap(process_func, args_list),
                total=len(args_list),
                desc='Chunking reasoning traces'
            ))
        # Replace the DataFrame
        result_df = pd.concat(parts, axis=0, ignore_index=True).set_index('index')
        self.df = self.df.merge(result_df, left_index=True, right_index=True).reset_index(drop=True)
        print(f"Expanded to {len(self.df)} rows after window sampling")
        
        # Apply number corruption
        self.corrupt_number(seed = seed, n_processes = n_processes)

    def process_response(self, response: str) -> None:
        reasoning, _ = response.split('</think>')
        reasoning += "</think>\n"

        return equal_chunk(reasoning, self.granularity)
    
    def corrupt_number(self, seed: int = 42, n_processes: int = 10) -> None:
        if self.task == "countdown":
            args_list = [
                (idx, row["corrupted_reasoning"], row["target"], seed)
                for idx, row in self.df.iterrows()
            ]
        else:
            raise NotImplementedError(f"Corrupt number not implemented for task {self.task}")
        
        with Pool(n_processes) as pool:
            all_results = list(tqdm(
                pool.imap(countdown_corrupt_number if self.task == "countdown" else math_corrupt_number, args_list),
                total=len(args_list),
                desc='Corrupting numbers'
            ))
        
        self.df["corrupted_reasoning"], self.df["replacement"] = zip(*all_results)
        
        
    def eval(self) -> None:
        try:
            template_prefix, template_suffix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True).\
                split("HANDLE")
            
            self.df["prompt"] = template_prefix + self.df["problem"] + template_suffix + \
                self.df['prefix_reasoning'] + self.df['corrupted_reasoning']
            
            self.response = self.generate(prompts=self.df["prompt"]).rename(columns = {'response': 'post_corruption_response'})
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            correctness = []
            if_answer = []
            for _, row in self.df.iterrows():
                solution = row['post_corruption_response']
                
                # Clean solution if it contains thinking steps
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                if self.task == "math":
                    ground_truth = row.get("solution")
                    score = self.compute_score(solution, ground_truth)
                elif self.task == "countdown":
                    numbers, target = row['nums'], row['target']
                    score = self.compute_score(solution, numbers, target)
                    if_answer.append(extract_solution(solution) != None)

                correctness.append(score)
            
            # Combine results
            self.df['still_correct'] = correctness
            
            if if_answer:
                self.df['if_answer'] = if_answer
            
            # Save results
            output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
            self.df.to_pickle(output_path)
            
            # Log summary statistics
            print(f"Evaluation complete. Accuracy: {self.df['still_correct'].mean():.2%}")
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--question_ids_fname", type=str, required=False, default=None, help="stress test question ids file name")
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
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768,
                        help="Maximum number of tokens to batch")
    parser.add_argument("--granularity", type=int, default=20,
                        help="Granularity of the reasoning chunks")
    parser.add_argument("--unit", type=float, default=0.25,
                        help="Unit of the reasoning chunks")
    parser.add_argument("--max_start_pos", type=float, default=0.25,
                        help="Maximum starting position for sliding window")
    parser.add_argument("--sample_size", type=int, default=4,
                        help="Number of samples for sliding window")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--generate_data_only", action="store_true",
                        help="Generate data only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    engine = CorruptNumbers(
        **vars(args),
    )
    
    engine.corrupt_thinking_fast(
        unit = args.unit,
        max_start_pos = args.max_start_pos,
        sample_size = args.sample_size,
        seed = args.seed,
        n_processes = 10
        )

    if not args.generate_data_only:
        engine.eval()