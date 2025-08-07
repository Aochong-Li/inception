import os
import pandas as pd
import sys
sys.path.append("/home/al2644/research")
from codebase.reasoning.llm_engine import *
import argparse
from reward_score.math import compute_score, last_boxed_only_string, remove_boxed
import random
from typing import Tuple
import logging
import json

class CorruptDigit(OpenLMEngine):
    def __init__(self,
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
                 **kwargs
                 ):
        # Initialize attributes first
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
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "corrupt_answer_digit")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            top_k=self.top_k
        )

        # Initialize parent class
        super().__init__(config=config)

        print(f"Start stress testing: {self.nick_name} on corrupted answer digit")

    def load_dataset(self) -> None:
        """Load dataset from pickle file.
        
        Args:
            dataset_path: Path to the pickle file containing the dataset
        """
        self.dataset_path = os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle")
        question_ids = json.load(open(os.path.join(self.results_dir, self.question_ids_fname)))

        try:
            self.df = pd.read_pickle(self.dataset_path)
            self.df = self.df[self.df['unique_id'].isin(question_ids) & (self.df['correct'] == 1)]
            self.df = self.df.drop_duplicates(subset = ["unique_id"]).reset_index(drop = True)
            self.df = self.df.drop(columns = ['correct']).rename(columns = {'response': 'original_response'})
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")

    def corrupt(self) -> None:
        """Corrupt the dataset by modifying numerical values.
        
        This method:
        1. Locates numerical values in the dataset
        2. Modifies the values
        3. Replaces the original values with corrupted ones
        """
    
        self.df['corrupt_reasoning'], self.df['original_digit'] = zip(*self.df['original_response'].apply(self.local_final_answer_digit))
        self.df = self.df.dropna(how='any').reset_index(drop = True)
        self.df['corrupt_digit'] = self.df['original_digit'].apply(self.modify)
        self.corrupt_answer_digit()
    
    def local_final_answer_digit(self, response: str) -> None:
        """Locate numerical values in the dataset that can be corrupted."""
        try:
            reasoning, solution = response.split('</think>')
            solution = remove_boxed(last_boxed_only_string(solution))
            all_digits = [c for c in solution if c.isdigit()]
            return reasoning, random.sample(all_digits, k = 1)[0]
        except:
            return None, None

    def modify(self, original_digit) -> None:
        """Modify the located numerical values."""
        original_digit = int(original_digit)
        replace_digit = original_digit + 1 if random.random() < 0.5 else original_digit - 1
        if replace_digit == -1:
            replace_digit = 9
        elif replace_digit == 10:
            replace_digit = 0
        
        return str(replace_digit)

    def corrupt_answer_digit(self) -> None:
        """Replace the original values with corrupted ones."""
        def corrupt_digit(row) -> Tuple[str, float]:
            string = row['corrupt_reasoning']
            original = row['original_digit']
            corrupted = row['corrupt_digit']
            percentile = row['percentile']
            """
            Replace the *last* percentile fraction of occurrences of `original` in `string`
            with `corrupted`, and return (new_string, actual_fraction_replaced).

            - percentile >= 1.0: replace *all* occurrences.
            - percentile <= 0.0: replace *only* the last occurrence.
            - 0.0 < percentile < 1.0: replace floor(percentile * total) occurrences, counting from the end;
            if that computes to 0, we still replace 1 occurrence.
            """
            # Case 0: no original digit
            total = string.count(original)
            # Case 1: replace everything
            if percentile >= 1.0:
                new_string = corrupted.join(string.split(original))
                return new_string, total

            # Case 2: replace only the last one
            if percentile <= 0.0:
                parts = string.rsplit(original, 1)
                new_string = corrupted.join(parts)
                return new_string, 1

            # Case 3: intermediate percentile
            # → how many to replace (floor), but at least 1
            num_to_replace = max(1, int(percentile * total))
            
            # split from the right that many times, then join with `corrupted`
            parts = string.rsplit(original, num_to_replace)
            new_string = corrupted.join(parts)
            return new_string, num_to_replace
        
        pct_df = pd.DataFrame({"percentile": [0.25, 0.5, 0.75, 1.0]})
        self.df = self.df.merge(pct_df, how='cross')
        self.df['corrupt_reasoning'], self.df['num_corrupted'] = zip(*self.df.apply(corrupt_digit, axis = 1))
        self.df = self.df.dropna(how = 'any').drop_duplicates(ignore_index=True)
    
    def eval(self) -> None:
        """Evaluate the model on the dataset and save results.
        
        This method:
        1. Generates model responses
        2. Computes correctness scores
        3. Saves results to disk
        """
        try:
            # The suffix after corrupt reasoning is to force the model to answer with </think>
            prompts = self.df['problem'] + self.df['corrupt_reasoning'] + '</think>\n' 

            self.response = self.generate(prompts=prompts).rename(columns = {'response': 'post_corruption_response'})
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            # Compute correctness scores
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
            
            # Log summary statistics
            accuracy = sum(correctness) / len(correctness)
            print(f"Evaluation complete. Accuracy: {accuracy:.2%}")
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--question_ids_fname", type=str, required=True, help="stress test question ids file name")
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
    
    args = parser.parse_args()
    engine = CorruptDigit(
        **vars(args),
    )
    
    engine.corrupt()
    engine.eval()