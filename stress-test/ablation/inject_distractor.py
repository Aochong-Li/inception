import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
import json
import random
import argparse

from core.llm_engine import *

from reward_score.math import compute_score as math_compute_score
from reward_score.countdown import compute_score as countdown_compute_score

from utils.chunk_r import equal_chunk

class InjectDistractor(OpenLMEngine):
    def __init__(self,
                 task: str,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 results_dir: str,
                 question_ids_fname: str,
                 num_distract_candidates: int,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 granularity: int = 20,
                 unit: float = 0.25,
                 overwrite: bool = False,
                 **kwargs
                 ):
        # Initialize attributes first
        self.task = task
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.question_ids_fname = question_ids_fname
        self.num_distract_candidates = num_distract_candidates
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.granularity = granularity
        self.unit = unit
        self.overwrite = overwrite

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "inject_distractor")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}.pickle")) and not overwrite:
            print(f"Stress test (Inject Distractor) already exists: {self.nick_name}")
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
            top_k=self.top_k
        )

        # Initialize parent class
        super().__init__(config=config)

        print(f"Start stress testing: {self.nick_name} on distractor injection")

    def load_dataset(self) -> None:
        """Load dataset from pickle file.
        
        Args:
            dataset_path: Path to the pickle file containing the dataset
            num_distract_candidates: Number of rows to use as distractor candidates
        """
        self.dataset_path = os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle")
        self.df = pd.read_pickle(self.dataset_path)
        self.df = self.df[(self.df["correct"] == 1.) & (self.df["response"].str.contains("</think>"))].reset_index(drop = True)
        
        if self.question_ids_fname is not None:
            question_ids = json.load(open(os.path.join(self.results_dir, self.question_ids_fname)))
        else:
            question_ids = None

        try:
            if self.task == "math":
                self.compute_score = math_compute_score
                raise NotImplementedError("Math task is currently not supported yet")
            elif self.task == "countdown":
                self.compute_score = countdown_compute_score
                self.df = self.df[self.df['problem'].isin(question_ids)].reset_index(drop = True) if question_ids is not None else self.df
                self.df = self.df.drop(columns = ['correct']).rename(columns = {'response': 'original_response'})
            self.select_distractor()
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")
    
    def select_distractor(self) -> None:
        self.df["reasoning_chunks"] = self.df["original_response"] \
            .apply(lambda x: x.split("</think>")[0]) \
            .apply(lambda x: equal_chunk(x, self.granularity))
        
        self.distractors = self.df[self.df['reasoning_chunks'].str.len().between(15, 50)]
        self.distractors = self.distractors.sample(n=self.num_distract_candidates, random_state=42)
        
        self.df = self.df[~self.df.index.isin(self.distractors.index)]
        self.df, self.distractors = self.df.reset_index(drop = True), self.distractors.reset_index(drop = True)

    def generate_distract_reasoning_countdown(self, row):
        original_ratio = row["original_ratio"]
        distractor_ratio = row["distractor_ratio"]

        original_chunks = row["reasoning_chunks"][: int(len(row["reasoning_chunks"]) * original_ratio)]
        original_reasoning = "".join(original_chunks)

        distractor = self.distractors.sample(n = 1).iloc[0]
        distractor_nums, distractor_target, distractor_chunks = distractor["nums"], distractor["target"], distractor["reasoning_chunks"]
            
        distractor_chunks = distractor_chunks[:int(len(distractor_chunks) * distractor_ratio)]
        distractor_reasoning = "".join(distractor_chunks).replace("<think>", "").strip("\n")

        if "<think>" not in original_reasoning:
            original_reasoning = "<think>\n" + original_reasoning
        
        if original_ratio > 0.0:
            original_reasoning = original_reasoning + "Let me think. "

        return distractor_nums, distractor_target, original_reasoning + distractor_reasoning

    def distract(self):
        windows = [round(p, 2) for p in np.arange(0, 1, self.unit)]
        distractor_windows = [round(p, 2) for p in np.arange(self.unit, 1 + 1e-5, self.unit)]

        self.df["original_ratio"] = len(self.df) * [windows]
        self.df = self.df.explode("original_ratio", ignore_index = True)
        self.df["distractor_ratio"] = len(self.df) * [distractor_windows]
        self.df = self.df.explode("distractor_ratio", ignore_index = True)

        if self.task == "countdown":
            distract_func = self.generate_distract_reasoning_countdown
        elif self.task == "math":
            raise NotImplementedError("Math task is currently not supported yet")
        
        self.df["distract_nums"], self.df["distract_target"], self.df["reasoning_w_distractor"] = zip(*self.df.apply(distract_func, axis=1))
        
    def eval(self) -> None:
        try:
            template_prefix, template_suffix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "HANDLE"}],
                tokenize=False,
                add_generation_prompt=True
            ).split("HANDLE")
            template_suffix = template_suffix.removesuffix("<think>\n")
            
            self.df["prompt"] = template_prefix + self.df['problem'] + template_suffix + self.df['reasoning_w_distractor']
            
            self.response = self.generate(prompts=list(self.df['prompt'])).rename(columns = {'response': 'post_distraction_response'})
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            original_correctness = []
            distractor_correctness = []
            
            for _, row in self.df.iterrows():
                solution = row['post_distraction_response']
                
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                if self.task == "math":
                    original_ground_truth = row['solution']
                    distractor_ground_truth = row['distract_solution']
                
                    original_score = self.compute_score(solution, original_ground_truth)
                    distractor_score = self.compute_score(solution, distractor_ground_truth)
                elif self.task == "countdown":
                    original_score = self.compute_score(solution, row['nums'], row['target'])
                    distractor_score = self.compute_score(solution, row['distract_nums'], row['distract_target'])
                    
                original_correctness.append(original_score)
                distractor_correctness.append(distractor_score)
            
            self.df['original_correct'] = original_correctness
            self.df['distractor_correct'] = distractor_correctness
            
            output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
            self.df.to_pickle(output_path)
            
            original_accuracy = sum(original_correctness) / len(original_correctness)
            distractor_accuracy = sum(distractor_correctness) / len(distractor_correctness)
            
            print(f"Original problem accuracy: {original_accuracy:.2%}")
            print(f"Distractor problem accuracy: {distractor_accuracy:.2%}")
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
    parser.add_argument("--question_ids_fname", type=str, default="stress_test_problems.json",
                        help="Name of the file containing the question ids")

    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling parameter")
    
    parser.add_argument("--unit", type=float, default=0.25,
                        help="Unit of the thinking chunks")                    
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--num_distract_candidates", type=int, default=20,
                        help="Number of problems to use as distractors")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")

    args = parser.parse_args()
    
    engine = InjectDistractor(
        **vars(args),
    )
    engine.distract()
    engine.eval()