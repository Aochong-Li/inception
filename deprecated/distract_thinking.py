import os
import pandas as pd
import sys
sys.path.append("/home/al2644/research")
from llm_engine import *
import argparse
from reward_score.math import compute_score, last_boxed_only_string, remove_boxed
import random
from typing import Tuple
import logging
import numpy as np

BEGIN_PHRASE = "Solve this problem:"
TRANSITION_PHRASE = " Alright, let me think about this problem. "

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Reasoner_QDistractRA(OpenLMEngine):  # Fixed typo in class name
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_path: str = None,
                 granularity: int = 20,
                 num_distract_candidates: int = 5,
                 output_dir: str = './results/hendrycks_math/sample200/distract_thinking',
                 tensor_parallel_size: int = 2,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 midway: bool = False,
                 inject_user_prompt: bool = False,
                 **kwargs
                 ):
        # Initialize attributes first
        self.nick_name = nick_name
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.granularity = granularity
        self.midway = midway
        self.inject_user_prompt = inject_user_prompt

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset from pickle file
        self.load_dataset(dataset_path, num_distract_candidates)
    
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

        print(f"Start evaluating {self.nick_name} on dataset: {dataset_path}")

    def chunking(self, model_response):
        """Split thinking into chunks of approximately equal size."""
        # Step 1: merge small chunks
        thinking = model_response.split('</think>')[0]
        chunks = thinking.split('\n\n')
        masks = [len(chunk.split()) > self.granularity for chunk in chunks]
            
        merged, buffer = [], []
        for c, m in zip(chunks, masks):
            if not m:
                buffer.append(c)
            else:
                if buffer:
                    merged.append('\n\n'.join(buffer))  # Use '\n\n' to maintain paragraph structure
                    buffer.clear()
                merged.append(c)
        if buffer:
            merged.append('\n\n'.join(buffer))
        
        # Step 2: merge small chunks to big chunks
        super_chunks, current = [], None
        for c in merged:
            if len(c.split()) > self.granularity:
                if current is not None:
                    super_chunks.append(current)
                current = c
            else:
                if current is None:
                    # no big chunk yet
                    current = c
                else:
                    current += '\n\n' + c  # Use '\n\n' to maintain paragraph structure
        
        if current is not None:
            super_chunks.append(current)
        
        return super_chunks

    def load_dataset(self, dataset_path: str, num_distract_candidates: int) -> None:
        """Load dataset from pickle file.
        
        Args:
            dataset_path: Path to the pickle file containing the dataset
            num_distract_candidates: Number of rows to use as distractor candidates
        """
        try:
            self.df = pd.read_pickle(dataset_path).drop(columns=['correct'])
            self.df = self.df.rename(columns={'response': 'original_response'})

            # Sample rows to use as distractors
            self.df['think_chunks'] = self.df['original_response'].apply(self.chunking)
            self.distract_candidates = self.df[self.df['think_chunks'].str.len().between(30,45)].sample(n=num_distract_candidates, random_state=45)
            self.distract_candidates = self.distract_candidates[['problem', 'solution', 'think_chunks']] \
                                                .rename(columns={
                                                    'problem': 'problem_distractor',
                                                    'solution': 'distract_solution',
                                                    'think_chunks': 'distract_reasoning'
                                                    })

            # Remove distractor rows from the main dataframe
            self.df.drop(columns=['think_chunks'], inplace=True)
            self.df = self.df.drop(self.distract_candidates.index)
            
            # Randomize the rows of the dataframe
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            distract_ids = np.random.randint(0, len(self.distract_candidates), size=len(self.df))
            self.distract_candidates = self.distract_candidates.iloc[distract_ids]
            self.distract_candidates.index = self.df.index
            self.df = pd.concat([self.df, self.distract_candidates], axis=1)

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")
    
    def generate_distract_reasoning(self, row):
        """Generate distract reasoning for the dataset.
        
        Args:
            row: DataFrame row containing percentile and other data
        
        Returns:
            Tuple containing distract reasoning, actual fraction used, and the distractor problem
        """
        percentile = row['percentile']
        
        # Sample a distractor candidate        
        chunks = row['distract_reasoning']
        num_chunks = max(1, int(len(chunks) * percentile))
        
        selected_chunks = chunks[:num_chunks]
        distract_reasoning = '\n\n'.join(selected_chunks)
        num_distract_tokens = len(self.tokenizer.encode(distract_reasoning))
        
        return distract_reasoning, num_distract_tokens

    def add_original_reasoning_prefix (self):
        pct_df = pd.DataFrame({"percentile": [0.10, 0.25, 0.5, 0.75, 0.9]})
        self.df = self.df.merge(pct_df, how='cross')

        self.df["original_reasoning_prefix"] = self.df["original_response"].apply(self.chunking)

        def process_prefix (row):
            percentile = row["percentile"]
            chunks = row["original_reasoning_prefix"]

            num_chunks = max(1, int(len(chunks) * percentile))
            selected_chunks = chunks[:num_chunks]
            prefix = '\n\n'.join(selected_chunks)
            num_prefix_tokens = len(self.tokenizer.encode(prefix))

            return prefix, num_prefix_tokens
        
        self.df["original_reasoning_prefix"], self.df["num_original_reasoning_prefix_tokens"] = zip(*self.df.apply(process_prefix, axis=1))
        
        self.df = self.df.drop(columns=["percentile"]).drop_duplicates()

    def distract(self):
        """Distract the dataset by adding distractor reasoning to problems."""
        # Define percentiles to test    
        pct_df = pd.DataFrame({"percentile": [0.10, 0.25, 0.5, 0.75, 0.9]})
        # Use cross merge to create combinations of problems and percentiles
        self.df = self.df.merge(pct_df, how='cross')
        
        # Apply distraction
        results = self.df.apply(self.generate_distract_reasoning, axis=1)
        self.df['distract_reasoning'] = [r[0] for r in results]
        self.df['num_distract_tokens'] = [r[1] for r in results]
    
        self.df = self.df.drop(columns=['percentile']).drop_duplicates()

        if self.midway:
            self.add_original_reasoning_prefix()
            self.df["distract_reasoning"] = self.df["original_reasoning_prefix"] + TRANSITION_PHRASE + self.df["distract_reasoning"]
            
    def eval(self) -> None:
        """Evaluate the model on the dataset and save results.
        
        Args:
            continue_thinking: Whether to allow the model to continue thinking
        """
        try:
            # Generate model responses
            if self.inject_user_prompt:
                probe = [{"role": "user", "content": "thisisprobe"}]
                _, suffix = self.tokenizer.apply_chat_template(probe, 
                                                                    tokenize=False, 
                                                                    add_generation_prompt=True
                                                                    ).split("thisisprobe")
                prompts = self.df["problem"].str.replace(suffix, "") + TRANSITION_PHRASE + self.df['distract_reasoning'] + suffix
            else:
                prompts = self.df['problem'] + self.df['distract_reasoning']
            
            self.response = self.generate(prompts=prompts).rename(columns = {'response': 'post_distraction_response'})
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            # Compute correctness scores for original problem
            original_correctness = []
            # Compute correctness scores for distractor problem
            distractor_correctness = []
            
            for _, row in self.df.iterrows():
                solution = row['post_distraction_response']
                original_ground_truth = row['solution']
                distractor_ground_truth = row['distract_solution']
                
                # Clean solution if it contains thinking steps
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                # Score against original problem
                original_score = compute_score(solution, original_ground_truth)
                original_correctness.append(original_score)
                
                # Score against distractor problem
                distractor_score = compute_score(solution, distractor_ground_truth)
                distractor_correctness.append(distractor_score)
            
            # Combine results
            self.df['original_correct'] = original_correctness
            self.df['distractor_correct'] = distractor_correctness
            
            # Save results
            fname = f"{self.nick_name}{'_midway' if self.midway else ''}{'_inject_user_prompt' if self.inject_user_prompt else ''}.pickle"
            output_path = os.path.join(self.output_dir, fname)
            self.df.to_pickle(output_path)
            
            # Log summary statistics
            original_accuracy = sum(original_correctness) / len(original_correctness)
            distractor_accuracy = sum(distractor_correctness) / len(distractor_correctness)
            print(f"Evaluation complete.")
            print(f"Original problem accuracy: {original_accuracy:.2%}")
            print(f"Distractor problem accuracy: {distractor_accuracy:.2%}")
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--dataset_path", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--output_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
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
    parser.add_argument("--num_distract_candidates", type=int, default=5,
                        help="Number of problems to use as distractors")
    parser.add_argument("--granularity", type=int, default=20,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--midway", type=str2bool, default=False,
                        help="Whether to add original reasoning prefix midway")
    parser.add_argument("--inject_user_prompt", type=str2bool, default=False,
                        help="Whether to inject user prompt")
    
    args = parser.parse_args()
    engine = Reasoner_QDistractRA(
        **vars(args),
    )

    engine.distract() 
    engine.eval()