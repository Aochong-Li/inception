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

class Reasoner_QDigitCorruptRA(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_path: str = None,
                 output_dir: str = '/share/goyal/lio/reasoning/eval/',
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
        self.corrupt_type = kwargs.get("corrupt_type", 'answer_digit')
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset from pickle file
        self.load_dataset(dataset_path)
    
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

    def load_dataset(self, dataset_path: str) -> None:
        """Load dataset from pickle file.
        
        Args:
            dataset_path: Path to the pickle file containing the dataset
        """
        try:
            self.df = pd.read_pickle(dataset_path).drop(columns = ['correct'])
            self.df = self.df.rename(columns = {'response': 'original_response'})
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")

    def corrupt(self) -> None:
        """Corrupt the dataset by modifying numerical values.
        
        This method:
        1. Locates numerical values in the dataset
        2. Modifies the values
        3. Replaces the original values with corrupted ones
        """
        if self.corrupt_type  == 'answer_digit':
            self.df['corrupt_reasoning'], self.df['original_digit'] = zip(*self.df['original_response'].apply(self.local_final_answer_digit))
            self.df = self.df.dropna(how='any').reset_index(drop = True)
            self.df['corrupt_digit'] = self.df['original_digit'].apply(self.modify)
            self.corrupt_answer_digit()
        elif self.corrupt_type  == 'midway':
            self.corrupt_in_middle()

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
            if total == 0:
                return None, None

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
        
        pct_df = pd.DataFrame({"percentile": [0.15, 0.25, 0.35, 0.5, 0.75, 0.9, 1.0]})
        self.df = self.df.merge(pct_df, how='cross')
        self.df['corrupt_reasoning'], self.df['num_corrupted'] = zip(*self.df.apply(corrupt_digit, axis = 1))
        self.df = self.df.dropna(how = 'any').drop_duplicates(ignore_index=True)
    
    def corrupt_in_middle(self, granularity: int = 20) -> None:
        """Replace the original values with corrupted ones in the middle of the thinking chain.
        
        This method:
        1. Chunks the thinking chain into segments
        2. For different percentiles, cuts the thinking in the middle
        3. Corrupts all digits by adding 1 to them
        4. Stitches the chunks back together
        
        Args:
            granularity: The minimum number of words to consider as a chunk
        """
        self.granularity = granularity
        
        def chunking(model_response):
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

        def corrupt_digit(row) -> Tuple[str, float]:
            """Corrupt digits in the thinking chunks based on percentile."""
            think_chunks = row['think_chunks']
            percentile = row['percentile']

            # Calculate how many chunks to keep based on percentile
            num_chunks = max(1, int(len(think_chunks) * percentile))
            
            # Only keep the first num_chunks
            selected_reasoning = '\n\n'.join(think_chunks[:num_chunks])
            frequent_digit, frequency = 0, 0

            for char in "0123456789":
                if selected_reasoning.count(char) > frequency:
                    frequent_digit = char
                    frequency = selected_reasoning.count(char)
            
            if frequency == 0:
                return None, None
            else:
                corrupt_digit = self.modify(frequent_digit)
                corrupted_reasoning = selected_reasoning.replace(frequent_digit, corrupt_digit)

                return corrupted_reasoning, frequency

            # Corrupt all digits in the selected chunks
            # corrupt_reasoning = []
            # for chunk in selected_chunks:
            #     # Use a more efficient approach to replace digits
            #     # Process the chunk character by character to avoid butterfly effects
            #     corrupted_chunk = ""
            #     for char in chunk:
            #         if char in '0123456789':
            #             # Replace digit with (digit+1)%10
            #             corrupted_chunk += str((int(char) + 1) % 10)
            #         else:
            #             corrupted_chunk += char
            #     corrupt_reasoning.append(corrupted_chunk)
            #             # Join the corrupted chunks
            # corrupt_reasoning = '\n\n'.join(corrupt_reasoning)
            # return corrupt_reasoning, actual_fraction
        
        # Precompute think_chunks for all rows at once
        self.df['think_chunks'] = self.df['original_response'].apply(chunking)
        # Define percentiles to test
        pct_df = pd.DataFrame({"percentile": [0.15, 0.25, 0.35, 0.5, 0.75, 0.9, 1.0]})
        
        # Use vectorized operations where possible
        self.df = self.df.merge(pct_df, how='cross')
        # Apply corruption in parallel if possible
        # Process rows sequentially instead of in parallel
        results = []
        for _, row in self.df.iterrows():
            results.append(corrupt_digit(row))
        
        self.df['corrupt_reasoning'] = [r[0] for r in results]
        self.df['frequency'] = [r[1] for r in results]

        # Clean up the dataframe
        self.df = self.df.dropna(how = 'any').drop_duplicates(ignore_index=True, subset=['corrupt_reasoning'])

    def eval(self) -> None:
        """Evaluate the model on the dataset and save results.
        
        This method:
        1. Generates model responses
        2. Computes correctness scores
        3. Saves results to disk
        """
        try:
            # Generate model responses
            prompts = self.df['problem'] + self.df['corrupt_reasoning']
            
            # if corrupt digit in answer digit, we force answer w/ </think>
            if self.corrupt_type == 'answer_digit':
                prompts += ' </think> Given' 

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
            output_path = os.path.join(self.output_dir, f"{self.nick_name}_type={self.corrupt_type}.pickle")
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
    parser.add_argument("--dataset_path", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--output_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
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
    parser.add_argument("--corrupt_type", type=str, default="answer_digit",
                        help="Type of corruption to apply")
    
    args = parser.parse_args()
    engine = Reasoner_QDigitCorruptRA(
        **vars(args),
    )
    
    engine.corrupt()
    engine.eval()