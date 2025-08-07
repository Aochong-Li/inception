import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from llm_engine import *
import argparse
from reward_score.math import compute_score
import logging
from typing import List, Set, Dict
import re
import random

class MeasureReasoningSoundness(OpenLMEngine):
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
                 overwrite: bool = False,
                 min_num_chunks: int = 10,
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
        self.overwrite = overwrite
        self.min_num_chunks = min_num_chunks

        self.output_dir = os.path.join(self.results_dir, f"reasoning_soundness")
        os.makedirs(self.output_dir, exist_ok=True)

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
            top_k=self.top_k
        )

        super().__init__(config=config)
        # Test Code
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_dataset(self):
        self.df = pd.read_pickle(os.path.join(self.results_dir, f"annotated_reasoning/{self.nick_name}.pickle"))
        self.df = self.df[self.df["chunks"].str.len() >= self.min_num_chunks]
        self.df = self.df.dropna(subset=["1st_answer_chunk_index"]).reset_index(drop=True)
        
    def skip_steps(self):
        # Load dataset again to avoid overwriting the original dataset
        self.load_dataset()

        # Test 3 cases:
        # 1. only answer chunk (tail)
        # 2. only start chunk + answer chunk (head + tail)
        # 3. only derivation chunks + answer chunk (no head)
        strategies = ["answer_only", "headntail", "no_head"]
        if not self.overwrite and os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}_skip_steps.pickle")):
            print("Skip Steps Test Already Done ...")
            return

        def build_prompt(row, strategy: str):
            problem = row["problem"]
            answer_index = int(row["1st_answer_chunk_index"])
            chunks = row["chunks"]
            start_chunk = chunks[0]

            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            if strategy == "answer_only":
                prompt += chunks[answer_index]
            elif strategy == "headntail":
                prompt += start_chunk + " " + chunks[answer_index]
            elif strategy == "no_head":
                prompt += "".join(chunks[1: answer_index + 1])
            
            return prompt
        
        for strategy in strategies:
            prompt_col = f"skip_steps_prompt_{strategy}"            
            self.df[prompt_col] = self.df.apply(build_prompt, axis=1, args=(strategy,))
            self.response = self.generate(prompts=self.df[prompt_col])
            self.df[f"skip_steps_response_{strategy}"] = list(self.response["response"])

            self.eval(f"skip_steps_response_{strategy}")
        
        self.save("skip_steps")
    
    def disturb_reasoning(self):
        # Load dataset again to avoid overwriting the original dataset
        self.load_dataset()

        def build_prompt(row):
            problem = row["problem"]
            answer_index = int(row["1st_answer_chunk_index"])
            chunks = row["chunks"]
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            start_chunk = chunks[0]
            derivation_chunks = chunks[1: answer_index]
            random.shuffle(derivation_chunks)
            prompt += start_chunk + " " + "".join(derivation_chunks) + " " + chunks[answer_index]
            
            return prompt
        
        if not self.overwrite and os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}_disturb_reasoning.pickle")):
            print("Disturb Reasoning Test Already Done ...")
            return 
        
        self.df["disturb_reasoning_prompt"] = self.df.apply(build_prompt, axis=1)
        self.response = self.generate(prompts=self.df["disturb_reasoning_prompt"])
        self.df["disturb_reasoning_response"] = list(self.response["response"])
        
        self.eval("disturb_reasoning_response")
        self.save("disturb_reasoning")

    def introduce_noise(self):
        self.load_dataset()

        def corrupt_numbers(input_txt: str):
            def extract_number(text: str) -> Set[str]:
                _NUMBER_RE = re.compile(
                    r'(?:\d+\.\d+|\d+|\.\d+)'        # 123.456   | 123   | .456
                )
                return {m.group(0) for m in _NUMBER_RE.finditer(text)}
            
            def replace_number(text: str, replacement: Dict[str, str]) -> str:
                _NUMBER_RE = re.compile(
                    r'(?<!\w)'          # not preceded by a letter/number/underscore
                    r'(?:\d+\.\d+|'     # 123.456
                    r'\d+|'             # 123
                    r'\.\d+)'           # .456
                    r'(?!\w)'           # not followed by a letter/number/underscore
                )
                return _NUMBER_RE.sub(
                    lambda m: replacement.get(m.group(0), m.group(0)),
                    text
                )
            
            def perturb_number(num: str, rng: random.Random) -> str:
                sign = ''
                if num[0] in '+-':            # keep explicit sign, if any
                    sign, num = num[0], num[1:]

                def _strip_leading_zero(s: str) -> str:
                    while len(s) > 1 and s[0] == '0' and s[1].isdigit():
                        s = s[1:]
                    return s

                # ---------------------- floats ------------------------------------------
                if '.' in num:
                    digits = num.replace('.', '')
                    if rng.random() < 0.5 or len(digits) < 2:
                        new_core = digits

                    else:
                        new_pos = rng.randint(0, len(digits))
                        new_core = digits[:new_pos] + '.' + digits[new_pos:]
                    new_core = _strip_leading_zero(new_core)

                # ---------------------- integers ----------------------------------------
                elif len(num) == 1:
                    # single digit → substitute
                    new_digit = rng.choice([d for d in '0123456789' if d != num])
                    new_core = new_digit
                else:
                    # multi-digit int
                    if rng.random() < .5:          # --- randomize the order of digits
                        lst = list(num)
                        rng.shuffle(lst)
                        new_core = ''.join(lst)
                    else:                           # ---- duplicate OR delete
                        i = rng.randint(0, len(num) - 1)
                        if rng.random() < 0.5 and len(num) > 1:   # delete
                            new_core = num[:i] + num[i+1:]
                        else:                                     # duplicate
                            new_core = num[:i+1] + num[i] + num[i+1:]
                    new_core = _strip_leading_zero(new_core)

                # Guarantee change; if not, recurse once (extremely rare)
                if new_core == num:
                    return perturb_number(num, rng)
                return sign + new_core 
            
            numbers = extract_number(input_txt)
            replacement = {number: perturb_number(number, random.Random()) for number in list(numbers)}
            
            return replace_number(input_txt, replacement)
        
        def build_prompt(row):
            problem = row["problem"]
            answer_index = int(row["1st_answer_chunk_index"])
            chunks = row["chunks"]
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)
            
            start_chunk = chunks[0]
            derivation_steps = "".join(chunks[1: answer_index])
            corruption_derivation = corrupt_numbers(derivation_steps)
            prompt += start_chunk + " " + corruption_derivation + " " + chunks[answer_index]
            
            return prompt
        
        if not self.overwrite and os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}_error_reasoning.pickle")):
            print("Error Reasoning Test Already Done ...")
            return
        
        self.df["error_reasoning_prompt"] = self.df.apply(build_prompt, axis=1)
        self.response = self.generate(prompts=self.df["error_reasoning_prompt"])
        self.df["error_reasoning_response"] = list(self.response["response"])
        
        self.eval("error_reasoning_response")
        self.save("error_reasoning")
    
    def eval(self, col: str):
        testname = col.replace("_response", "")
        try:
            correctness = []
            for _, row in self.df.iterrows():
                solution = row[col]
                ground_truth = row['solution']
                
                # Clean solution if it contains thinking steps
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                score = compute_score(solution, ground_truth)
                correctness.append(score)
            
            # Combine results
            self.df[f'{testname}_correct'] = correctness

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save(self, testname: str):
        output_path = os.path.join(self.output_dir, f"{self.nick_name}_{testname}.pickle")
        self.df.to_pickle(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--nick_name", type=str, default="Qwen3-14B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--results_dir", type=str, default="./outputs/benchmark")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_num_chunks", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    engine = MeasureReasoningSoundness(**vars(args))
    
    engine.skip_steps()
    engine.disturb_reasoning()
    engine.introduce_noise()