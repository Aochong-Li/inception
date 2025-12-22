import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from core.llm_engine import *
from core.openai_engine import *

from reward_score.math500 import math_if_boxed, last_boxed_only_string, remove_boxed
from utils.chunk_r import equal_chunk
from utils.corrupt_num import *
from more_itertools import chunked

class TeacherGuide(OpenLMEngine):
    def __init__(
        self,
        model_name: str,
        nick_name: str,
        tokenizer_name: str,
        results_dir: str,
        sample_size: int,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        num_responses_per_problem: int = 1,
        max_num_batched_tokens: int = 8192,
        mini_batch_size: int = None,
        granularity: int = 30,
        overwrite: bool = False,
        client_name: str = "",
        **kwargs,
    ):
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.sample_size = sample_size
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_responses_per_problem = num_responses_per_problem
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.sample_size = sample_size
        self.granularity = granularity
        self.overwrite = overwrite
        self.client_name = client_name

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "teacher_guide")
        os.makedirs(self.output_dir, exist_ok=True)
        
        out_pickle = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Stress test (Teacher Guide) already exists: {self.nick_name}")
            exit()
        
        cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_position_embeddings = cfg.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if self.client_name == '':    
            # Initialize model config
            config = ModelConfig(
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                n=self.num_responses_per_problem,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_position_embeddings
            )

            # Initialize parent class
            # super().__init__(config=config)
        
        self.load_dataset()
    
        print(f"Start stress testing: {self.nick_name} on teacher guided reasoning")
    
    @staticmethod
    def convert_model_is_correct(label) -> float:
        if label == True:
            return 1.0
        elif label == False:
            return 0.0
        elif type(label) == float:
            return label
        else:
            raise ValueError(f"Invalid label: {label}")

    def load_dataset(self) -> None:
        self.df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle"))
        self.df["model_is_correct"] = self.df["model_is_correct"].apply(self.convert_model_is_correct)
        
        # Keep only problems that the model cannot solve
        stats = self.df.groupby("problem").agg({"model_is_correct": "sum"}).reset_index().rename(columns={"model_is_correct": "solve_n"})
        self.df = self.df.merge(stats, on="problem")
        self.df = self.df[self.df["solve_n"] == 0]
        self.df = self.df[["problem", "solution", "source"]]

        # Load teacher responses
        self.qwq_df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", "QwQ-32B.pickle"))
        self.qwq_df["model_is_correct"] = self.qwq_df["model_is_correct"].apply(self.convert_model_is_correct)
        self.qwq_df["teacher"] = "QwQ-32B"
        self.qwen3_df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", "Qwen3-235B-A22B.pickle"))
        self.qwen3_df["model_is_correct"] = self.qwen3_df["model_is_correct"].apply(self.convert_model_is_correct)
        self.qwen3_df["teacher"] = "Qwen3-235B-A22B"
        self.r1_df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", "DeepSeek-R1.pickle"))
        self.r1_df["model_is_correct"] = self.r1_df["model_is_correct"].apply(self.convert_model_is_correct)
        self.r1_df["teacher"] = "DeepSeek-R1"

        self.teacher_df = pd.concat([self.qwq_df, self.qwen3_df, self.r1_df], ignore_index=True) \
            [["problem", "response", "teacher", "model_is_correct"]]
        self.teacher_df = self.teacher_df[self.teacher_df["model_is_correct"] == 1.0]

        # Remove the potential suffix from the problem
        suffix = "Please reason and put the final answer inside \\\\boxed{} tag."
        self.df["problem"] = self.df["problem"].apply(lambda x: x.replace(suffix, "").strip())
        self.teacher_df["problem"] = self.teacher_df["problem"].apply(lambda x: x.replace(suffix, "").strip())
        self.teacher_df["response"] = self.teacher_df["response"].apply(lambda x: x.replace("<think>\n", "").strip())

        # Keep problems that at least one teacher can solve
        def count_teacher_solutions(row):
            problem = row["problem"]
            num_teacher_solutions = self.teacher_df[self.teacher_df["problem"] == problem]["teacher"].nunique()
            return num_teacher_solutions
        
        self.df["num_teacher_solutions"] = self.df.apply(count_teacher_solutions, axis = 1)
        self.df = self.df[self.df["num_teacher_solutions"] > 0]

        if self.sample_size is not None:
            self.df = self.df.sort_values(by = "num_teacher_solutions", ascending = False).drop_duplicates(subset = "problem")
            self.df = self.df[:self.sample_size].reset_index(drop=True)
        
        self.teacher_df = self.teacher_df[self.teacher_df["problem"].isin(self.df["problem"])] \
            .drop_duplicates(subset = ["problem", "teacher"]).drop(columns = ["model_is_correct"]) \
            .rename(columns = {"response": "teacher_response"}).reset_index(drop=True)
        
        self.df = self.df.merge(self.teacher_df, on = "problem")

        print(f"Loaded {len(self.df)} data with teacher guidance")
        
    def guide(self, row):
        ratio = row["ratio"]
        teacher_response = row["teacher_response"]

        if "</think>" in teacher_response:
            teacher_reasoning = teacher_response.split("</think>")[0]
        else:
            teacher_reasoning = teacher_response

        teacher_chunks = equal_chunk(teacher_reasoning, self.granularity)
        num_chunks = int(len(teacher_chunks) * ratio)
        teacher_chunks = teacher_chunks[:num_chunks]
        
        return "".join(teacher_chunks)

    def guide_reasoning(self):
        windows = [0.1, 0.2, 0.4, 0.6, 0.8]
        self.df["ratio"] = len(self.df) * [windows]
        self.df = self.df.explode("ratio", ignore_index = True)
        self.df["teacher_reasoning"] = self.df.apply(self.guide, axis = 1)
        
    def local_eval(self) -> None:
        self.responses = []
        self.mini_batch_size = self.df.shape[0] if self.mini_batch_size is None else self.mini_batch_size
        for batch in chunked(list(self.df["prompt"]), self.mini_batch_size):
            new_sampling_params = [
                {
                    "max_tokens": min(self.max_tokens, self.max_position_embeddings - len(self.tokenizer.encode(prompt)) - 1)
                }
                for prompt in batch
            ]
            out = self.generate(prompts=batch, new_sampling_params=new_sampling_params)
            self.responses.append(out)
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'student_response'})
        self.df = self.df.loc[np.repeat(self.df.index, self.num_responses_per_problem)].reset_index(drop=True)
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def api_eval(self) -> None:
        os.makedirs(self.output_dir + f"/api", exist_ok=True)

        engine = OpenAI_Engine(
            input_df=self.df,
            nick_name=f"distract_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/reasoning",
            cache_filepath=os.path.join(self.output_dir, f"api/{self.nick_name}_api_responses.pickle"),
            model=self.model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode="completions"
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs(overwrite=self.overwrite)
        self.response = self.response.explode(['response']).set_index('idx').rename(columns={'response': 'post_distraction_response'})
        self.df = self.df.merge(self.response, left_index=True, right_index=True)

    def apply_chat_template(self, row) -> str:
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "HANDLE"}],
            tokenize=False,
            add_generation_prompt=True
        ).split("HANDLE")

        if '<think>' not in template_suffix:
            if "openthinker" in self.model_name.lower():
                template_suffix = template_suffix + "<think> "
            else:
                template_suffix = template_suffix + "<think>\n"

        problem = row["problem"]
        teacher_reasoning = row["teacher_reasoning"]
        prompt = template_prefix + problem + template_suffix + teacher_reasoning

        return prompt

    def eval(self) -> None:
        self.df["prompt"] = self.df.apply(self.apply_chat_template, axis = 1)
        self.df["teacher_reasoning_token_counts"] = self.df["teacher_reasoning"].apply(lambda x: len(self.tokenizer.encode(x)))
        
        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        # # HACK
        # if os.path.exists(output_path):
        #     existing_df = pd.read_pickle(output_path)
        #     existing_df = existing_df[existing_df["prompt"].isin(self.df["prompt"])].reset_index(drop=True)
        #     self.df = self.df[~self.df["prompt"].isin(existing_df["prompt"])].reset_index(drop=True)
        
        # if len(self.df) == 0:
        #     print(f"No new problems to evaluate for {self.nick_name}")
        #     return
        # # END OF HACK
        
        if self.client_name == '':
            self.local_eval()
        else:
            self.api_eval()

        self.result_df = self.df.copy()
        self.result_df['pred'] = self.result_df['student_response'].apply(lambda x: x.split('</think>')[-1].strip() if '</think>' in x else x)
        self.result_df['ground_truth'] = self.result_df['solution']
        self.result_df['if_boxed'] = self.result_df['pred'].apply(math_if_boxed)

        self.result_df.to_pickle(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of problems to sample for the stress test")
    
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter")
    parser.add_argument("--num_responses_per_problem", type=int, default=8,
                        help="Number of teacher responses to use")
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768,
                        help="Maximum number of tokens in a batch")
    
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini batch size for generation")
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")

    args = parser.parse_args()
    
    engine = TeacherGuide(
        **vars(args),
    )
    engine.guide_reasoning()
    engine.eval()