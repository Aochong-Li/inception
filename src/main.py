"""
Main pipeline orchestration for iterative refusal-aware inception.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import fasttext
from datasets import load_dataset

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.llm_engine import OpenLMEngine, ModelConfig
from core.openai_engine import OpenAI_Engine

from src.refusal_model import classify_reasoning_trace, chop_at_refusal

#### This is for handling numpy version issues for fasttext
if not hasattr(np, '_fasttext_original_array'):
    np._fasttext_original_array = np.array

def simple_fasttext_patch(obj, *args, **kwargs):
    if kwargs.get('copy') is False:
        kwargs['copy'] = True
        
    return np._fasttext_original_array(obj, *args, **kwargs)

np.array = simple_fasttext_patch
####

ARCHITECT_MODEL_CHAT_TEMPLATE = {
    "open-thoughts/OpenThinker3-7B": '''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{inquiry}<|im_end|>\n<|im_start|>assistant\n<think> {reasoning}'''
}

TARGET_MODEL_CHAT_TEMPLATE = {
    "deepseek-ai/DeepSeek-R1-0528": '''<｜begin▁of▁sentence｜><｜User｜>{inquiry}<｜Assistant｜><think>\n{reasoning}''',
    "deepseek-ai/DeepSeek-V3.2": '''<｜begin▁of▁sentence｜><｜User｜>{inquiry}<｜Assistant｜><think>{reasoning}''',
    "Qwen/Qwen3-235B-A22B-Thinking-2507": '''<|im_start|>user\n{inquiry}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}''',
    "Qwen/Qwen3-Next-80B-A3B-Thinking": '''<|im_start|>user\n{inquiry}<|im_end|>\nassistant\n<think>\n{reasoning}''',
    "moonshotai/Kimi-K2-Thinking": "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>{inquiry}<|im_end|><|im_assistant|>assistant<|im_middle|><think> {reasoning}",
    "zai-org/GLM-4.6": "[gMASK]<sop><|user|>\n{inquiry}\n<think>{reasoning}"
}

class InceptionEngine:
    def __init__(
        self,
        target_model_name: str,
        target_nick_name: str,
        architect_model_name: str,
        dataset_name: str,
        split_name: str,
        results_dir: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_tokens: int = 32768,
        architect_initial_max_tokens: int = 128,
        architect_reiterate_max_tokens: int = 64,
        max_iterations: int = 3,
        max_num_batched_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        mini_batch_size: int = None,
        sample_size: int = None,
        min_reasoning_tokens: int = 0,
        overwrite: bool = False,
        client_name: str = "",
        **kwargs,
    ):
        self.target_model_name = target_model_name
        self.architect_model_name = architect_model_name
        self.target_nick_name = target_nick_name

        self.dataset_name = dataset_name
        self.split_name = split_name
        self.results_dir = results_dir

        self.architect_initial_max_tokens = architect_initial_max_tokens
        self.architect_reiterate_max_tokens = architect_reiterate_max_tokens
        self.max_iterations = max_iterations

        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.sample_size = sample_size
        self.min_reasoning_tokens = min_reasoning_tokens
        self.overwrite = overwrite
        self.client_name = client_name

        self.output_dir = os.path.join(self.results_dir, f"max_iterations_{self.max_iterations}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        out_pickle = os.path.join(self.output_dir, f"{self.target_nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Inception results for {self.target_nick_name} already exists")
            exit()

        self.architect_tokenizer = AutoTokenizer.from_pretrained(self.architect_model_name)
        self.architect_engine = OpenLMEngine(
            ModelConfig(
                model_name=self.architect_model_name,
                tokenizer_name=self.architect_model_name,
                max_tokens=self.architect_initial_max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k

            )
        )
        os.makedirs(self.output_dir + f"/api", exist_ok=True)
        self.target_engine = OpenAI_Engine(input_df=pd.DataFrame())

        self.architect_chat_template = ARCHITECT_MODEL_CHAT_TEMPLATE[self.architect_model_name]
        self.target_chat_template = TARGET_MODEL_CHAT_TEMPLATE[self.target_model_name]
        self.refusal_classifier = fasttext.load_model(os.path.join(os.path.dirname(__file__), "../fasttext_models", "refusal_model.bin"))
        
        self.load_dataset()
    
    def load_dataset(self) -> None:
        try:
            dataset = load_dataset(self.dataset_name)[self.split_name]
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset_name} split {self.split_name} from Hugging Face: {e}")

        if self.sample_size is not None:
            self.df = pd.DataFrame(dataset).sample(n=self.sample_size, random_state=42).reset_index(drop = True)
        else:
            self.df = pd.DataFrame(dataset)
        
        self.df['reasoning'] = ""

    def run(self):
        self.result_df = pd.DataFrame()
        for i in range(self.max_iterations):
            print(f"Iteration: {i+1}/{self.max_iterations}")
            
            self.architect_engine_cont(i)
            response = self.target_engine_cont(i)
            if i < self.max_iterations - 1:
                response = self.remove_refusal(response)

            self.concatenate_reasoning(response, i)
        
        self.result_df = pd.concat([self.result_df, self.df], ignore_index=True)

        self.save_results()

    def architect_engine_cont(self, iteration_idx: int) -> None:
        self.df['prompt'] = self.df.apply(lambda x: self.architect_chat_template.format(inquiry=x['inquiry'], reasoning=x['reasoning']), axis=1)
        max_tokens = self.architect_initial_max_tokens if iteration_idx == 0 else self.architect_reiterate_max_tokens
        sampling_overrides = len(self.df) * [{'max_tokens': max_tokens}]
        
        response = self.architect_engine.generate(prompts=self.df['prompt'], sampling_overrides=sampling_overrides)
        response['response'] = response['response'].apply(lambda x: x.split("</think>")[1].rstrip() if "</think>" in x else x)

        assert len(response) == len(self.df), "response and dataframe must have the same length"
        response.index = self.df.index
        self.df[f'architect_iteration_{iteration_idx}'] = response['response']
        self.df['reasoning'] = self.df['reasoning'] + self.df[f'architect_iteration_{iteration_idx}']
        self.df['reasoning'] = self.df['reasoning'].apply(lambda x: x.strip())

    def target_engine_cont(self, iteration_idx: int) -> pd.DataFrame:
        nick_name = f"{self.target_nick_name}_iteration_{iteration_idx}"
        self.df['prompt'] = self.df.apply(lambda x: self.target_chat_template.format(inquiry=x['inquiry'], reasoning=x['reasoning']), axis=1)

        self.target_engine = OpenAI_Engine(
            input_df=self.df,
            nick_name= nick_name,
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/wmdp",
            cache_filepath=os.path.join(self.output_dir, "api", f"{nick_name}.pickle"),
            model=self.target_model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode="completions"
        )
        
        self.target_engine.run_model(self.overwrite, num_workers=50)
        response = self.target_engine.retrieve_outputs()

        response["response"] = response["response"].fillna('')
        response = response.explode(['response']).set_index('idx')
        response['response'] = response['response'].apply(lambda x: x.split("</think>")[0].rstrip() if "</think>" in x else x)

        return response[['response']]
    
    def remove_refusal(self, response: pd.DataFrame) -> pd.DataFrame:
        def remove(row: pd.Series) -> pd.Series:
            response = row["response"]
            first_refusal_idx = classify_reasoning_trace(response, self.refusal_classifier, refusal_threshold=0.5, mode="firstonly")
            if first_refusal_idx is not None:
                return chop_at_refusal(response, first_refusal_idx)
            else:
                return response

        response['post_removal_response'] = response.apply(remove, axis=1)

        return response

    def _remove_last_paragraph(self, text: str) -> str:
        """Remove the last non-empty paragraph from text (split by double newlines)."""
        paragraphs = text.split('\n\n')

        for i in range(len(paragraphs) - 1, -1, -1):
            if paragraphs[i].strip() and paragraphs[i] != '\n':
                paragraphs.pop(i)
                break

        return '\n\n'.join(paragraphs)

    def concatenate_reasoning (self, response: pd.DataFrame, iteration_idx: int):
        self.df = self.df.merge(response, left_index=True, right_index=True, how='left')

        if "post_removal_response" in response.columns:
            self.df = self.df.rename(columns={'post_removal_response': f'target_iteration_{iteration_idx}'})

            self.df['reasoning'] = self.df['reasoning'] + self.df[f'target_iteration_{iteration_idx}']
            self.df['reasoning'] = self.df['reasoning'].apply(lambda x: x.strip())

            no_refusal_idx = self.df[self.df[f'target_iteration_{iteration_idx}'] == self.df['response']].index
            self.df = self.df.drop(columns=['response'])

            graduate_idx = []
            for idx in no_refusal_idx:
                reasoning = self.df.at[idx, 'reasoning']
                num_tokens = len(self.architect_tokenizer.encode(reasoning))

                if num_tokens >= self.min_reasoning_tokens:
                    graduate_idx.append(idx)
                else:
                    self.df.at[idx, 'reasoning'] = self._remove_last_paragraph(reasoning)

            if graduate_idx:
                self.result_df = pd.concat([self.result_df, self.df.loc[graduate_idx]], ignore_index=True)
                self.df = self.df.drop(graduate_idx).reset_index(drop=True)

        else:
            self.df = self.df.rename(columns={'response': f'target_iteration_{iteration_idx}'})
            self.df['reasoning'] = self.df['reasoning'] + self.df[f'target_iteration_{iteration_idx}']
            self.df['reasoning'] = self.df['reasoning'].apply(lambda x: x.strip())
 
    def save_results (self):
        self.result_df.to_pickle(os.path.join(self.output_dir, f"{self.target_nick_name}.pickle"))
        
if __name__=="__main__":
    """
    Example usage:
        python src/main.py \
            --target_model_name "deepseek-ai/DeepSeek-V3.2" \
            --target_nick_name "DeepSeek-V3.2" \
            --architect_model_name "open-thoughts/OpenThinker3-7B" \
            --dataset_name "aochongoliverli/wmdp_biochem_inquiries_800" \
            --split_name "test" \
            --results_dir ./results \
            --tensor_parallel_size 1 \
            --dtype "bfloat16" \
            --max_tokens 32768 \
            --min_reasoning_tokens 2048 \
            --architect_initial_max_tokens 256 \
            --architect_reiterate_max_tokens 64 \
            --max_iterations 3 \
            --temperature 0.6 \
            --top_p 1.0 \
            --sample_size 10 \
            --client_name "deepinfra" \
            --overwrite
    """
    parser = argparse.ArgumentParser(description="Parse Arguments for Inception")

    parser.add_argument("--target_model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--target_nick_name", type=str, required=True, help="Nickname for the target model")
    parser.add_argument("--architect_model_name", type=str, required=True, help="Name of the architect model to use")
    
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--split_name", type=str, required=True, help="Name of the split to use")
    parser.add_argument("--results_dir", type=str, default='./results/inception', 
                       help="Directory to save evaluation results")
    
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Maximum number of output tokens")
    parser.add_argument("--architect_initial_max_tokens", type=int, default=256,
                        help="Maximum tokens for architect's initial reasoning generation")
    parser.add_argument("--architect_reiterate_max_tokens", type=int, default=64,
                        help="Maximum tokens for architect's reiteration")
    parser.add_argument("--min_reasoning_tokens", type=int, default=2048,
                        help="Minimum number of tokens in reasoning chain before graduation to results")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of inception iterations")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192,
                        help="Maximum number of tokens in a batch")
    
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini batch size for generation")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of problems to sample for the stress test")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")

    args = parser.parse_args()

    engine = InceptionEngine(
        **vars(args),
    )
    engine.run()