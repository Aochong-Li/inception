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
from dataclasses import load_dataset

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.llm_engine import OpenLMEngine, ModelConfig
from core.openai_engine import OpenAI_Engine

#### This is for handling numpy version issues for fasttext
_original_array = np.array
def _patched_array(obj, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None):
    try:
        return _original_array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like)
    except ValueError as e:
        if "Unable to avoid copy" in str(e) and copy is False:
            return _original_array(obj, dtype=dtype, copy=True, order=order, subok=subok, ndmin=ndmin, like=like)
        raise
np.array = _patched_array
####

ARCHITECT_MODEL_CHAT_TEMPLATE = {
    "OpenThinker3-7B-Qwen": '''<|im_start|>system<|im_end|><|im_start|>user<|im_middle|>{question}<|im_end|><|im_start|>assistant<|im_middle|><think> {reasoning}''',
}

TARGET_MODEL_CHAT_TEMPLATE = {
    "DeepSeek-R1-0528": '''<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n{reasoning}''',
    "DeepSeek-V3.2": '''<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>{reasoning}''',
    "Qwen3-235B-A22B-Thinking-2507": '''<|im_start|>user\{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}''',
    "Qwen3-Next-80B-A3B-Thinking": '''<|im_start|>user\n{question}<|im_end|>\nassistant\n<think>\n{reasoning}''',
    "moonshotai/Kimi-K2-Thinking": "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>{question}<|im_end|><|im_assistant|>assistant<|im_middle|><think> {reasoning}",
    "zai-org/GLM-4.6": "[gMASK]<sop><|user|>\n{question}\n<think>{reasoning}"
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
        architect_initial_max_tokens: int = 512,
        architect_reiterate_max_tokens: int = 128,
        max_num_batched_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        mini_batch_size: int = None,
        sample_size: int = None,
        granularity: int = 30,
        unit: float = 0.2,
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
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.architect_initial_max_tokens = architect_initial_max_tokens
        self.architect_reiterate_max_tokens = architect_reiterate_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.sample_size = sample_size
        self.granularity = granularity
        self.unit = unit
        self.overwrite = overwrite
        self.client_name = client_name

        self.output_dir = os.path.join(self.results_dir, "inception")
        os.makedirs(self.output_dir, exist_ok=True)
        
        out_pickle = os.path.join(self.output_dir, f"{self.target_nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Inception results for {self.target_nick_name} already exists")
            exit()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.architect_model_name)
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
        self.target_engine = OpenAI_Engine(input_df=pd.DataFrame(), client_name=self.client_name)
        self.refusal_classifier = fasttext.load_model(os.path.join(os.path.dirname(__file__), "../fasttext", "refusal_model.bin"))

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
        
    def run(self):
        """
        STEP 1: init reasoning from architect engine
        STEP 2: ENTER FOR LOOP
            a. find unfinished reasoning
            b. continue with target engine
            c. find and remove the first refusal chunk
            d. continue with architect engine
        STEP 3: save results
        """






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
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'incepted_response'})
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def api_eval(self) -> None:
        os.makedirs(self.output_dir + f"/api", exist_ok=True)

        engine = OpenAI_Engine(
            input_df=self.df,
            nick_name=f"inception_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/wmdp",
            cache_filepath=os.path.join(self.output_dir, f"api/{self.nick_name}_api_responses.pickle"),
            model=self.model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode="completions"
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs()
        self.response = self.response.explode(['response']).set_index('idx').rename(columns={'response': 'incepted_response'})
        self.df = self.df.merge(self.response[["incepted_response"]], left_index=True, right_index=True)
        
if __name__=="__main__":
    """
    Example usage:
        python pipeline/main.py \
        --target_model_name "deepseek-ai/DeepSeek-V3.2" \
        --architect_model_name "open-thoughts/OpenThinker3-7B" \
        --nick_name "DeepSeek-V3.2" \
        --dataset_name "aochongoliverli/wmdp_inquiries_300" \
        --split_name "test" \
        --results_dir "./results/inception" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --dtype "bfloat16" \
        --max_tokens 32768 \
        --temperature 0.6 \
        --sample_size 5 \
        --client_name "deepinfra"
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
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--unit", type=float, default=0.2,
                        help="Unit of the thinking chunks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")

    args = parser.parse_args()
    engine = InceptionEngine(
        **vars(args),
    )
    engine.run()
    engine.eval()