import os
import pandas as pd
from transformers import AutoModelForCausalLM
from core.llm_engine import *
from core.openai_engine import *

import argparse
from datasets import load_dataset, load_from_disk, Dataset

import numpy as np
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True 
    elif v.lower() == "false":
        return False

class BenchmarkEval(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_name_or_path: str = 'aochongoliverli/wmdp_inquiries_300',
                 subset_name: str = None,
                 split_name: str = 'train',
                 sample_size: int = None,
                 output_dir: str = './results/wmdp_inquiries/benchmark',
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 system_prompt: str = None, 
                 max_tokens: int = 16384,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 sample_k: int = 1,
                 max_num_batched_tokens: int = 8192,
                 overwrite: bool = False,
                 client_name: str = '',
                 filename_suffix: str = ''
                 ):

        # Initialize attributes first
        self.model_name = model_name
        self.nick_name = nick_name
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_k = sample_k
        self.overwrite = overwrite
        self.max_num_batched_tokens = max_num_batched_tokens
        self.client_name = client_name
        self.filename_suffix = filename_suffix
        self.system_prompt = system_prompt

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filepath = os.path.join(
            self.output_dir, f"{self.nick_name}{self.filename_suffix if self.filename_suffix else ''}.pickle"
            )
        if not self.overwrite:
            if os.path.exists(self.output_filepath):
                print(f"Results already exist for {self.nick_name}")
                exit()

        # Load dataset
        self.load_dataset(
            dataset_name_or_path,
            subset_name,
            split_name,
            sample_size
        )

        if self.client_name == '':
            # Run locally
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
                n = self.sample_k,
                max_num_batched_tokens=self.max_num_batched_tokens
            )
            # Download model weights if not already downloaded
            _ = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            # Initialize parent class
            super().__init__(config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Start evaluating {self.nick_name} on dataset: {dataset_name_or_path} | subset: {subset_name} | split: {split_name} | avg@{self.sample_k}")

    def load_dataset(self, dataset_name: str, subset_name: str, split_name: str, sample_size: int) -> None:
        try:
            dataset = load_from_disk(dataset_name)[split_name]
        except Exception as e:
            print(f"Error loading from disk: {e}")
            try:
                dataset = load_dataset(dataset_name, subset_name)[split_name]
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}")
        if sample_size:
            self.df = pd.DataFrame(dataset).sample(n=sample_size, random_state=45).reset_index(drop = True)
        else:
            self.df = pd.DataFrame(dataset)

    def apply_chat_template (self, question: str):
        chat_history = [
            {'role': 'user', 'content': question}
        ]
        if self.system_prompt:
            chat_history.insert(0, {'role': 'system', 'content': self.system_prompt})

        tokenized_prompt = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize = False,
            add_generation_prompt = True
        )
        
        return tokenized_prompt
    
    def eval(self) -> None:
        if self.client_name == '':
            self.local_eval()
        else:
            self.api_eval()

        self.df = self.df.loc[np.repeat(self.df.index, self.sample_k)].reset_index(drop=True)
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)
        self.df.to_pickle(self.output_filepath)

        # Post-process
        self.df['thinking'] = self.df['response'].apply(lambda x: x.split('</think>')[0].strip() if '</think>' in x else x)
        self.df.to_pickle(self.output_filepath)
        
    def api_eval(self) -> None:
        os.makedirs(self.output_dir + "/api", exist_ok=True)
        if self.system_prompt:
            self.df["prompt"] = self.df["inquiry"].apply(lambda x: x + " " + self.system_prompt)
        else:
            self.df["prompt"] = self.df["inquiry"]

        engine = OpenAI_Engine(
            input_df=self.df,
            prompt_template="{prompt}",
            developer_message="",
            template_map={"prompt": "prompt"},
            nick_name=f"benchmark_eval_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/wmdp",
            cache_filepath=self.output_dir + f"/api/{self.nick_name}_api_responses.pickle",
            model=self.model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.sample_k,
            mode="chat_completions"
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs(overwrite=self.overwrite)
        self.response = self.response.set_index('idx').explode(['response']).reset_index(drop=True)

    def local_eval(self) -> None:
        prompts = self.df['inquiry'].apply(self.apply_chat_template)
        self.response = self.generate(prompts=prompts)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--dataset_name_or_path", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--subset_name", type=str, default=None, help="Name of the dataset subset (default: None)")
    parser.add_argument("--split_name", type=str, default='test', help="Dataset split to use (default: test)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use (default: None)")
    parser.add_argument("--output_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
    parser.add_argument("--filename_suffix", type=str, default="")

    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75,
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
    parser.add_argument("--sample_k", type=int, default=1,
                        help="Sample@k parameter")

    parser.add_argument("--overwrite", type=str2bool, default=False,
                        help="Overwrite existing results")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192,
                        help="Maximum number of tokens to batch")
    parser.add_argument("--client_name", type=str, default='',
                        help="Name of the client to use")
    args = parser.parse_args()

    SYSTEM_PROMPT = None
    engine = BenchmarkEval(
        **vars(args),
        system_prompt=SYSTEM_PROMPT
    )
    engine.eval()