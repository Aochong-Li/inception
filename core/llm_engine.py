import os
import time
import logging
from typing import Union, List, Optional, Dict
from dataclasses import dataclass

import pandas as pd
import ray
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

# Allow longer max_model_len in vLLM
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["RAY_CGRAPH_get_timeout"] = "1200"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

@dataclass
class ModelConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    lora_path: Optional[str] = None
    lora_name: str = "adapter"
    max_tokens: int = 512
    max_model_len: int = 32768
    temperature: float = 0.6
    n: int = 1
    top_p: float = 0.95
    top_k: int = 32
    stop_tokens: Optional[List[str]] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    gpu_memory_utilization: float = 0.75
    dtype: str = 'bfloat16'
    max_num_batched_tokens: Optional[int] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str = 'mp'
    trust_remote_code: bool = True
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    enforce_eager: bool = True
    data_parallel_replicas: int = 1
    ray_address: Optional[str] = None


@ray.remote(num_gpus=1, num_cpus=2)
class _VLLMWorker:
    """Ray actor that wraps a vLLM model for data-parallel inference."""
    def __init__(self, model_config: Dict, sampling_params: Dict, lora_config: Optional[Dict] = None):
        self.model = LLM(
            model=model_config["model_name"],
            tokenizer=model_config.get("tokenizer_name") or model_config["model_name"],
            dtype=model_config["dtype"],
            gpu_memory_utilization=model_config["gpu_memory_utilization"],
            max_model_len=model_config["max_model_len"],
            max_num_batched_tokens=model_config.get("max_num_batched_tokens"),
            tensor_parallel_size=model_config["tensor_parallel_size"],
            pipeline_parallel_size=model_config["pipeline_parallel_size"],
            distributed_executor_backend=model_config["distributed_executor_backend"],
            trust_remote_code=model_config["trust_remote_code"],
            enable_chunked_prefill=model_config["enable_chunked_prefill"],
            enable_prefix_caching=model_config["enable_prefix_caching"],
            enforce_eager=model_config["enforce_eager"],
            enable_lora=lora_config is not None,
        )
        self._sampling_params_base = sampling_params
        self._lora_req = None
        if lora_config:
            self._lora_req = LoRARequest(
                lora_config["lora_name"],
                1,
                lora_config["lora_path"]
            )

    def generate(self, prompts: List[str], sampling_overrides: Optional[List[Dict]] = None) -> List[str]:
        """Generate completions for a batch of prompts."""
        if not prompts:
            return []
        
        if sampling_overrides:
            sampling_params = [
                SamplingParams(**{**self._sampling_params_base, **override})
                for override in sampling_overrides
            ]
        else:
            sampling_params = SamplingParams(**self._sampling_params_base)
        
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=self._lora_req
        )
        return [out.text for req in outputs for out in req.outputs]

class OpenLMEngine:
    """Production-ready vLLM inference engine with optional data parallelism."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.tokenizer_name = config.tokenizer_name or config.model_name
        self._dp_enabled = config.data_parallel_replicas > 1
        self._workers = None
        self._lora_req = None
        
        self._init_sampling_params()
        self._load_tokenizer()
        
        if self._dp_enabled:
            self._init_data_parallel()
        else:
            self._load_model()

    def _init_sampling_params(self) -> None:
        """Initialize base sampling parameters."""
        self.sampling_params = {
            "n": self.config.n,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "stop": self.config.stop_tokens,
            "logprobs": self.config.logprobs,
            "prompt_logprobs": self.config.prompt_logprobs,
        }

    def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=self.config.trust_remote_code
        )
        self.tokenizer.model_max_length = self.config.max_model_len

    def _load_model(self) -> None:
        """Load vLLM model for single-replica inference."""
        logging.info(f"Loading model: {self.model_name}")
        
        self.model = LLM(
            model=self.model_name,
            tokenizer=self.tokenizer_name,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            distributed_executor_backend=self.config.distributed_executor_backend,
            trust_remote_code=self.config.trust_remote_code,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            enable_prefix_caching=self.config.enable_prefix_caching,
            enforce_eager=self.config.enforce_eager,
            enable_lora=self.config.lora_path is not None,
        )
        
        if self.config.lora_path:
            self._lora_req = LoRARequest(
                self.config.lora_name,
                1,
                self.config.lora_path
            )

    def _init_data_parallel(self) -> None:
        """Initialize Ray and create worker pool for data parallelism."""
        if not ray.is_initialized():
            ray.init(
                address=self.config.ray_address,
                ignore_reinit_error=True
            )
        
        model_config = {
            "model_name": self.config.model_name,
            "tokenizer_name": self.tokenizer_name,
            "dtype": self.config.dtype,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "trust_remote_code": self.config.trust_remote_code,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enforce_eager": self.config.enforce_eager,
        }
        
        lora_config = None
        if self.config.lora_path:
            lora_config = {
                "lora_name": self.config.lora_name,
                "lora_path": self.config.lora_path,
            }
        
        self._workers = [
            _VLLMWorker.remote(model_config, self.sampling_params, lora_config)
            for _ in range(self.config.data_parallel_replicas)
        ]
        
        logging.info(
            f"Initialized data-parallel with {self.config.data_parallel_replicas} replica(s)."
        )

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_overrides: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Generate responses for prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            sampling_overrides: Optional per-prompt sampling parameter overrides
        
        Returns:
            DataFrame with 'response' column containing generated text
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if sampling_overrides and len(sampling_overrides) != len(prompts):
            raise ValueError(
                f"sampling_overrides length ({len(sampling_overrides)}) "
                f"must match prompts length ({len(prompts)})"
            )
        
        start = time.monotonic()
        
        if not self._dp_enabled:
            responses = self._generate_single(prompts, sampling_overrides)
        else:
            responses = self._generate_distributed(prompts, sampling_overrides)
        
        duration = time.monotonic() - start
        logging.info(f"Generated {len(prompts)} prompt(s) in {duration:.2f}s")
        
        return pd.DataFrame(responses, columns=['response'])

    def _generate_single(
        self,
        prompts: List[str],
        sampling_overrides: Optional[List[Dict]]
    ) -> List[str]:
        """Generate using single vLLM instance."""
        if sampling_overrides:
            sampling_params = [
                SamplingParams(**{**self.sampling_params, **override})
                for override in sampling_overrides
            ]
        else:
            sampling_params = SamplingParams(**self.sampling_params)
        
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=self._lora_req
        )
        return [out.text for req in outputs for out in req.outputs]

    def _generate_distributed(
        self,
        prompts: List[str],
        sampling_overrides: Optional[List[Dict]]
    ) -> List[str]:
        """Generate using data-parallel workers."""
        num_workers = len(self._workers)
        
        # Shard prompts and overrides using round-robin
        prompt_shards = [[] for _ in range(num_workers)]
        override_shards = [[] for _ in range(num_workers)] if sampling_overrides else None
        
        for i, prompt in enumerate(prompts):
            worker_idx = i % num_workers
            prompt_shards[worker_idx].append(prompt)
            if override_shards is not None:
                override_shards[worker_idx].append(sampling_overrides[i])
        
        # Submit work to workers
        futures = [
            worker.generate.remote(
                prompt_shards[i],
                override_shards[i] if override_shards else None
            )
            for i, worker in enumerate(self._workers)
            if prompt_shards[i]  # Only submit if shard is non-empty
        ]
        
        # Gather results
        results = ray.get(futures) if futures else []
        return [response for shard_results in results for response in shard_results]

    def interactive_console(self) -> None:
        """Interactive console with tab switching between chat and generate modes."""
        mode = "chat"  # Start in chat mode
        session = PromptSession()
        
        # Setup key bindings
        kb = KeyBindings()
        
        @kb.add('c-m', eager=True)
        def _(event):
            event.current_buffer.validate_and_handle()
        
        @kb.add('tab')
        def _(event):
            # Toggle mode on tab
            nonlocal mode
            mode = "generate" if mode == "chat" else "chat"
            # Clear current input and show mode change
            event.current_buffer.text = ""
            print(f"\n→ Switched to {mode.upper()} mode\n")
        
        print("╭─────────────────────────────────────────╮")
        print("│   Interactive LLM Console               │")
        print("├─────────────────────────────────────────┤")
        print("│  Tab         : Switch chat/generate     │")
        print("│  Ctrl+Enter  : Submit                   │")
        print("│  Ctrl+C      : Exit                     │")
        print("╰─────────────────────────────────────────╯\n")
        
        while True:
            try:
                # Show current mode
                mode_color = "ansicyan" if mode == "chat" else "ansigreen"
                prompt_text = HTML(f'<{mode_color}>[{mode.upper()}]</{mode_color}> You: ')
                
                user_input = session.prompt(
                    prompt_text,
                    multiline=True,
                    key_bindings=kb
                )
                
                if not user_input.strip():
                    continue
                
                # Process based on mode
                if mode == "chat":
                    conversation = [{'role': 'user', 'content': user_input}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    df = self.generate(formatted_prompt)
                else:
                    df = self.generate(user_input)
                
                # Display response
                print("\n" + "─" * 60)
                for resp in df['response']:
                    print(f"{resp}")
                print("─" * 60 + "\n")
                    
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n⚠ Error: {e}\n")

    def console_generate(self) -> None:
        """Legacy: Interactive mode for generation."""
        print("Note: Use interactive_console() for better UI with tab switching.\n")
        print("Interactive generation mode. Type 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() == 'exit':
                    print("Exiting interactive session.")
                    break
                
                if not user_input:
                    continue
                
                df = self.generate(user_input)
                for resp in df['response']:
                    print(f"Assistant: {resp}\n")
                    
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive session.")
                break
            except Exception as e:
                logging.error(f"Generation error: {e}")

    def console_chat_completions(self) -> None:
        """Legacy: Interactive mode for chat."""
        print("Note: Use interactive_console() for better UI with tab switching.\n")
        print("Interactive chat mode. Type 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() == 'exit':
                    print("Exiting interactive session.")
                    break
                
                if not user_input:
                    continue
                
                conversation = [{'role': 'user', 'content': user_input}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                df = self.generate(formatted_prompt)
                for resp in df['response']:
                    print(f"Assistant: {resp}\n")
                    
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive session.")
                break
            except Exception as e:
                logging.error(f"Generation error: {e}")

if __name__ == '__main__':
    config = ModelConfig(
        model_name="open-thoughts/OpenThinker3-1.5B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_tokens=32768,
        temperature=0.6,
        top_p=1.0,
        top_k=-1,
    )
    engine = OpenLMEngine(config)
    engine.interactive_console()