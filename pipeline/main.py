"""
Main pipeline orchestration for iterative refusal-aware inception.
"""

import os
import sys
import pandas as pd
from typing import Optional, List
from transformers import AutoTokenizer

# Add project root directory to path for imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.llm_engine import OpenLMEngine, ModelConfig
from core import openaiapi
from pipeline.config import (
    MAX_ITERATIONS,
    MIN_TOKENS_ADDED,
    OSS_MODEL,
    ARCHITECT_MODEL,
    OSS_PROMPT_TEMPLATE,
    OSS_PROMPT_WITH_REASONING_TEMPLATE
)

from pipeline.models import create_architect_engine, load_refusal_model
from pipeline.reasoning import extract_reasoning_trace, count_tokens, preprocess_reasoning
from pipeline.refusal import find_first_refusal_chunk_idx, chop_at_refusal
from pipeline.architect import get_blank_slate_reasoning, get_architect_continuation


class OSSRefusalFeedbackInception:
    """
    Iterative refusal-aware inception pipeline for OSS models.
    
    Implements a feedback loop that:
    1. Starts with blank slate architect reasoning
    2. Iteratively feeds reasoning to OSS model
    3. Detects refusals and chops reasoning
    4. Gets architect continuation (30 tokens)
    5. Accumulates reasoning across iterations
    6. Stops when conditions are met (max iterations, no refusal, < 10 tokens added)
    """
    
    def __init__(
        self,
        model_name: str,
        nick_name: str,
        tokenizer_name: str,
        results_dir: str,
        architect_model_name: str = None,
        architect_tokenizer_name: str = None,
        refusal_model_path: str = None,
        oss_client_name: str = "deepinfra",
        oss_temperature: float = 0.6,
        oss_max_tokens: int = 32768,
        **kwargs
    ):
        """
        Initialize the OSS refusal feedback inception pipeline.
        
        Args:
            model_name: OSS model identifier for API (e.g., "openai/gpt-oss-120b")
            nick_name: Nickname for this model run
            tokenizer_name: Tokenizer name/path (used for token counting)
            results_dir: Directory containing benchmark results
            architect_model_name: Architect model name (defaults to ARCHITECT_MODEL)
            architect_tokenizer_name: Architect tokenizer name
            refusal_model_path: Path to FastText refusal model
            oss_client_name: API client name ("deepinfra", "deepseek", "togetherai", "openrouter") - default: "deepinfra"
            oss_temperature: Temperature for OSS model (default: 0.6)
            oss_max_tokens: Max tokens for OSS model (default: 32768)
            **kwargs: Additional model configuration parameters (for architect model)
        """
        # Store basic attributes
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.oss_client_name = oss_client_name
        self.oss_temperature = oss_temperature
        self.oss_max_tokens = oss_max_tokens
        
        # Architect model config
        self.architect_model_name = architect_model_name or ARCHITECT_MODEL
        self.architect_tokenizer_name = architect_tokenizer_name or self.architect_model_name
        
        # Initialize tokenizer for token counting
        # TODO: raise error if tokenizer_name is not provided
        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                print(f"Warning: Failed to load tokenizer '{tokenizer_name}', falling back to 'gpt2': {e}")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Initialize architect engine (always local)
        self.architect_engine = create_architect_engine(
            self.architect_model_name,
            self.architect_tokenizer_name,
            **kwargs
        )
        
        self.refusal_model = load_refusal_model(refusal_model_path)
        
        self.load_datasets()
    
    def load_datasets(self):
        """Load inquiries from benchmark pickle file."""
        # TODO: this shouldn't load from benchmark results, it should load from raw inquiries

        pickle_path = "./results/wmdp_inquiries_300/benchmark/GPT-OSS-120B.pickle"
        
        # Validate path exists
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        df = pd.read_pickle(pickle_path)
        
        # Validate 'inquiry' column exists
        if 'inquiry' not in df.columns:
            raise ValueError(f"Column 'inquiry' not found in {pickle_path}. Available columns: {list(df.columns)}")
        
        # Extract only 'inquiry' column and limit to first 5
        self.df = df[['inquiry']].head(1).reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} inquiries from {pickle_path}")
    
    def generate_oss_response(self, prompts: List[str]) -> List[str]:
        """
        Generate responses from OSS model via third-party API provider.
        
        Args:
            prompts: List of prompt strings (OSS format with special tokens)
            
        Returns:
            List of response strings
        """
        # Use third-party API provider (DeepInfra, DeepSeek, etc.) - OSS uses completions format
        responses = []
        for prompt in prompts:
            # OSS prompt format includes special tokens - send as completion prompt
            # API providers (like DeepInfra) host GPT-OSS-120B and understand the OSS token format
            response, errors, attempts = openaiapi.generate_completions(
                input_prompt=prompt,  # Send full prompt with OSS tokens
                model=self.model_name,  # "openai/gpt-oss-120b" - API provider recognizes this
                client_name=self.oss_client_name,  # "deepinfra", "deepseek", etc.
                temperature=self.oss_temperature,
                max_tokens=self.oss_max_tokens,
                n=1
            )
            if response and len(response) > 0:
                responses.append(response[0])
            else:
                print(f"Warning: API call failed with errors: {errors}")
                responses.append("")
        return responses
    
    def should_stop_iteration(
        self,
        tokens_added: int,
        refusal_idx: Optional[int],
        iteration: int
    ) -> bool:
        """
        Check if iteration loop should stop.
        
        Args:
            tokens_added: Number of tokens added in this iteration
            refusal_idx: Index of first refusal chunk (None if no refusal)
            iteration: Current iteration number (0-indexed)
            
        Returns:
            True if should stop, False otherwise
        """
        if iteration >= MAX_ITERATIONS - 1:
            return True
        if tokens_added < MIN_TOKENS_ADDED:
            return True
        if refusal_idx is None:
            return True
        return False
    
    def apply_prompt_template(self, question: str, reasoning: str, use_full_template: bool = False) -> str:
        """
        Apply OSS-specific prompt template.
        
        Args:
            question: User inquiry
            reasoning: Reasoning trace to inject
            use_full_template: If True, use full template (for initial call), else use unclosed template
            
        Returns:
            Formatted prompt string
        """
        if use_full_template:
            return OSS_PROMPT_TEMPLATE.format(question=question, reasoning=reasoning)
        else:
            return OSS_PROMPT_WITH_REASONING_TEMPLATE.format(question=question, reasoning=reasoning)
    
    def run_pipeline(self):
        """
        Execute the full iterative pipeline.
        
        Flow:
        1. Initialization: Get blank slate reasoning from architect, feed to OSS
        2. Iterative loop (max 8 iterations):
           - Feed cumulative reasoning to OSS
           - Extract reasoning trace
           - Calculate tokens added
           - Detect refusal and chop
           - Get architect continuation (30 tokens)
           - Append to cumulative reasoning
           - Check stopping conditions
        3. Save results with final reasoning trace
        """
        print("=" * 80)
        print("Starting Iterative Refusal-Aware Inception Pipeline")
        print("=" * 80)
        
        # Get architect tokenizer (needed for architect operations)
        architect_tokenizer = self.architect_engine.tokenizer
        
        # Initialize iteration tracking
        self.df['iteration_data'] = None
        self.df['final_reasoning_trace'] = None
        self.df['iterations_completed'] = 0
        self.df['stop_reason'] = ''
        
        # Step 1: Initialization - Get blank slate reasoning from architect
        print("\nStep 1: Getting blank slate reasoning from architect...")
        
        initial_reasoning_list = []
        # TODO: this needs to be parallelized
        for idx, row in self.df.iterrows():
            inquiry = row['inquiry']
            blank_slate_reasoning = get_blank_slate_reasoning(
                inquiry,
                self.architect_engine,
                architect_tokenizer
            )
            initial_reasoning_list.append(blank_slate_reasoning)
        
        self.df['initial_architect_reasoning'] = initial_reasoning_list
        
        # Step 2: Feed initial reasoning to OSS model (first call)
        print("Step 2: Feeding initial reasoning to OSS model...")
        initial_prompts = self.df.apply(
            lambda row: self.apply_prompt_template(
                row['inquiry'],
                row['initial_architect_reasoning'],
                use_full_template=True
            ),
            axis=1
        ).tolist()
        
        initial_responses = self.generate_oss_response(initial_prompts)
        self.df['initial_oss_response'] = initial_responses
        import pdb; pdb.set_trace()
        refusal_idx = find_first_refusal_chunk_idx(
            initial_responses[0],
            self.refusal_model,
            preprocess_func=preprocess_reasoning
        )

        
        
        # Extract reasoning trace from initial OSS response
        self.df['cumulative_reasoning'] = self.df['initial_oss_response'].apply(
            extract_reasoning_trace
        )
        
        # Step 3: Iterative loop
        print("\nStep 3: Starting iterative loop...")
        for iteration in range(MAX_ITERATIONS):
            print(f"\n  Iteration {iteration + 1}/{MAX_ITERATIONS}")
            
            # Filter rows that haven't stopped yet
            active_rows = self.df[(self.df['stop_reason'].isna()) | (self.df['stop_reason'] == '')].copy()
            
            if len(active_rows) == 0:
                print("  All inquiries have already stopped")
                break
            
            iteration_data_list = []
            
            for idx in active_rows.index:
                row = self.df.loc[idx]
                inquiry = row['inquiry']
                previous_reasoning = row['cumulative_reasoning'] or ""
                
                # Feed cumulative reasoning to OSS model
                prompt = self.apply_prompt_template(
                    inquiry,
                    previous_reasoning,
                    use_full_template=False
                )
                
                responses = self.generate_oss_response([prompt])
                oss_response = responses[0] if responses else ""
                
                # Skip if empty response
                if not oss_response:
                    print(f"  Warning: Empty response for inquiry {idx}, skipping iteration")
                    continue
                
                # Extract reasoning trace
                new_reasoning = extract_reasoning_trace(oss_response)
                
                # Calculate tokens in the new reasoning trace (direct count, not difference)
                tokens_added = count_tokens(new_reasoning, self.tokenizer)
                
                # Find refusal
                refusal_idx = find_first_refusal_chunk_idx(
                    new_reasoning,
                    self.refusal_model,
                    preprocess_func=preprocess_reasoning
                )
                
                # Check stopping conditions
                should_stop = self.should_stop_iteration(tokens_added, refusal_idx, iteration)
                
                iteration_data = {
                    'iteration': iteration + 1,
                    'oss_response': oss_response,
                    'oss_reasoning': new_reasoning,
                    'tokens_added': tokens_added,
                    'refusal_chunk_idx': refusal_idx,
                    'should_stop': should_stop
                }
                
                if should_stop:
                    # Determine stop reason (check in priority order)
                    if iteration >= MAX_ITERATIONS - 1:
                        stop_reason = "max_iterations_reached"
                    elif tokens_added < MIN_TOKENS_ADDED:
                        stop_reason = f"tokens_added_{tokens_added}_below_threshold"
                    else:
                        stop_reason = "no_refusal_detected"
                    
                    iteration_data['stop_reason'] = stop_reason
                    iteration_data['chopped_reasoning'] = None
                    iteration_data['architect_continuation'] = None
                    iteration_data['updated_cumulative_reasoning'] = new_reasoning
                    
                    # Update dataframe
                    self.df.at[idx, 'final_reasoning_trace'] = new_reasoning
                    self.df.at[idx, 'iterations_completed'] = iteration + 1
                    self.df.at[idx, 'stop_reason'] = stop_reason
                else:
                    # Chop at refusal
                    chopped_reasoning = chop_at_refusal(new_reasoning, refusal_idx, preprocess_func=preprocess_reasoning)
                    
                    # Get architect continuation (150 tokens) - dynamically generated based on full chopped OSS reasoning
                    architect_continuation = get_architect_continuation(
                        inquiry,
                        chopped_reasoning,
                        self.architect_engine,
                        architect_tokenizer
                    )
                    
                    # Append architect continuation to chopped reasoning
                    updated_cumulative = " ".join([chopped_reasoning.strip(), architect_continuation.strip()])
                    
                    iteration_data['chopped_reasoning'] = chopped_reasoning
                    iteration_data['architect_continuation'] = architect_continuation
                    iteration_data['updated_cumulative_reasoning'] = updated_cumulative
                    
                    # Update cumulative reasoning for next iteration
                    self.df.at[idx, 'cumulative_reasoning'] = updated_cumulative
                
                # Store iteration data
                current_data = self.df.at[idx, 'iteration_data']
                if current_data is None or (isinstance(current_data, float) and pd.isna(current_data)):
                    self.df.at[idx, 'iteration_data'] = [iteration_data]
                else:
                    self.df.at[idx, 'iteration_data'].append(iteration_data)
                
                iteration_data_list.append(iteration_data)
            
            # Check if all active rows stopped
            all_stopped = all(
                data.get('should_stop', False) for data in iteration_data_list
            )
            if all_stopped:
                print(f"  All inquiries stopped at iteration {iteration + 1}")
                break
        
        # Step 4: Save results
        print("\nStep 4: Saving results...")
        output_path = os.path.join(
            self.results_dir,
            "inception",
            f"{self.nick_name}_iterative_refusal_feedback.pickle"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_pickle(output_path)
        
        print(f"\nPipeline complete! Results saved to {output_path}")
        print(f"Processed {len(self.df)} inquiries")
        print(f"Average iterations: {self.df['iterations_completed'].mean():.2f}")
        print("=" * 80)
        
        return self.df


if __name__ == "__main__":
    # TODO: changet this to arg parse not hardcoded

    pipeline = OSSRefusalFeedbackInception(
        model_name="openai/gpt-oss-120b",  # Model identifier (API provider hosts this)
        nick_name="GPT-OSS-120B",
        tokenizer_name="openai/gpt-oss-120b",  # For token counting
        results_dir="./pipeline/inquiries",  # Points to inquiries folder with benchmark/ subdirectory
        oss_client_name="deepinfra",  # Options: "deepinfra", "deepseek", "togetherai", "openrouter"
        oss_temperature=0.6,
        oss_max_tokens=32768,
    )
    import pdb; pdb.set_trace()
    pipeline.run_pipeline()
