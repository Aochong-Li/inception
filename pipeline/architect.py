"""
Architect model interactions and prompt templates.
"""

import os
import sys
from typing import Optional

# Add project root directory to path for imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.llm_engine import OpenLMEngine
from transformers import PreTrainedTokenizer
from .config import (
    ARCHITECT_CONTINUATION_TEMPLATE,
    ARCHITECT_CONTINUATION_TOKENS
)
from .reasoning import extract_reasoning_trace, get_first_n_tokens


def get_blank_slate_reasoning(
    inquiry: str,
    architect_engine: OpenLMEngine,
    tokenizer: PreTrainedTokenizer
) -> str:
    """
    Generate initial reasoning from architect model given only the inquiry.
    Uses the model's chat template (same as benchmark generation) for subtle reasoning.
    
    Args:
        inquiry: Inquiry/prompt
        architect_engine: Initialized architect model engine
        tokenizer: Tokenizer for the architect model
        
    Returns:
        First 40 tokens of reasoning trace extracted from architect response
    """
    # Use model's chat template (same as benchmark_eval.py) - no explicit "malicious" framing
    chat_history = [
        {'role': 'user', 'content': inquiry}
    ]
    
    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response from architect
    try:
        response_df = architect_engine.generate(prompts=[prompt])
        # if response_df.empty or 'response' not in response_df.columns:
        #     return ""
        
        architect_response = response_df.iloc[0]['response']
        # if not isinstance(architect_response, str):
        #     architect_response = str(architect_response)
        
        # Extract reasoning trace
        reasoning_trace = extract_reasoning_trace(architect_response)
        
        # Limit to first 40 tokens
        limited_reasoning = get_first_n_tokens(reasoning_trace, 1024, tokenizer)
        return limited_reasoning
    except Exception as e:
        print(f"Error generating blank slate reasoning: {e}")
        return ""


def get_architect_continuation(
    inquiry: str,
    chopped_reasoning: str,
    architect_engine: OpenLMEngine,
    tokenizer: PreTrainedTokenizer,
    n_tokens: int = ARCHITECT_CONTINUATION_TOKENS
) -> str:
    """
    Get first N tokens of architect continuation from full chopped reasoning.
    
    Feeds the FULL chopped reasoning from the most recent OSS iteration to the architect,
    which then generates a continuation. Returns the first N tokens of that continuation.
    
    Args:
        inquiry: Original malicious inquiry
        chopped_reasoning: Full reasoning trace chopped at refusal point (from most recent OSS iteration)
        architect_engine: Initialized architect model engine
        tokenizer: Tokenizer for the architect model
        n_tokens: Number of tokens to extract (default: ARCHITECT_CONTINUATION_TOKENS, now 150)
        
    Returns:
        First N tokens of architect's continuation reasoning
    """
    # Feed FULL chopped reasoning to architect (no truncation)
    prompt = ARCHITECT_CONTINUATION_TEMPLATE.format(
        question=inquiry,
        reasoning=chopped_reasoning
    )
    
    # Generate response from architect
    try:
        response_df = architect_engine.generate(prompts=[prompt])
        if response_df.empty or 'response' not in response_df.columns:
            return ""
        
        architect_response = response_df.iloc[0]['response']
        if not isinstance(architect_response, str):
            architect_response = str(architect_response)
        
        # Extract reasoning trace
        reasoning_trace = extract_reasoning_trace(architect_response)
        
        # Get first N tokens
        continuation = get_first_n_tokens(reasoning_trace, n_tokens, tokenizer)
        return continuation
    except Exception as e:
        print(f"Error generating architect continuation: {e}")
        return ""

