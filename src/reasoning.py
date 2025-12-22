"""
Reasoning trace extraction, preprocessing, chunking, and token counting utilities.
"""

import re
from typing import List, Optional
from transformers import PreTrainedTokenizer
from pipeline.config import OSS_REASONING_DELIMITER
from pipeline.utilities import clean_text, split_sentences, build_windows


def extract_reasoning_trace(response: str, delimiter: str = None) -> str:
    """
    Extract reasoning trace from OSS or architect response.
    
    Args:
        response: Full response text from model
        delimiter: Delimiter to split on (defaults to OSS_REASONING_DELIMITER)
        
    Returns:
        Extracted reasoning trace string
    """
    if delimiter is None:
        delimiter = OSS_REASONING_DELIMITER
    
    if not isinstance(response, str):
        return str(response) if response is not None else ""
    
    # Try OSS delimiter first
    if delimiter in response:
        parts = response.split(delimiter, 1)
        return parts[0].strip()
    
    # Try redacted_reasoning tags (for architect or other models)
    if '</think>' in response:
        reasoning = response.split('</think>')[0].strip()
        reasoning = reasoning.replace('<think>', '').strip()
        return reasoning
    
    # Try regex pattern for redacted_reasoning
    pattern = re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL | re.IGNORECASE)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    
    # If no delimiter found, return the whole text
    return response.strip()


def preprocess_reasoning(reasoning: str) -> List[str]:
    """
    Preprocess reasoning into sentence-based windows.
    
    Args:
        reasoning: Raw reasoning trace string
        
    Returns:
        List of window strings (each window is 2-3 sentences)
    """
    cleaned = clean_text(reasoning)
    sentences = split_sentences(cleaned)
    windows = build_windows(sentences)
    return [" ".join(window) for window in windows]


def count_tokens(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Count tokens in text using tokenizer.
    
    Args:
        text: Text to count tokens for
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    if not text or not isinstance(text, str):
        return 0
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)
    except Exception:
        # Fallback to word count if tokenization fails
        return len(text.split())


def get_first_n_tokens(text: str, n: int, tokenizer: PreTrainedTokenizer) -> str:
    """
    Extract first N tokens from text.
    
    Args:
        text: Text to extract tokens from
        n: Number of tokens to extract
        tokenizer: Tokenizer instance
        
    Returns:
        String containing first N tokens
    """
    if not text or not isinstance(text, str):
        return ""
    
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) <= n:
            return text
        
        # Decode first n tokens
        truncated_encoded = encoded[:n]
        truncated_text = tokenizer.decode(truncated_encoded, skip_special_tokens=True)
        return truncated_text
    except Exception:
        # Fallback: return first n words if tokenization fails
        words = text.split()
        if len(words) <= n:
            return text
        return " ".join(words[:n])

