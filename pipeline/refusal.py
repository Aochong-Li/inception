"""
Refusal detection logic using FastText model.
"""

from typing import Optional
import fasttext
from pipeline.reasoning import preprocess_reasoning


def find_first_refusal_chunk_idx(
    reasoning_trace: str, 
    refusal_model: fasttext.FastText,
    preprocess_func=None
) -> Optional[int]:
    """
    Find the first chunk index that contains a refusal.
    
    Args:
        reasoning_trace: Reasoning trace to analyze
        refusal_model: FastText model for refusal detection
        preprocess_func: Function to preprocess reasoning into chunks (defaults to preprocess_reasoning)
        
    Returns:
        Index of first refusal chunk, or None if no refusal found
    """
    if preprocess_func is None:
        preprocess_func = preprocess_reasoning
    
    chunks = preprocess_func(reasoning_trace)
    
    for idx, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            continue
        
        try:
            predictions = refusal_model.predict(chunk.strip(), k=1)
            labels = predictions[0]
            if labels:
                label_str = labels[0]
                # FastText labels are typically "__label__0" or "__label__1"
                if '__label__' in label_str:
                    label_int = int(label_str.split('__label__')[1])
                    if label_int == 1:  # Refusal detected
                        return idx
        except Exception as e:
            # If prediction fails, continue to next chunk
            if idx == 0:  # Only log first failure to avoid spam
                print(f"Warning: FastText prediction failed for chunk {idx}: {e}")
            continue
    
    return None


def chop_at_refusal(
    reasoning_trace: str, 
    refusal_idx: Optional[int],
    preprocess_func=None
) -> str:
    """
    Chop reasoning trace at refusal point (remove refusal chunk and all after).
    
    Args:
        reasoning_trace: Full reasoning trace
        refusal_idx: Index of first refusal chunk (None if no refusal)
        preprocess_func: Function to preprocess reasoning into chunks (defaults to preprocess_reasoning)
        
    Returns:
        Chopped reasoning trace (up to but not including refusal chunk)
    """
    if preprocess_func is None:
        preprocess_func = preprocess_reasoning
    
    chunks = preprocess_func(reasoning_trace)
    
    if refusal_idx is None:
        # No refusal found, return all chunks
        return " ".join(chunks)
    else:
        # Return chunks up to (but not including) first refusal
        return " ".join(chunks[:refusal_idx])

