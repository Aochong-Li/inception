"""
Utility functions for text processing, sentence splitting, and window building.
"""

import re
from typing import List, Sequence


def clean_text(text: str) -> str:
    """Remove asterisks and newlines, collapse whitespace."""
    cleaned = text.replace("*", " ")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences based on punctuation."""
    if not text:
        return []
    # Light sentence splitting: split on punctuation followed by whitespace
    parts = SENTENCE_SPLIT_REGEX.split(text)
    sentences: List[str] = []
    for p in parts:
        s = p.strip()
        if s:
            sentences.append(s)
    return sentences


def build_windows(sentences: Sequence[str]) -> List[List[str]]:
    """
    Create 2-3 sentence windows with 1-sentence overlap; prefer 3 where possible.
    
    Args:
        sentences: List of sentence strings
        
    Returns:
        List of window lists, where each window contains 2-3 sentences
    """
    # 2–3 sentence windows with 1-sentence overlap; prefer 3 where possible
    win_size = 3
    step = 2  # 1-sentence overlap for size 3 windows

    n = len(sentences)
    if n < 2:
        return []

    chunks: List[List[str]] = []
    i = 0
    while i + win_size <= n:
        chunks.append(list(sentences[i : i + win_size]))
        i += step

    # Remainder handling: if 2 sentences remain, make a 2-sentence chunk
    if i < n:
        remaining = n - i
        if remaining == 2:
            chunks.append(list(sentences[i : i + 2]))
        elif remaining == 1:
            # Avoid single-sentence chunks; try to create a final 2-sentence chunk
            if n >= 2 and (not chunks or chunks[-1] != list(sentences[-2:])):
                chunks.append(list(sentences[-2:]))
    return chunks

