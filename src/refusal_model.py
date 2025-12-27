from typing import Optional, List, Sequence
import fasttext
import numpy as np
import pandas as pd
import nltk
import re

def clean_text(text: str) -> str:
    """Remove asterisks and newlines, collapse whitespace."""
    cleaned = text.replace("*", "ß ")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.lower()
    return cleaned

def split_sentences_nltk(text: str) -> List[str]:
    """
    Split text into sentences using NLTK's punkt tokenizer.
    Preserves original text formatting and handles complex punctuation.
    """
    if not text:
        return []
    return nltk.sent_tokenize(text)

def classify_reasoning_trace(
    reasoning_trace: str,
    refusal_model: fasttext.FastText,
    refusal_threshold: float = 0.6,
    mode="firstonly"
) -> Optional[int]:
    """
    Classify each sentence in the reasoning trace for refusal.

    Args:
        reasoning_trace: The original text to classify
        refusal_model: FastText model for classification
        mode: "firstonly" returns index of first refusal sentence,
              "all" returns DataFrame with all results

    Returns:
        If mode="firstonly": sentence index (int) of first refusal, or None if no refusal
        If mode="all": DataFrame with columns ['sentence', 'label']
    """
    sentences = split_sentences_nltk(reasoning_trace)
    cleaned_sentences = [clean_text(sentence) for sentence in sentences]
    res = []

    for idx, sentence in enumerate(cleaned_sentences):
        window = " ".join(cleaned_sentences[max(0, idx - 1): min(len(sentences), idx + 2)])
        try:
            predictions = refusal_model.predict(window.strip(), k=1)
            labels = predictions[0]
            label_str = labels[0]
            conf = predictions[1][0]

            label_int = int(label_str.split('__label__')[1])
            label_int = 1 if label_int == 1 and conf >= refusal_threshold else 0

            if label_int == 1 and mode == "firstonly":
                return idx  # Return sentence index of first refusal
            res.append({'sentence': sentence, 'label': label_int})

        except Exception as e:
            raise e

    if mode == "firstonly":
        return None
    else:
        return pd.DataFrame(res)

def chop_at_refusal(
    reasoning_trace: str,
    refusal_idx: Optional[int],
) -> str:
    """
    Chop the reasoning trace at the first refusal sentence.

    Args:
        reasoning_trace: The original text
        refusal_idx: Sentence index where refusal was detected (from classify_reasoning_trace)

    Returns:
        Original text up to (not including) the refusal sentence.
        If refusal_idx is None, returns original text unchanged.
    """
    if refusal_idx is None:
        return reasoning_trace

    sentences = split_sentences_nltk(reasoning_trace)
    return " ".join(sentences[:refusal_idx])

