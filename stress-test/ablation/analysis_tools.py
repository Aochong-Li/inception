import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer

def first_answer_pos (model_name: str, nick_name: str, results_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = pd.read_pickle(os.path.join(results_dir, "annotated_reasoning", f"{nick_name}.pickle"))
    df = df[(df["correct"] == 1) & (df["1st_answer_chunk_index"].notnull())]
    
    def count_tokens (row):
        chunks = row["chunks"]
        index = int(row["1st_answer_chunk_index"])
        pre_answer_chunks = chunks[:index+1]
        post_answer_chunks = chunks[index+1:]
        
        pre_answer_tokens = len(tokenizer.encode("".join(pre_answer_chunks)))
        post_answer_tokens = len(tokenizer.encode("".join(post_answer_chunks)))
        
        return pre_answer_tokens, post_answer_tokens
    
    df["derivation_tokens"], df["verification_tokens"] = zip(*df.apply(count_tokens, axis=1))
    df["thinking_tokens"] = df["derivation_tokens"] + df["verification_tokens"]

    df.to_pickle(os.path.join(results_dir, "annotated_reasoning", "first_answer_pos", f"{nick_name}_first_answer_pos.pickle"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--analysis_type", type=str, required=True)
    args = parser.parse_args()
    
    if args.analysis_type == "first_answer_pos":
        first_answer_pos(args.model_name, args.nick_name, args.results_dir)