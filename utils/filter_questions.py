import os
import json
import pandas as pd
import argparse
import sys
import random
from typing import Optional
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_score.math import *

def has_answer_digit(response: str) -> bool:
    try:
        _, answer = response.split('</think>')
        answer = remove_boxed(last_boxed_only_string(answer))
        return any(c.isdigit() for c in answer)
    except:
        return False

def filter_math_questions_stress_test (model_list: list,
                                  eval_dir: str,
                                  question_id: str="unique_id",
                                  min_correct: int=1,
                                  test_size: Optional[int]= None
                                  ):
    random.seed(42)
    common_question_ids = set()
    benchmark_dir = os.path.join(eval_dir, "benchmark")
    for fname in os.listdir(benchmark_dir):
        if any(model in fname for model in model_list):
            df = pd.read_pickle(os.path.join(benchmark_dir, fname))
            df = df[df["response"].apply(has_answer_digit)]
            question_ids = df.groupby(question_id)[["correct"]].sum()
            question_ids = question_ids[question_ids["correct"] >= min_correct].index.tolist()

            if len(common_question_ids) == 0:
                common_question_ids = set(question_ids)
            else:
                common_question_ids = common_question_ids.intersection(set(question_ids))
    
    if test_size is not None:
        stress_test_sample = random.sample(list(common_question_ids), test_size)
    else:
        stress_test_sample = list(common_question_ids)
    output_path = os.path.join(eval_dir, f"stress_test_question_ids.json")

    with open(output_path, "w") as f:
        json.dump(stress_test_sample, f)

def filter_countdown_questions_stress_test (model_list: list,
                                            eval_dir: str,
                                            levels: list,
                                            test_size: Optional[int]= None
                                            ):
    levels = [int(level) for level in levels]
    benchmark_dir = os.path.join(eval_dir, "benchmark")

    for fname in os.listdir(benchmark_dir):
        if any(model in fname for model in model_list):
            df = pd.read_pickle(os.path.join(benchmark_dir, fname))
            df = df[
                    (df["level"].isin(levels)) & 
                    (df["correct"] == 1.) & 
                    (df["response"].str.contains("</think>"))
                ].reset_index(drop = True)

            problems = df["problem"].tolist()
            try:
                common_problems = common_problems.intersection(set(problems))
            except:
                common_problems = set(problems)

    if test_size is not None:
        common_problems = random.sample(list(common_problems), test_size)
    
    output_path = os.path.join(eval_dir, f"stress_test_problems.json")
    with open(output_path, "w") as f:
        json.dump(list(common_problems), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)

    parser.add_argument("--question_id", type=str, required=False)
    parser.add_argument("--min_correct", type=int, required=False)
    parser.add_argument("--test_size", type=int, required=False)
    parser.add_argument("--levels", nargs="+", required=False)
    args = parser.parse_args()

    if args.task == "math":
        filter_math_questions_stress_test(ast.literal_eval(args.model_list),
                                    args.eval_dir,
                                    args.question_id,
                                    args.min_correct,
                                    args.test_size
                                    )
    elif args.task == "countdown":
        filter_countdown_questions_stress_test(ast.literal_eval(args.model_list),
                                    args.eval_dir,
                                    args.levels,
                                    args.test_size
                                    )