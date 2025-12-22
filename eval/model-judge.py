import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.openai_engine import OpenAI_Engine
import pandas as pd
from pathlib import Path
import argparse
import re
import os
import time

from reward_score.math500 import math_verify_score, math_if_boxed

PROMPT_TEMPLATE = '''### System Prompt
You are an unbiased examiner who evaluates whether a student's answer to a given question is correct. 
Your task is to determine if the student's final answer matches the standard answer provided, based solely on correctness and the question's specific requirements. 
Do not perform any additional calculations or reinterpret the question. Simply compare the student's answer to the standard answer to determine if it satisfies the question's requirements.

Focus strictly on:
1. Understanding the exact requirement of the question.
2. Comparing the student's final answer directly and rigorously to the provided standard answer.
3. Your task is not to solve the problem but to determine whether the student's answer is correct based on the question's requirements. Avoid any unnecessary analysis, assumptions, or re-solving the problem.

Note:
- For intervals/ranges: The student's answer must cover the EXACT SAME range as the standard answer, NOT just any single value or subset within that range;
- If the standard answer contains multiple solutions connected by "or"/"and", all of them must be listed in the student's answer;
- If student's response does not mention any answer, it is considered WRONG;
- You must be deterministic and rigorous - always declare the answer as either CORRECT or WRONG;
- Small rounding differences are permitted.

Your response must include:
### Short Analysis
Provide a short and evidence-backed analysis between <analysis> </analysis> tags, in which you should extract the final solution value from the standard answer and the student's answer and judge whether they are the same.

### Correctness
Based on the analysis, you should report a label CORRECT or WRONG between <judge> </judge> tags (e.g., <judge>CORRECT</judge> or <judge>WRONG</judge>).

### User Prompt
Problem: {problem}

Standard Answer: {standard_answer}

Student Answer: {student_answer}'''

class ModelJudge():
    def __init__(self,
                 input_df: pd.DataFrame,
                 problem_col: str,
                 gt_col: str,
                 response_col: str,
                 pred_col: str,
                 output_dir: str,
                 nick_name: str
                 ):
        self.input_df = input_df
        self.problem_col = problem_col
        self.gt_col = gt_col
        self.response_col = response_col
        self.pred_col = pred_col
        self.output_dir = output_dir
        self.nick_name = nick_name
        
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_pred(self, row):
        pred = row[self.pred_col]
        response = row[self.response_col]
        if "\\boxed{" in pred:
            return pred
        else:
            return response

    def extract_label(self, row):
        raw_response = row['model_judge']

        response = raw_response.lower().replace(' ', '').replace('\n', '').strip()
        pattern = r"<judge>(.*?)</judge>"

        match = re.search(pattern, response)
        if match and match.group(1) == "correct":
            return 1.0
        elif match and match.group(1) == "wrong":
            return 0.0
        elif "CORRECT" in raw_response and "WRONG" not in raw_response:
            return 1.0
        elif "WRONG" in raw_response and "CORRECT" not in raw_response:
            return 0.0
        else:
            return 0.0

    def run(self, overwrite: bool = True, strict_boxed: bool = True) -> pd.DataFrame:
        self.eval_df = self.input_df.copy()
        
        if "gt" in self.eval_df.columns:
            self.eval_df = self.eval_df.rename(columns={"gt": "ground_truth"})
        if "model_is_correct" in self.eval_df.columns:
            self.eval_df = self.eval_df.drop(columns=["model_is_correct"])
        if "model_judge" in self.eval_df.columns:
            self.eval_df = self.eval_df.drop(columns=["model_judge"])
        
        self.eval_df['model_is_correct'] = self.eval_df.apply(lambda x: math_verify_score(x[self.pred_col], x[self.gt_col], x[self.response_col]), axis=1)
        if strict_boxed:
            self.keep_subset = self.eval_df[
                (self.eval_df['model_is_correct'] == 1.0) | ((self.eval_df['model_is_correct'] == 0.0) & (~self.eval_df["if_boxed"]))
                ].reset_index(drop=True)
            
            self.check_subset  = self.eval_df[
                (self.eval_df['model_is_correct'] == 0.0) & (self.eval_df["if_boxed"])
            ].reset_index(drop=True)
        else:
            self.keep_subset = self.eval_df[(self.eval_df['model_is_correct'] == 1.0)].reset_index(drop=True)
            self.check_subset = self.eval_df[(self.eval_df['model_is_correct'] == 0.0)].reset_index(drop=True)
        
        self.check_subset['model_pred'] = self.check_subset.apply(self.extract_pred, axis=1)
        self.check_subset = self.check_subset.drop(columns = ['model_is_correct'])
        
        engine = OpenAI_Engine(
            input_df=self.check_subset,
            prompt_template=PROMPT_TEMPLATE,
            template_map={"problem": self.problem_col, "standard_answer": self.gt_col, "student_answer": "model_pred"},
            nick_name=f"model_judge_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/reasoning",
            cache_filepath=self.output_dir + f"/{self.nick_name}_model_judge.pickle",
            model = "deepseek-chat",
            client_name = "deepseek"
        )
        
        engine.run_model(overwrite=overwrite)
        self.response = engine.retrieve_outputs()
        if 'response' in self.response.columns:
            self.response = self.response.set_index('idx').rename(columns={'response': 'model_judge'})
            self.response = self.response.explode(['model_judge'])

        self.response['model_is_correct'] = self.response.apply(self.extract_label, axis=1)
        self.response.to_pickle(self.output_dir + f"/{self.nick_name}_model_judge.pickle")
        return self.response
    
    def merge(self, strict_has_answer: bool = False) -> pd.DataFrame:
        assert self.check_subset.index.equals(self.response.index), "check_subset and response have different indices"

        self.check_subset = self.check_subset.merge(
            self.response[['model_judge','model_is_correct']], 
            left_index=True,
            right_index=True).drop(columns=['model_pred']
            )

        self.result_df = pd.concat([self.keep_subset, self.check_subset], axis=0, ignore_index=True)
        if strict_has_answer:
            self.result_df = self.strict_has_answer(self.result_df)
        return self.result_df
    
    def strict_has_answer(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df[self.response_col] == df[self.pred_col], 'model_is_correct'] = 0.0
        return df
    
if __name__ == "__main__":
    """
    Example usage:
    python reward_score/model-judge.py \
      --input_filepath ./results/allmath/benchmark/LIMO-Qwen-32B.pickle \
      --output_dir ./results/allmath/benchmark/model_judge \
      --nick_name LIMO-Qwen-32B

    python reward_score/model-judge.py \
      --input_dir ./results/allmath/benchmark \
      --output_dir ./results/allmath/benchmark/model_judge
    """

    parser = argparse.ArgumentParser(
        description="Judge model predictions using an LLM."
    )
    parser.add_argument("--input_dir", type=str, required=False, help="Directory to read input")
    parser.add_argument("--input_filepath", type=str, required=False, help="Path to input pickle file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--nick_name", type=str, required=False, help="Nickname for this run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if set.")
    
    args = parser.parse_args()
    problem_col = "problem"
    gt_col = "ground_truth"
    pred_col = "pred"
    response_col = "response"
    strict_boxed = False
    strict_has_answer = True
    
    if args.input_filepath:
        input_df = pd.read_pickle(args.input_filepath)
        judge_engine = ModelJudge(
            input_df=input_df,
            problem_col=problem_col,
            gt_col=gt_col,
            response_col=response_col,
            pred_col=pred_col,
            output_dir=args.output_dir,
            nick_name=args.nick_name
        )
        judge_engine.run(overwrite=args.overwrite, strict_boxed=strict_boxed)
        result_df = judge_engine.merge(strict_has_answer=strict_has_answer)
        result_df.to_pickle(os.path.join(args.output_dir, f"{args.nick_name}.pickle"))
        
    elif args.input_dir:
        finished = []
        time_limit = 10 * 60 * 60 # 10 hours
        start_time = time.time()
        while True:
            if time.time() - start_time > time_limit:
                break

            for fname in os.listdir(args.input_dir):
                if fname.endswith(".pickle") and fname not in finished:
                    input_df = pd.read_pickle(os.path.join(args.input_dir, fname))
                    nick_name = fname.replace(".pickle", "")

                    if not args.overwrite and os.path.exists(os.path.join(args.output_dir, f"{nick_name}_model_judge.pickle")):
                        print(f"Skipping {nick_name} because it already exists")
                        finished.append(fname)
                        continue
                    
                    if "LIMO" in nick_name:
                        strict_has_answer = False

                    judge_engine = ModelJudge(
                        input_df=input_df,
                        problem_col=problem_col,
                        gt_col=gt_col,
                        response_col=response_col,
                        pred_col=pred_col,
                        output_dir=args.output_dir,
                        nick_name=nick_name
                    )
                    
                    print(f"Starting to judge {nick_name}")
                    judge_engine.run(overwrite=args.overwrite, strict_boxed=strict_boxed)
                    result_df = judge_engine.merge(strict_has_answer=strict_has_answer)
                    result_df.to_pickle(os.path.join(args.output_dir, fname))
                    print(f"Finished judging {nick_name}")
                    finished.append(fname) 
            
            print(f"Waiting for 60 seconds before checking again")
            time.sleep(60)