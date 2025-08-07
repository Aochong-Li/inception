from core.openai_engine import OpenAI_Engine
import pandas as pd
from pathlib import Path
import argparse
import re
import os
import time

PROMPT_TEMPLATE = """### System Prompt
You are an experienced examiner who evaluates whether a student's answer to a given question is correct.
Two canonical questions (Question ONE and Question TWO) and their solutions are provided. Your job is to decide which question the student is answering and whether the stated answer matches that question's solution. Do not re-solve the problems; simply compare.

### Decision Rules (deterministic)
1. If the student's final answer addresses Question ONE and matches the final answer in Solution ONE, report ONE.
2. If it addresses Question TWO and matches the final answer in Solution TWO, report TWO.
3. If it explicitly answers BOTH questions and each part matches its corresponding solution (rare), report BOTH.
4. If it matches neither, report WRONG.

### Evaluation Focus
- Understand the exact requirement of each question (e.g., what the question asks for).
- Analyze which of the questions the student answer is targeting at the end of its response.
- Compare the student's final answer directly and only to the final answer, not intermediate results, in the corresponding solution. The final answer is usually boxed in solutions.
- Your task is not to solve the problem but to determine whether the student's answer is correct based on the question's requirements. Avoid any unnecessary analysis, assumptions, or re-solving the problem.
- - If the two solutions happen to be numerically identical but the student's explanation clearly corresponds to only one question, report that question (ONE or TWO), not BOTH.

### Strict Matching Guidance
- Intervals/ranges: The student's answer must cover the EXACT SAsME range as the corresponding solution, NOT just a value or subset.
- If a solution lists multiple required values/items joined by "or"/"and", all of them must appear appropriately in the student's answer for a match.
- If the student's response provides no substantive answer, report WRONG.
- Always output exactly one of ONE, TWO, BOTH, or WRONG.

Your response must include:

### Short Analysis
Provide a brief (< 50 words) comparison explaining which question the student answered and whether it matches the solution, between <analysis> </analysis> tags.

### Correctness
At the end, report ONE, TWO, BOTH, or WRONG between <judge> </judge> tags (e.g., <judge>TWO</judge>).

### User Prompt
>>>Question ONE: {problem1}

>>>Solution ONE: {solution1}

>>>Question TWO: {problem2}

>>>Solution TWO: {solution2}

>>>Student's Final Answer: {model_pred}
"""

class ModelJudge():
    def __init__(self,
                 input_df: pd.DataFrame,
                 problem1_col: str,
                 problem2_col: str,
                 solution1_col: str,
                 solution2_col: str,
                 response_col: str,
                 pred_col: str,
                 output_dir: str,
                 nick_name: str
                 ):
        self.input_df = input_df
        self.problem1_col = problem1_col
        self.problem2_col = problem2_col
        self.solution1_col = solution1_col
        self.solution2_col = solution2_col
        self.response_col = response_col
        self.pred_col = pred_col
        self.output_dir = output_dir
        self.nick_name = nick_name
        
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_pred(self, row):
        pred = row[self.pred_col]
        if pred != "":
            return pred
        return row[self.response_col]    
    
    def extract_label(self, row):
        raw_response = row['model_judge']

        try:
            response = raw_response.lower().replace(' ', '').replace('\n', '').strip()
            pattern = r"<judge>(.*?)</judge>"

            match = re.search(pattern, response)
            if match and match.group(1) == "one":
                return True, False
            elif match and match.group(1) == "two":
                return False, True
            elif match and match.group(1) == "both":
                return True, True
            elif match and match.group(1) == "wrong":
                return False, False
            else:
                return None, None
        except Exception as e:
            print(f"Error extracting label from {raw_response}")
            print(e)
            return None, None

    def run(self, overwrite: bool = True) -> None:
        self.eval_df = self.input_df.copy()
        self.eval_df['model_pred'] = self.eval_df.apply(self.extract_pred, axis=1)
        
        engine = OpenAI_Engine(
            input_df=self.eval_df,
            prompt_template=PROMPT_TEMPLATE,
            template_map={
                "problem1": self.problem1_col,
                "problem2": self.problem2_col,
                "solution1": self.solution1_col,
                "solution2": self.solution2_col,
                "model_pred": "model_pred"
                },
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
        
        self.response[f'{self.solution1_col}_is_correct'], self.response[f'{self.solution2_col}_is_correct'] = zip(*self.response.apply(self.extract_label, axis=1))
        self.response.to_pickle(self.output_dir + f"/{self.nick_name}_model_judge.pickle")
    
    def merge(self) -> pd.DataFrame:
        self.eval_df = self.eval_df.merge(self.response[['model_judge', f'{self.solution1_col}_is_correct', f'{self.solution2_col}_is_correct']], left_index=True, right_index=True)

        return self.eval_df
    
if __name__ == "__main__":
    """
    Example usage:
    python distractor-judge.py \
      --input_filepath ./results/allmath/inject_distractor/R1-Distill-Qwen-32B.pickle \
      --output_dir ./results/allmath/inject_distractor/model_judge \
      --nick_name R1-Distill-Qwen-32B

    python distractor-judge.py \
      --input_dir ./results/allmath/inject_distractor \
      --output_dir ./results/allmath/inject_distractor/model_judge
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
    problem1_col = "problem"
    problem2_col = "distractor_problem"
    solution1_col = "solution"
    solution2_col = "distractor_solution"
    response_col = "post_distraction_response"
    pred_col = "pred"
    
    if args.input_filepath:
        input_df = pd.read_pickle(args.input_filepath)
        judge_engine = ModelJudge(
            input_df=input_df,
            problem1_col=problem1_col,
            problem2_col=problem2_col,
            solution1_col=solution1_col,
            solution2_col=solution2_col,
            response_col=response_col,
            pred_col=pred_col,
            output_dir=args.output_dir,
            nick_name=args.nick_name,
        )
        judge_engine.run(overwrite=args.overwrite)
        result_df = judge_engine.merge()
        result_df.to_pickle(os.path.join(args.output_dir, f"{args.nick_name}_result.pickle"))
        
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

                    if "pred" not in input_df.columns:
                        print(f"Skipping {nick_name} because it doesn't have pred column")
                        continue
                    
                    judge_engine = ModelJudge(
                        input_df=input_df,
                        problem1_col=problem1_col,
                        problem2_col=problem2_col,
                        solution1_col=solution1_col,
                        solution2_col=solution2_col,
                        response_col=response_col,
                        pred_col=pred_col,
                        output_dir=args.output_dir,
                        nick_name=nick_name,
                    )
                    print(f"Starting to judge {nick_name}")
                    judge_engine.run(overwrite=args.overwrite)
                    result_df = judge_engine.merge()
                    result_df.to_pickle(os.path.join(args.output_dir, f"{nick_name}_result.pickle"))
                    print(f"Finished judging {nick_name}")
                    finished.append(fname)
            
            print(f"Waiting for 60 seconds before checking again")
            time.sleep(60)