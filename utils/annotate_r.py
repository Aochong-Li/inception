import os
import pandas as pd
from openai_engine import OpenAI_Engine
from utils.chunk_r import equal_chunk
from reward_score.math import *
import re
import argparse
import json
import ast

class ReasoningAnnotator:
    def __init__(self,
                 nick_name: str,
                 reasoning_col: str = "response",
                 results_dir: str = None,
                 granuality: int = None,
                 overwrite: bool = False
    ):
        self.nick_name = nick_name
        self.reasoning_col = reasoning_col
        self.results_dir = results_dir
        self.granuality = granuality
        self.overwrite = overwrite

        self.load_dataset()
    
    def load_dataset(self):
        if "math-500" in self.results_dir:
            self.df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle"))
            self.df = self.df[self.df["correct"] == 1]
            # To Save time, for each question we only take one response
            self.df = self.df.drop_duplicates(subset=["problem", "answer"]).reset_index(drop=True)
        elif "deepmath_7to9" in self.results_dir:
            self.think_df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle"))
            self.nothink_df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{self.nick_name}_nothinking.pickle"))

            # Find problems that think mode can always solve but nothink mode cannot
            think_correct_problems = self.think_df.groupby("problem")[["correct"]].all()
            think_correct_problems = set(think_correct_problems[think_correct_problems["correct"]].index)

            nothink_correct_problems = self.nothink_df.groupby("problem")[["correct"]].all()
            nothink_correct_problems = set(nothink_correct_problems[nothink_correct_problems["correct"]].index)

            problems = think_correct_problems - nothink_correct_problems
            self.df = self.think_df[self.think_df["problem"].isin(problems)]
            
            self.df = self.df.drop_duplicates(subset=["problem", "solution"]).reset_index(drop=True)
    
    def locate_1st_attempt(self):
        raise NotImplementedError("This function is not implemented yet")
        def process_response(row):
            response = row[self.reasoning_col]
            problem = row["problem"]
            answer = row["answer"]
            
            if "</think>" not in response:
                return None, None
            reasoning, _ = response.split("</think>")

            chunks = equal_chunk(reasoning, self.granuality)
            prompt = """<instruction>
You are given a sequence of reasoning steps, divided into chunks, showing how a model solves a problem. Your task is to precisely identify the first potential answer that model has reached. The first candidate does NOT NEED to be correct. But, the model should either explicitly mention that it can be the answer or imply that it thinks that it can be the answer before any checking or verification. The first answer candidate does NOT need to be confirmed, verified, or boxed. 
    
You should scan through the chunks from the start, ensuring that the candidate that you detect is indeed the earliest attempt:
1.    Skim over earlier chunks that merely state the problem, set up notation, or provide just derivation.
2.    Dismiss later chunks that just verify or check the first answer candidate.
3.    The first answer candidate may be later checked or even changed. But you SHOULD NOT report later versions.
4.    After you locate and identify the first answer candidate, you MUST verify that it is the very first answer candidate by scanning backward carefully to ensure that earlier chunks do not any contain potential answer attempt. If mentioned before, update your answer. 

You should first spend time verbalizing your searching and verification process within <search></search> tags. And once identified, you should report the chunk number in <chunk_number></chunk_number> tags and put the candidate in <first_candidate></first_candidate> tags. Follow this response format:
<search>YOU_CAN_SEARCH_HERE</search>
<chunk_number>NUMBER</chunk_number>
<first_candidate>FIRST_CANDIDATE</first_candidate>

Regardless of its correctness, FIRST_CANDIDATE should be formatted like the final solution ({answer}) so that I can easily verify it matches the final answer. If no candidate is found (rare), report None for NUMBER and FIRST_CANDIDATE.
</instruction>

problem: {problem}
reasoning trace:
""".format(answer = answer, problem = problem)

            for idx, chunk in enumerate(chunks):
                if f"\\boxed{{{answer}}}" in chunk or idx == len(chunks) - 1:
                    continue
                
                prompt += f"\n>>>>>CHUNK {idx} START\n"
                prompt += chunk
                prompt += f"\n>>>>>CHUNK {idx} END\n"
            
            return chunks, prompt
        
        def extract_chunk_number_and_first_candidate(response: list[str]):
            response = response[0]

            chunk_number_pattern = r"<chunk_number>(.*?)</chunk_number>"
            first_candidate_pattern = r"<first_candidate>(.*?)</first_candidate>"
            
            chunk_number_match = re.search(chunk_number_pattern, response)
            first_candidate_match = re.search(first_candidate_pattern, response)

            if chunk_number_match and first_candidate_match:
                chunk_number = chunk_number_match.group(1).lower().replace("_", "").replace("chunk", "")
                first_candidate = first_candidate_match.group(1)
                try:
                    chunk_number = int(chunk_number)
                except:
                    chunk_number = None
                return chunk_number, first_candidate
            else:
                return None, None
        
        cache_output_dir = os.path.join(self.results_dir, f"annotated_reasoning")
        if not os.path.exists(cache_output_dir):
            os.makedirs(cache_output_dir)
        
        if not self.overwrite and os.path.exists(os.path.join(cache_output_dir, f"{self.nick_name}.pickle")):
            self.df = pd.read_pickle(os.path.join(cache_output_dir, f"{self.nick_name}.pickle")).sample(n=30, random_state=42)
            if "1st_attempt_chunk_index" in self.df.columns:
                return

        self.df["chunks"], self.df["1st_attempt_prompt"] = zip(*self.df.apply(process_response, axis = 1))
        self.df = self.df.dropna(subset=["chunks", "1st_attempt_prompt"])

        self.locate_attempt_engine = OpenAI_Engine(
            model="deepseek-chat",
            input_df=self.df,
            prompt_template="{prompt}",
            template_map={"prompt": "1st_attempt_prompt"},
            nick_name=f"locate_1st_attempt_{self.nick_name}",
            temperature=0.7,
            cache_filepath=os.path.join(cache_output_dir, f"{self.nick_name}_first_attempt.pickle"),
        )

        self.locate_attempt_engine.run_model(overwrite=self.overwrite)
        self.outputs = self.locate_attempt_engine.retrieve_outputs().rename(columns={"response": "1st_attempt_annotation_raw"})
        self.outputs["1st_attempt_chunk_index"], self.outputs["1st_attempt_first_candidate"] = zip(*self.outputs["1st_attempt_annotation_raw"].apply(extract_chunk_number_and_first_candidate))
        self.outputs.index = self.df.index
        self.df = self.df.merge(self.outputs, left_index=True, right_index=True)
        # TODO: remove test
        self.df.to_pickle(os.path.join(cache_output_dir, f"{self.nick_name}_test.pickle"))

    def locate_1st_answer(self):
        def process_response(row):
            response = row[self.reasoning_col]
            problem = row["problem"]
            answer = row["answer"]
            
            if "</think>" not in response:
                return None, None
            reasoning, _ = response.split("</think>")

            chunks = equal_chunk(reasoning, self.granuality)
            prompt = """<instruction> You are given a sequence of reasoning steps, divided into chunks, showing how a model solves a problem. Your task is to carefully search for and precisely identify the first chunk where the correct answer is reached, even if not yet confirmed, verified, or boxed. The answer may appear implicitly at first—your job is to confirm its first correct appearance. This task requires meticulous attention to traces.

You should scan through the chunks from the start, ensuring that the chunk you report is the earliest point:
1.    Skim over earlier chunks that merely state the problem, set up notation, or provide partial work.
2.    Dismiss later chunks that just verify or check the answer after identifying first appearance.
3.    The correct answer does NOT NEED to be explicitly confirmed or officially stated as the final answer, as long as it is derived as a result and it is correct.
4.    When you identify the chunk for the first time, you MUST verify it is the first appearance by scanning backward carefully to ensure that this result is not mentioned in any of the earlier chunks. If mentioned before, update your answer. 

You should first spend time verbalizing your searching and verification process within <search></search> tags. And once identified, you should report the chunk number in <first_answer_chunk_number></first_answer_chunk_number> tags. Follow this response format:
<search>YOU_CAN_SEARCH_HERE</search>
<first_answer_chunk_number>NUMBER</first_answer_chunk_number>
You must only mark the first correct arrival of the answer, not subsequent confirmations or corrections. If no chunk is found (very rare), report None for NUMBER.
</instruction>
problem: {problem}

answer: {answer}

reasoning trace:
""".format(problem=problem, answer = answer)

            for idx, chunk in enumerate(chunks):
                if f"\\boxed{{{answer}}}" in chunk or idx == len(chunks) - 1:
                    continue
                
                prompt += f"\n>>>>>CHUNK {idx} START\n"
                prompt += chunk
                prompt += f"\n>>>>>CHUNK {idx} END\n"
            
            return chunks, prompt
        
        def extract_chunk_number(response: list[str]):
            response = response[0]

            pattern = r"<first_answer_chunk_number>(.*?)</first_answer_chunk_number>"
            match = re.search(pattern, response)
            if match:
                chunk_number = match.group(1).lower().replace("_", "").replace("chunk", "")
                try:
                    return int(chunk_number)
                except:
                    return None
            else:
                return None

        cache_output_dir = os.path.join(self.results_dir, f"annotated_reasoning")
        if not os.path.exists(cache_output_dir):
            os.makedirs(cache_output_dir)
        
        if not self.overwrite and os.path.exists(os.path.join(cache_output_dir, f"{self.nick_name}.pickle")):
            self.df = pd.read_pickle(os.path.join(cache_output_dir, f"{self.nick_name}.pickle")).sample(n=30, random_state=42)
            if "1st_answer_chunk_index" in self.df.columns:
                return
        
        self.df["chunks"], self.df["1st_answer_prompt"] = zip(*self.df.apply(process_response, axis = 1))
        self.df = self.df.dropna(subset=["chunks", "1st_answer_prompt"])

        self.locate_answer_engine = OpenAI_Engine(
            model="deepseek-chat",
            input_df=self.df,
            prompt_template="{prompt}",
            template_map={"prompt": "1st_answer_prompt"},
            nick_name=f"locate_1st_answer_{self.nick_name}",
            temperature=0.7,
            cache_filepath=os.path.join(cache_output_dir, f"{self.nick_name}_first_answer.pickle"),
        )

        self.locate_answer_engine.run_model(overwrite=self.overwrite)
        self.outputs = self.locate_answer_engine.retrieve_outputs().rename(columns={"response": "1st_answer_annotation_raw"})
        self.outputs["1st_answer_chunk_index"] = self.outputs["1st_answer_annotation_raw"].apply(extract_chunk_number)
        self.outputs.index = self.df.index
        self.df = self.df.merge(self.outputs, left_index=True, right_index=True)
        
        self.df.to_pickle(os.path.join(cache_output_dir, f"{self.nick_name}.pickle"))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--reasoning_col", type=str, default="response")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--granuality", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    import ipdb; ipdb.set_trace()

    annotator = ReasoningAnnotator(
        **vars(args)
    )
    
    annotator.locate_1st_answer()