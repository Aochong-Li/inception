import argparse
import json
import torch
import gc
import contextlib
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm.envs as envs
import re
from math import comb
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator_model_path', type=str, default="")
    parser.add_argument('--file_path', type=str, default="",
                        help="Path to the inference results file (jsonl)")
    parser.add_argument("--k", type=int, default=1,
                        help="Value of k for pass@k calculation")
    return parser.parse_args()


def parse_model_output(output):
    # Initialize the result dictionary
    result = {"analysis": None, "correctness": False}
    
    # 改进的正则表达式模式
    analysis_pattern = r"##\s*Analysis\s*(.*?)\s*##\s*Correctness"
    correctness_pattern = r"##\s*Correctness\s*(.*?)$"

    # Extract Analysis
    analysis_match = re.search(analysis_pattern, output, re.DOTALL)
    if analysis_match:
        result["analysis"] = analysis_match.group(1).strip()

    # Extract Correctness
    correctness_match = re.search(correctness_pattern, output, re.DOTALL)
    if correctness_match:
        correctness_value = correctness_match.group(1).strip()
        result["correctness"] = "correct" in correctness_value.lower()
            
    return result["analysis"], result["correctness"]


def evaluate_responses(evaluator_model, evaluator_tokenizer, examples):
    prompts = []
    index_map = []
    
    for i, example in enumerate(examples):
        question = example["question"]
        gold_answer = example["gold_answer"]
        # gen_answers = example["generated_answers"]
        gen_answers = example["generated_responses"]
        
        for j, final_ans in enumerate(gen_answers):
            messages = [
                {"role": "system", "content": """You are an experienced examiner who evaluates whether a student's answer to a given question is correct. 
Your task is to determine if the student's final answer matches the standard answer provided, based solely on correctness and the question's specific requirements. 
Do not perform any additional calculations or reinterpret the question. Simply compare the student's answer to the standard answer to determine if it satisfies the question's requirements.

Focus strictly on:
1. Understanding the exact requirement of the question.
2. Comparing the student's final answer directly to the provided standard answer.
3. Your task is not to solve the problem but to determine whether the student's answer is correct based on the question's requirements. Avoid any unnecessary analysis, assumptions, or re-solving the problem.

Your response must include:
## Analysis
<Provide a brief and direct analysis that compares the student's answer to the standard answer>

## Correctness
<CORRECT/WRONG>"""},
                {"role": "user", "content": f"""Question: {question}

Standard Answer: {gold_answer}

Student's Final Answer: {final_ans}"""}
            ]
            
            prompt_text = evaluator_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            index_map.append((i, j))
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    eval_completions = evaluator_model.generate(prompts, sampling_params)
    
    num_prompts = len(prompts)
    all_evaluations = [[None]*len(example["generated_answers"]) for example in examples]

    for idx in range(num_prompts):
        i, j = index_map[idx]
        text = eval_completions[idx].outputs[0].text
        analysis, correctness = parse_model_output(text)
        all_evaluations[i][j] = {
            "analysis": analysis,
            "is_correct": correctness,
        }

    return all_evaluations


def calculate_metrics(evaluations, k):
    correct_cnt = 0
    pass_at_k_list = []
    
    for eval_list in evaluations:
        is_correct_list = [ev['is_correct'] for ev in eval_list]
        is_correct = any(is_correct_list)
        
        if is_correct:
            correct_cnt += 1
            
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(is_correct_list)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    # pass@k = 1 - C(n-c,k) / C(n,k)
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
        else:
            pass_at_k_list.append(1 if is_correct else 0)
    
    total = len(evaluations)
    accuracy = correct_cnt / total
    pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list) if pass_at_k_list else accuracy
    
    return {
        "accuracy": accuracy,
        f"pass@{k}": pass_at_k,
        "correct_count": correct_cnt,
        "total_count": total
    }


def main():
    args = parse_args()
    
    # Load inference results
    print("Loading inference results...")
    examples = []
    with open(args.file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
            
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    print(f"available_gpus: {available_gpus}")
    
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    
    print("Initializing evaluator model...")
    evaluator_model = LLM(
        model=args.evaluator_model_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
    )
    
    evaluator_tokenizer = AutoTokenizer.from_pretrained(
        args.evaluator_model_path, 
        trust_remote_code=True
    )
    
    print("Evaluating responses...")
    evaluations = evaluate_responses(
        evaluator_model, 
        evaluator_tokenizer, 
        examples
    )
    
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(evaluations, args.k)
    
    # Save results
    print("Saving results...")
    results = []
    for i, example in enumerate(examples):
        result = {
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "generated_responses": example["generated_responses"],  # 添加这行
            "generated_answers": example["generated_answers"],
            "evaluations": evaluations[i],
            "answers_correctness": [ev["is_correct"] for ev in evaluations[i]],
            "is_correct": any(ev["is_correct"] for ev in evaluations[i])
        }
        if "id" in example:
            result["id"] = example["id"]
        if "source" in example:
            result["source"] = example["source"]
        results.append(result)
    
    with open(args.file_path, 'w', encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})")
    print(f"Pass@{args.k}: {metrics[f'pass@{args.k}']:.4f}")


if __name__ == "__main__":
    main()