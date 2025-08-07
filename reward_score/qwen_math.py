import pandas as pd
from pebble import ProcessPool
from typing import Any
from concurrent.futures import TimeoutError

# from qwen_eval.eval_utils.utils.grader import math_equal
# from qwen_eval.eval_utils.utils.parser import extract_answer

from qwen25_eval.grader import math_equal
from qwen25_eval.parser import extract_answer

def compute_score(solution_str, ground_truth) -> tuple[int, Any]:
    extracted_gt = extract_answer(ground_truth, "math")
    extracted_answer = extract_answer(solution_str, "math")
    
    return math_equal(extracted_answer, extracted_gt)


if __name__ == "__main__":
    """
    conda activate zero
    python reward_score/qwen_math.py --file_path ./results/math8k/benchmark/QwQ-32Btrain.pickle
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    import pdb; pdb.set_trace()
    df = pd.read_pickle(args.file_path)
    from tqdm import tqdm
    tqdm.pandas()
    df["qwen_math_score"] = df.progress_apply(lambda x: compute_score(x["pred"], x["gt"]), axis=1)

    df.to_pickle(args.file_path)

def parse_response_dataframe(input_df: pd.DataFrame, gt_col: str, response_col: str) -> pd.DataFrame:
    df = input_df.copy()

    df['pred'] = df[response_col].apply(lambda x: x.split('</think>')[-1].strip() if '</think>' in x else x)
    df['gt'] = df[gt_col]
    return df
    
    '''
    math_equal is not a reliable evaluation. so we only extract answer and ground truth from the response.
    '''

    # params = [(idx, str(row['gt']), str(row['pred']), str(row['source'])) for idx, row in df.iterrows()]

    # from tqdm import tqdm

    # results = []
    # with ProcessPool(max_workers=num_proc) as pool:
    #     future = pool.map(eval_sample, params, timeout=120)
    #     iterator = future.result()

    #     for i in tqdm(range(len(params)), desc="Evaluating"):
    #         try:
    #             result = next(iterator)
    #             results.append(result)
    #         except TimeoutError:
    #             idx = params[i][0]
    #             print(f"Timeout for sample {idx}")
    #             results.append((idx, False))
    #         except Exception as e:
    #             idx = params[i][0]
    #             print(f"Error for sample {idx}: {e}")
    #             results.append((idx, False))

    # # results is a list of (idx, bool)
    # results_df = pd.DataFrame(results, columns=['idx', 'is_correct'])
    # results_df.set_index('idx', inplace=True)
    # df = df.merge(results_df, left_index=True, right_index=True)

    # return df