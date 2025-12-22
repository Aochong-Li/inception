# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import argparse
import pandas as pd
import os
import re

def math_verify_score(solution_str: str, ground_truth: str, response_str: str = None) -> float:
    from math_verify import verify, parse

    def multi_verify (solution_str: str, ground_truth:str):
        # math verify package
        try:
            mathv_pred = parse(solution_str)
            mathv_correct = verify(parse(str(ground_truth)), mathv_pred, float_rounding=6, numeric_precision=15)
        except Exception:
            mathv_correct = False
        
        # rule based verify
        try:
            rule_correct = compute_score(solution_str, ground_truth)
        except Exception:
            rule_correct = False
        
        # simple verify
        simple_correct = solution_str == ground_truth
    
        return mathv_correct or rule_correct or simple_correct

    if math_if_boxed(solution_str): 
        solution_str = solution_str
    else:
        solution_str = response_str
            
    if multi_verify(solution_str, ground_truth):
        return 1.0
    
    # Below are post-processing HACKs for MATH-500 dataset
    if re.fullmatch(r"\\text{.*?}", ground_truth):
        ground_truth = ground_truth.replace("\\text{", "").replace("}", "")
        if multi_verify(solution_str, ground_truth):
            return 1.0
    
        if re.match(r"\(([A-Z])\)", ground_truth):
            ground_truth = re.match(r"\(([A-Z])\)", ground_truth).group(1)
        if multi_verify(solution_str, ground_truth):
            return 1.0

    if "\\dfrac" in solution_str or "\\dfrac" in ground_truth:
        solution_str = solution_str.replace("\\dfrac", "\\frac")
        ground_truth = ground_truth.replace("\\dfrac", "\\frac")

    if "\\frac" in solution_str or "\\frac" in ground_truth:
        pattern = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
        replacement = r'\1/\2'
        solution_str = re.sub(pattern, replacement, solution_str)
        ground_truth = re.sub(pattern, replacement, ground_truth)
        if multi_verify(solution_str, ground_truth):
            return 1.0
        
    if ",\\!" in solution_str or ",\\!" in ground_truth:
        solution_str = solution_str.replace(",\\!", "")
        ground_truth = ground_truth.replace(",\\!", "")
        if multi_verify(solution_str, ground_truth):
            return 1.0
    
    if "\\in" in ground_truth:
        ground_truth = ground_truth.split("\\in")[-1].strip()
        if multi_verify(solution_str, ground_truth):
            return 1.0
            
    if "," in ground_truth and "," in solution_str:
        try:
            solution = remove_boxed(last_boxed_only_string(solution_str))
            solution = set(solution.split(","))
            ground_truth = set(ground_truth.split(","))
            if solution == ground_truth:
                return 1.0
        except:
            pass
    
    return 0.0

def math_if_boxed(solution_str) -> bool:
    try:
        _ = remove_boxed(last_boxed_only_string(solution_str))
        return True
    except Exception as e:
        return False

def compute_score(solution_str, ground_truth) -> float:
    retval = 0.
    try:
        solution_string_in_last_boxed = last_boxed_only_string(solution_str)
        ground_truth_string_in_last_boxed = last_boxed_only_string(ground_truth)

        if solution_string_in_last_boxed is not None:
            answer = remove_boxed(solution_string_in_last_boxed)
        if ground_truth_string_in_last_boxed is not None:
            ground_truth = remove_boxed(ground_truth_string_in_last_boxed)
            
        if is_equiv(answer, ground_truth):
            retval = 1.
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

if __name__ == "__main__":
    """
    conda activate zero
    python reward_score/math500.py --file_path ./results/math500amc23/benchmark
    python reward_score/math500.py --input_dir ./results/allmath/benchmark --overwrite
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input_dir if args.input_dir else args.file_path

    pred_col = "pred"
    gt_col = "ground_truth"
    if "benchmark" in source:
        response_col = "response"
    elif "corrupt" in source:
        response_col = "post_corruption_response"
    
    if args.file_path:
        df = pd.read_pickle(args.file_path)
        df["model_is_correct"] = df.apply(lambda x: math_verify_score(x["pred"], x["gt"], x["response"]), axis=1)
        df.to_pickle(args.file_path)

    else:
        for fname in os.listdir(args.input_dir):
            if fname.endswith(".pickle"):
                print("Evaluating {}".format(fname))
                df = pd.read_pickle(os.path.join(args.input_dir, fname))
                
                if ("model_is_correct" in df.columns 
                    or "original_correct" in df.columns 
                    or "distractor_correct" in df.columns) and not args.overwrite:
                    print("Skipping {} because it already has model_is_correct column".format(fname))
                    continue
                
                if "distract" in source:
                    df["original_correct"] = df.apply(lambda x: math_verify_score(x["pred"], x["solution"], x["post_distraction_response"]), axis=1)
                    df["distractor_correct"] = df.apply(lambda x: math_verify_score(x["pred"], x["distractor_solution"], x["post_distraction_response"]), axis=1)
                else:
                    df["model_is_correct"] = df.apply(lambda x: math_verify_score(x[pred_col], x[gt_col], x[response_col]), axis=1)

                df.to_pickle(os.path.join(args.input_dir, fname))
