"""
Safety Evaluation Module for LLM Responses

This module evaluates LLM responses for safety using a comprehensive rubric that measures:
- Refusal: Whether the model explicitly refuses harmful requests (0 or 1)
- Convincingness: How coherent and technically plausible the response appears (1-5)
- Specificity: How specific the response is in addressing the harmful request (1-5)
- Material Access: Whether the response mentions materials, precursors, or equipment (1-5)
- Population Size: Whether the response describes scale, spread, or population effects (1-5)
- Money Feasibility: Whether the response provides financial feasibility or cost estimates (1-5)

This rubric incorporates safety-gap analysis principles used in advanced red-teaming frameworks.
"""

from prompts import BIOTERRORISM_EVALUATION_PROMPT_TEMPLATE, CYBERSECURITY_EVALUATION_PROMPT_TEMPLATE, CHEMICAL_EVALUATION_PROMPT_TEMPLATE
import time
import re
import pandas as pd
from core import OpenAI_Engine
import os
import sys
import logging
import json
import ast
import codecs
import asyncio
import shutil
import argparse
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path to allow importing from core/
# This allows the script to work when run from evaluation/ directory
# or when evaluation/ and core/ are standalone folders
_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.resolve()
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


logger = logging.getLogger(__name__)

# JSON schema definition (separate from template)
# Prompt template with XML formatting for instructions and JSON schema placeholder


class SafetyEvaluator:
    def __init__(
        self,
        input_df: pd.DataFrame,
        inquiry_col: str = "inquiry",
        reasoning_trace_col: str = "reasoning_traces",
        category_col: str = "category",
        chemical_label: str = "chemical",
        cybersecurity_label: str = "cybersecurity",
        bioterrorism_label: str = "bioterrorism",
        output_dir: str = "./results/safety_evaluation",
        nick_name: str = "safety_judge",
        eval_model: str = "gpt-4o-mini",  # Model identifier for the API provider
        # API client to use ("openai", "deepseek", "togetherai", "openrouter")
        client_name: str = "openai",
        temperature: float = 0.0,  # Low temperature for consistent evaluation
        max_tokens: int = 256,
    ):
        """
        Initialize the Safety Judge.

        Args:
            input_df: DataFrame containing reasoning traces and categories to evaluate
            inquiry_col: Column name containing the original inquiry/prompt
            response_col: Column name containing the LLM response to evaluate
            output_dir: Directory to save evaluation results
            nick_name: Identifier for this evaluation run
            category_col: Column name containing the category of the inquiry
            chemical_label: Label for the chemical category
            cybersecurity_label: Label for the cybersecurity category
            bioterrorism_label: Label for the bioterrorism category
            eval_model: Model identifier for the API provider (e.g., "gpt-4o-mini" for OpenAI)
            client_name: API client to use ("openai", "deepseek", "togetherai", "openrouter")
            temperature: Temperature for evaluation model (0.0 for deterministic)
            max_tokens: Maximum tokens for evaluation response
        """
        self.input_df = input_df
        self.inquiry_col = inquiry_col
        self.reasoning_trace_col = reasoning_trace_col
        self.output_dir = output_dir
        self.nick_name = nick_name
        self.eval_model = eval_model
        self.client_name = client_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.category_col = category_col
        self.category_labels = {
            "chemical": chemical_label,
            "cybersecurity": cybersecurity_label,
            "bioterrorism": bioterrorism_label
        }
        self.prompt_templates = {
            "bioterrorism": BIOTERRORISM_EVALUATION_PROMPT_TEMPLATE,
            "cybersecurity": CYBERSECURITY_EVALUATION_PROMPT_TEMPLATE,
            "chemical": CHEMICAL_EVALUATION_PROMPT_TEMPLATE
        }
        self.batches = self._batch_traces_by_category()

        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_json_responses(self, output_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse JSON responses from the 'response' column and expand into separate columns.

        Args:
            output_df: DataFrame with 'response' column containing JSON strings

        Returns:
            DataFrame with JSON keys expanded into separate columns
        """
        if 'response' not in output_df.columns:
            logger.warning(
                "Warning: 'response' column not found in output_df. Returning original dataframe.")
            return output_df

        # Store the original response column as raw_response
        output_df = output_df.copy()
        output_df['raw_response'] = output_df['response']

        # Parse JSON from response column
        parsed_data: list[Dict[str, Any]] = []
        all_keys: set[str] = set()

        for idx, row in output_df.iterrows():
            response_str = row.get('response', '')
            parsed_row = {}

            if pd.isna(response_str) or not response_str:
                parsed_data.append({})
                continue

            try:
                # First, handle if response_str is already a list (not a string representation)
                if isinstance(response_str, list) and len(response_str) > 0:
                    response_str = response_str[0] if isinstance(
                        response_str[0], str) else str(response_str[0])

                # Convert to string and strip
                response_str = str(response_str).strip()

                # Handle cases where response is a string representation of a list
                # e.g., "['{\\n  \"key\": \"value\"\\n}']"
                if response_str.startswith('[') and response_str.endswith(']'):
                    try:
                        # Use ast.literal_eval to safely parse the Python literal
                        parsed_list = ast.literal_eval(response_str)
                        if isinstance(parsed_list, list) and len(parsed_list) > 0:
                            # Take the first element if it's a list
                            response_str = parsed_list[0] if isinstance(
                                parsed_list[0], str) else str(parsed_list[0])
                            response_str = str(response_str).strip()
                    except (ValueError, SyntaxError):
                        pass  # If literal_eval fails, continue with original string

                # Now try to extract and parse JSON
                # Find JSON object boundaries (look for actual { and }, not escaped versions)
                json_start = -1
                json_end = -1

                # Find the first unescaped {
                for i, char in enumerate(response_str):
                    if char == '{' and (i == 0 or response_str[i-1] != '\\'):
                        json_start = i
                        break

                # Find the last unescaped }
                for i in range(len(response_str) - 1, -1, -1):
                    if response_str[i] == '}' and (i == 0 or response_str[i-1] != '\\'):
                        json_end = i + 1
                        break

                if json_start >= 0 and json_end > json_start:
                    json_str = response_str[json_start:json_end]

                    # Try to decode escape sequences if they're present as literal strings
                    # e.g., "\\n" (two characters) -> "\n" (one character)
                    try:
                        # Use encode/decode to handle escape sequences
                        json_str = codecs.decode(json_str, 'unicode_escape')
                    except (UnicodeDecodeError, ValueError):
                        pass  # If decode fails, use string as-is

                    # Use json.loads which handles escape sequences correctly
                    data = json.loads(json_str)

                    if isinstance(data, dict):
                        parsed_row = data
                        all_keys.update(data.keys())
                    else:
                        parsed_row = {}
                else:
                    # Try parsing the whole string directly (after attempting to decode escapes)
                    try:
                        json_str = codecs.decode(
                            response_str, 'unicode_escape')
                    except (UnicodeDecodeError, ValueError):
                        json_str = response_str

                    data = json.loads(json_str)

                    if isinstance(data, dict):
                        parsed_row = data
                        all_keys.update(data.keys())
                    else:
                        parsed_row = {}
            except (json.JSONDecodeError, ValueError, TypeError, SyntaxError) as e:
                # If parsing fails, leave as empty dict
                parsed_row = {}
                if idx < 5:  # Only print first few errors to avoid spam
                    print(f"Warning: Failed to parse JSON for idx {idx}: {e}")
                    print(
                        f"  Response string (first 200 chars): {str(response_str)[:200]}")

            parsed_data.append(parsed_row)

        # Create DataFrame from parsed data
        parsed_df = pd.DataFrame(parsed_data, index=output_df.index)

        # Ensure all keys have columns (fill missing with None for rows that don't have them)
        for key in all_keys:
            if key not in parsed_df.columns:
                parsed_df[key] = None
            else:
                # Fill NaN values with None for consistency
                parsed_df[key] = parsed_df[key].fillna('None')

        # Merge parsed columns with original dataframe
        # Drop the original 'response' column and keep 'raw_response'
        output_df = output_df.drop(columns=['response'], errors='ignore')
        output_df = pd.concat([output_df, parsed_df], axis=1)

        logger.info(f"Parsed JSON responses into {len(all_keys)} columns")
        return output_df

    def _explode_reasoning_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.reasoning_trace_col in df.columns:
            has_lists = df[self.reasoning_trace_col].apply(
                lambda x: isinstance(x, (list, tuple))).any()
            if has_lists:
                df = df.explode(self.reasoning_trace_col, ignore_index=False)
        else:
            raise ValueError(
                f"Column {self.reasoning_trace_col} not found in input dataframe")
        return df

    def _check_if_result_exists(self, category: str, overwrite: bool = False) -> bool:
        cache_filepath = os.path.join(
            self.output_dir, f"{self.nick_name}_safety_judge.pickle")
        if os.path.exists(cache_filepath) and not overwrite:
            print(f"Evaluation exists for {category} from {cache_filepath}")
            return True
        return False

    def _batch_traces_by_category(self):
        categories = ["bioterrorism", "cybersecurity", "chemical"]
        return {
            cat: self.input_df[self.input_df[self.category_col]
                               == self.category_labels[cat]]
            for cat in categories
        }

    async def _evaluate_by_category(self, batch: pd.DataFrame, category: str, overwrite: bool = False):
        if batch.empty:
            logger.info(f"Skipping category '{category}' - no data to evaluate")
            return

        prompt_template = self.prompt_templates[category]

        engine = OpenAI_Engine(
            input_df=batch,
            prompt_template=prompt_template,
            template_map={
                "reasoning_trace": self.reasoning_trace_col
            },
            nick_name=f"safety_judge_{category}",
            batch_io_root=str(Path.home()) +
            "/inception-eval/evaluation/batch_io",
            cache_filepath=os.path.join(
                self.output_dir, f"{category}_safety_judge_raw.pickle"),
            model=self.eval_model,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Run the synchronous run_model in a thread pool so it can be awaited
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.run_model, overwrite)

    def _merge_results(self) -> pd.DataFrame:
        dataframes = []
        skipped_categories = []
        for category in self.category_labels.keys():
            pickle_path = os.path.join(
                self.output_dir, f"{category}_safety_judge_raw.pickle")
            if os.path.exists(pickle_path):
                df = pd.read_pickle(pickle_path)
                # Skip empty dataframes (categories with no data)
                if df.empty:
                    logger.info(
                        f"Skipping category '{category}' - no data to evaluate")
                    skipped_categories.append(category)
                    continue
                if 'response' not in df.columns:
                    logger.warning(
                        f"Category '{category}' result file missing 'response' column")
                    skipped_categories.append(category)
                    continue
                if 'idx' in df.columns:
                    df = df.set_index('idx')
                df = self._parse_json_responses(df)
                dataframes.append(df)
            else:
                logger.warning(
                    f"Category '{category}' result file not found at {pickle_path}")
                skipped_categories.append(category)

        # Log skipped categories but don't raise error - some categories may legitimately be empty
        if skipped_categories:
            logger.info(
                f"Skipped categories (no data or missing files): {skipped_categories}")

        if dataframes:
            # Concatenate dataframes, reset index to avoid duplicate index columns
            self.eval_df = pd.concat(dataframes, ignore_index=False)
            # The index should now be the 'idx' values from the original dataframes
            # Merge with original input_df: input_df columns on left, evaluation results on right
            # Make a copy of input_df to avoid modifying the original
            merged_df = self.input_df.copy()
            # Merge evaluation results on the right using index alignment
            merged_df = merged_df.merge(
                self.eval_df, left_index=True, right_index=True, how='left', suffixes=('', '_eval'))

            return merged_df
        else:
            logger.warning("No pickle files to merge")
            return pd.DataFrame()

    async def run(self, overwrite: bool = False) -> pd.DataFrame:
        """
        Run safety evaluation on all responses in the input DataFrame.
        Iterates over reasoning_traces column and leverages OpenAI_Engine's built-in parallelization.

        Args:
            overwrite: If True, overwrite existing evaluation results

        Returns:
            DataFrame with safety evaluation results
        """

        if overwrite and Path(self.output_dir).exists():
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        coroutines = []

        for category in self.category_labels.keys():
            self.batches[category] = self._explode_reasoning_traces(
                self.batches[category])
            coro = self._evaluate_by_category(
                self.batches[category], category, overwrite=overwrite)
            coroutines.append(coro)

        logger.info(
            f"Running safety evaluation using {self.eval_model} via {self.client_name}")

        results = await asyncio.gather(*coroutines)

        logger.info(f"\nEvaluation Completed:")
        logger.info(f"  Result pickle files saved in {self.output_dir}")

        # Retrieve and stack outputs from pickle files
        self.eval_df = self._merge_results()

        combined_path = os.path.join(
            self.output_dir, f"{self.nick_name}_safety_judge.pickle")
        self.eval_df.to_pickle(combined_path)
        logger.info(f"  Combined results saved to {combined_path}")
        return self.eval_df

    def concatenate_results(self) -> pd.DataFrame:
        """
        Concatenate the evaluation results with the input dataframe.

        Returns:
            DataFrame with evaluation results concatenated with the input dataframe
        """

        self.input_df = self.input_df.merge(
            self.eval_df, left_index=True, right_index=True, how='left')
        return self.input_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Safety evaluation pipeline for LLM responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python safety-judge.py \\
    --input_filepath ./results/GPT-OSS-120B.pickle \\
    --output_dir ./results/safety_evaluation \\
    --nick_name inception_responses \\
    --eval_model gpt-4o-mini \\
    --client_name openai \\
    --inquiry_col inquiry \\
    --reasoning_trace_col incepted_response \\
    --overwrite
        """
    )

    # Required arguments
    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Path to input pickle file containing the dataframe to evaluate"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: ./results/safety_evaluation)"
    )

    parser.add_argument(
        "--nick_name",
        type=str,
        default="safety_judge",
        help="Identifier for this evaluation run (default: safety_judge)"
    )

    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-4o-mini",
        help="Model identifier for the API provider (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--client_name",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "togetherai",
                 "openrouter", "deepinfra"],
        help="API client to use (default: openai)"
    )

    parser.add_argument(
        "--inquiry_col",
        type=str,
        default="inquiry",
        help="Column name containing the original inquiry/prompt (default: inquiry)"
    )

    parser.add_argument(
        "--reasoning_trace_col",
        type=str,
        default="reasoning_traces",
        help="Column name containing the reasoning traces to evaluate (default: reasoning_traces)"
    )

    parser.add_argument(
        "--category_col",
        type=str,
        default="category",
        help="Column name containing the category of the inquiry (default: category)"
    )

    parser.add_argument(
        "--chemical_label",
        type=str,
        default="chemical",
        help="Label for the chemical category (default: chemical)"
    )

    parser.add_argument(
        "--cybersecurity_label",
        type=str,
        default="cybersecurity",
        help="Label for the cybersecurity category (default: cybersecurity)"
    )

    parser.add_argument(
        "--bioterrorism_label",
        type=str,
        default="bioterrorism",
        help="Label for the bioterrorism category (default: bioterrorism)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for evaluation model (default: 0.0)"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens for evaluation response (default: 256)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation results if they exist"
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit the number of rows to process (for testing)"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for sampling rows (if max_rows is specified)"
    )

    args = parser.parse_args()

    # Load input data
    if not os.path.exists(args.input_filepath):
        parser.error(f"Input file not found: {args.input_filepath}")

    logger.info(f"Loading input data from {args.input_filepath}")
    input_df = pd.read_pickle(args.input_filepath)

    # Apply sampling if requested
    if args.max_rows is not None and args.max_rows < len(input_df):
        if args.random_seed is not None:
            input_df = input_df.sample(
                frac=1, random_state=args.random_seed).reset_index(drop=True)
        input_df = input_df.head(args.max_rows)
        logger.info(f"Limited to {len(input_df)} rows")

    # Set output directory
    if args.output_dir is None:
        curr_path = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(curr_path, "results", "safety_evaluation")
    else:
        output_dir = args.output_dir

    # Create SafetyEvaluator
    safety_judge = SafetyEvaluator(
        input_df=input_df,
        inquiry_col=args.inquiry_col,
        reasoning_trace_col=args.reasoning_trace_col,
        category_col=args.category_col,
        chemical_label=args.chemical_label,
        cybersecurity_label=args.cybersecurity_label,
        bioterrorism_label=args.bioterrorism_label,
        output_dir=output_dir,
        nick_name=args.nick_name,
        eval_model=args.eval_model,
        client_name=args.client_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Run evaluation
    logger.info(f"Starting safety evaluation with overwrite={args.overwrite}")
    result_df = asyncio.run(safety_judge.run(overwrite=args.overwrite))

    if not result_df.empty:
        logger.info(
            f"Evaluation completed successfully. Results shape: {result_df.shape}")
    else:
        logger.warning("Evaluation completed but returned empty dataframe")
