#!/usr/bin/env python3
"""Preprocess inception results for SafetyEvaluator."""
import argparse
import pandas as pd
from pathlib import Path


def get_full_trace(row):
    """Reconstruct full reasoning trace by concatenating all iterations in order.

    Order: architect_0 + target_0 + architect_1 + target_1 + ... + architect_4 + target_4
    Skips null/empty iterations. No separators - text is already continuous.

    Handles both column naming conventions:
    - architect_iteration_0, target_iteration_0 (preferred)
    - architect_0, target_0 (fallback)

    Continues as far into the inception iterations as possible, stopping when
    iterations become empty or the inception procedure ended.
    """
    parts = []
    for i in range(5):
        # Try both naming conventions (prefer longer names first)
        arch_col_long = f'architect_iteration_{i}'
        arch_col_short = f'architect_{i}'
        tgt_col_long = f'target_iteration_{i}'
        tgt_col_short = f'target_{i}'

        # Determine which architect column exists (prefer long name)
        arch_col = arch_col_long if arch_col_long in row.index else (
            arch_col_short if arch_col_short in row.index else None)
        tgt_col = tgt_col_long if tgt_col_long in row.index else (
            tgt_col_short if tgt_col_short in row.index else None)

        # Add architect iteration if it exists and is not empty
        if arch_col is not None:
            arch_val = row[arch_col]
            if pd.notna(arch_val):
                arch_str = str(arch_val).strip()
                if arch_str:  # Only add if not empty after stripping
                    # Use original value, not stripped
                    parts.append(str(arch_val))

        # Add target iteration if it exists and is not empty
        if tgt_col is not None:
            tgt_val = row[tgt_col]
            if pd.notna(tgt_val):
                tgt_str = str(tgt_val).strip()
                if tgt_str:  # Only add if not empty after stripping
                    # Use original value, not stripped
                    parts.append(str(tgt_val))

    return ''.join(parts) if parts else None


def get_latest_iteration(row, iteration_cols):
    """Return the value from the latest non-null iteration column."""
    for col in reversed(iteration_cols):
        if pd.notna(row[col]):
            return row[col]
    return None


def preprocess(input_path: str, output_path: str = None):
    """Add reasoning_traces column by reconstructing full trace from all iterations."""
    df = pd.read_pickle(input_path)

    # Reconstruct full trace from all iterations
    df['reasoning_traces'] = df.apply(get_full_trace, axis=1)
    print(f"Reconstructed full traces from all iterations")

    if output_path is None:
        output_path = str(Path(input_path).with_suffix('')) + \
            '_preprocessed.pickle'

    df.to_pickle(output_path)
    print(
        f"Saved: {output_path} ({len(df)} rows, {df['reasoning_traces'].notna().sum()} non-null)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess inception results for SafetyEvaluator")
    parser.add_argument("input_path", help="Path to input pickle file")
    parser.add_argument("--output_path", help="Path to output pickle file")
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)
