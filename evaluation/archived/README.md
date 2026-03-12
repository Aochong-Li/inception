# Archived Evaluation Data

This directory contains archived evaluation results from previous runs:

- **eval_five_judges/** – Five-judge evaluation outputs (multiple judge models × target models)
- **eval_three_models_results/** – Three-model safety evaluation results (DeepSeek-V3.2, Qwen3-235B-A22B-Thinking-2507, GLM-4.6)
- **evaluation-results/** – General evaluation results
- **results/** – Ablation and other benchmark results
- **batch_io/** – Batch API input/output JSONL files from past runs

New evaluation runs write to `evaluation/eval_five_judges`, `evaluation/eval_three_models_results`, `evaluation/results`, etc. (at the evaluation/ level, not in archived/).
