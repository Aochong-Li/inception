# Evaluation Scripts

Scripts for running and analyzing evaluations. Run from the project root, e.g.:

```bash
uv run python evaluation/scripts/run_three_model_eval.py --eval_model deepseek-chat --client_name deepseek
uv run python evaluation/scripts/verify_eval_completeness.py
```

## Run scripts (execute evaluations)

- `run_three_model_eval.py` – Three-model pipeline (DeepSeek, Qwen3, GLM)
- `run_five_judges.py` – Five-judge evaluation (OpenRouter)
- `run_five_judges_eval.py` – Five-judge eval using existing preprocessed data
- `run_reeval_three_models.py` – Re-evaluate failed rows for three models
- `run_strongreject_three_models.py` – StrongREJECT evaluation on three models
- `run_strongreject_evals.py` – StrongREJECT for all models in evaluation/inception_data/max_iterations_5/think
- `run_batch_eval.py` – Batch evaluation for all models
- `run_ablation_eval.py` – Ablation experiments (DeepSeek V3.2)
- `reeval_failed.py` – Re-evaluate failed safety evaluations across models

## Analysis scripts

- `verify_eval_completeness.py` – Check for missing/failed rows in archived results
- `analyze_five_judges.py` – Analyze five-judge evaluation outputs
- `aggregate_results.py` – Aggregate per-model results into cumulative summaries
- `aggregate_benchmark_results.py` – Aggregate think-vs-instruct benchmark results
- `diagnose_parse_failures.py` – Diagnose DeepSeek judge parse failures
- `run_deepseek_truncated_reeval.sh` – Truncated-only reeval for DeepSeek judge
- `inspect_results.ipynb` – Jupyter notebook for inspecting results

## Pipeline modules (in evaluation/, not here)

- `safety-judge.py` – SafetyEvaluator
- `danger_score.py` – Danger score computation
- `preprocess_results.py` – Preprocessing utilities
- `prompts/` – Prompt loading
