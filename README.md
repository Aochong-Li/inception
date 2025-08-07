## Installation

```bash
uv venv rlvr_eval --python 3.12 rlvr_eval
source rlvr_eval/bin/activate

uv pip install -r requirements.txt
uv pip install latex2sympy2==1.9.1 --no-deps

```

## Run all models

```bash
bash scripts/run_all_models.sh
```