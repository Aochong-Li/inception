## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync

```

## In /core create a .env file containing DEEPINFRA_API_KEY

## Ensure NVIDIA GPU is available
'''bash
uv run -- python -c "import torch; print(torch.cuda.is_available())"
```

## Run pipeline
```bash
uv run pipeline/pipeline.py
```
