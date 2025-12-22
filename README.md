# Inception: LLM Safety Evaluation Pipeline

Safety evaluation pipeline for LLM responses using batch processing and parallel API calls.

## Installation

```bash

# Setup environment
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Configure API keys
echo "OPENAI_API_KEY=your_key_here" > core/.env
```

## Usage

```bash
cd evaluation
python safety-judge.py \
  --input_filepath /path/to/data.pickle \
  --inquiry_col inquiry \
  --reasoning_trace_col incepted_response \
  --overwrite
```

See `python safety-judge.py --help` for all options.

## Project Structure

- `core/`: LLM API clients and batch processing engine
- `evaluation/`: Safety evaluation pipeline
