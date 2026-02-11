# Inception: Iterative Reasoning Injection Attack on LLMs

## Overview

Inception is an adversarial attack technique targeting reasoning-enabled large language models (LLMs). The core idea is to inject partial reasoning traces into a target model's chain-of-thought, progressively steering it past its safety guardrails. An **architect model** (a small, locally-run reasoning LLM) generates adversarial reasoning seeds, which are injected into the target model's `<think>` block via the completions API. A **refusal classifier** detects and removes safety-triggered refusals from the target's output, and the cleaned reasoning is fed back for subsequent iterations. Samples that produce complete responses without triggering refusals "graduate" to the results set.

The technique is evaluated on WMDP (Weapons of Mass Destruction Proliferation) benchmark inquiries across biochemistry, chemistry, and cybersecurity domains.

## Architecture

```
                         ┌──────────────────┐
                         │  WMDP Inquiries   │
                         │  (HuggingFace)    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │    InceptionEngine (main)   │
                    │                             │
              ┌─────┤  for i in max_iterations:   ├─────┐
              │     └─────────────────────────────┘     │
              ▼                                         ▼
   ┌──────────────────┐                    ┌─────────────────────┐
   │ Architect Engine  │                    │  Target Engine      │
   │ (local vLLM)      │───reasoning───────▶│  (remote API)       │
   │ OpenThinker3-7B   │                    │  DeepSeek/Qwen/etc  │
   └──────────────────┘                    └──────────┬──────────┘
                                                      │
                                           ┌──────────▼──────────┐
                                           │ Refusal Classifier   │
                                           │ (FastText)           │
                                           │                      │
                                           │ refusal? ──yes──▶ chop│
                                           │          ──no───▶ keep│
                                           └──────────┬──────────┘
                                                      │
                                           ┌──────────▼──────────┐
                                           │ Graduation Check     │
                                           │ tokens >= threshold? │
                                           │  yes ──▶ results     │
                                           │  no  ──▶ next iter   │
                                           └─────────────────────┘
```

## How It Works

### Iteration Loop (`src/main.py`)

Each sample (inquiry) goes through up to `max_iterations` (default 3) rounds:

1. **Architect generates reasoning** (`architect_engine_cont`): The architect model (OpenThinker3-7B, run locally via vLLM) receives the inquiry plus any accumulated reasoning and generates a continuation. On iteration 0 it produces up to `architect_initial_max_tokens` (default 256) tokens; on subsequent iterations, `architect_reiterate_max_tokens` (default 64) tokens. The architect's output after `</think>` is extracted and appended to the reasoning chain.

2. **Target model continues** (`target_engine_cont`): The accumulated reasoning is formatted into the target model's specific chat template and sent via the **completions API** (not chat completions), so the model continues the reasoning directly rather than responding to a message. The prompt looks like: `<chat_template>{inquiry}<assistant><think>{accumulated_reasoning}`. The target model generates up to 32,768 tokens.

3. **Refusal detection and removal** (`remove_refusal`): The target's response is split into sentences (NLTK punkt tokenizer). Each sentence is classified using a FastText binary classifier with a 3-sentence sliding window for context. If refusal is detected (label 1 with confidence >= 0.5), everything from that sentence onward is chopped off.

4. **Graduation** (`concatenate_reasoning`): If the target's response had no refusal AND the total reasoning length exceeds `min_reasoning_tokens` (default 2048 tokens), the sample graduates to results. If reasoning is too short, the last paragraph is removed and the sample continues to the next iteration. If refusal was detected, the chopped response is appended and the sample continues.

### Key Mechanism: Completions Mode

The attack uses the raw **completions** endpoint (not chat completions). This allows injecting text directly into the model's thinking block, so the model sees the adversarial reasoning as its own chain-of-thought rather than as user input.

### Model-Specific Chat Templates

Each target model has a custom prompt template (`TARGET_MODEL_CHAT_TEMPLATE` in `src/main.py`) that matches its native format:

| Model | Template Pattern |
|-------|-----------------|
| DeepSeek-V3.2 | `<｜begin▁of▁sentence｜><｜User｜>{inquiry}<｜Assistant｜><think>{reasoning}` |
| Qwen3-235B | `<\|im_start\|>user\n{inquiry}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n{reasoning}` |
| Kimi-K2 | `<\|im_system\|>system<\|im_middle\|>...<\|im_user\|>user<\|im_middle\|>{inquiry}<\|im_end\|><\|im_assistant\|>assistant<\|im_middle\|><think> {reasoning}` |
| GLM-4.6 | `[gMASK]<sop><\|user\|>\n{inquiry}\n<think>{reasoning}` |

## Dataset

### Source: WMDP Benchmark

The base questions come from the [WMDP (Weapons of Mass Destruction Proliferation)](https://huggingface.co/datasets/cais/wmdp) dataset, covering:

- **wmdp-bio**: Biochemistry / pathogen / bioweapon questions
- **wmdp-chem**: Chemical weapons / poisoning questions
- **wmdp-cyber**: Cybersecurity exploitation questions (not currently used in the main pipeline)

### Inquiry Generation (`prepare_dataset/generate_wmdp_inquiries.py`)

The raw WMDP multiple-choice questions are transformed into realistic adversarial inquiries using an LLM (Qwen3-235B-A22B-Instruct):

1. Load WMDP-Bio and WMDP-Chem questions (sampled per category)
2. For each question, an LLM generates:
   - A **situation** analysis (user background and malicious intent) within `<situation>` tags
   - A **self-contained inquiry** (one sentence, explicit, no references to the original question) within `<inquiry>` tags
3. Few-shot examples guide the generation style
4. Results are pushed to HuggingFace (e.g., `aochongoliverli/wmdp_biochem_inquiries_800`)

The `GenerateWMDPShotExamples` subclass generates additional few-shot examples from questions not already in the main inquiry dataset.

## Refusal Classifier

### Training Pipeline

The FastText refusal classifier is trained in two stages:

#### Stage 1: Generate Labels (`fasttext_models/train/prepare_data.py`)

1. Collect first-iteration reasoning traces from previous inception runs (pickle files from multiple target models)
2. Extract `<think>` content, split into sentences, create 3-sentence sliding windows
3. Use Qwen3-4B (local, via vLLM) as a safety classifier to label each window:
   - **Label 0**: No safety alert -- model continues reasoning normally
   - **Label 1**: Safety alert -- model triggers refusal/warning/policy citation
4. The classifier prompt distinguishes between models *mentioning* safety as technical knowledge vs. models *refusing* to continue

#### Stage 2: Train FastText (`fasttext_models/train/train_fasttext.py`)

1. Load labeled data (heavily imbalanced: ~92% label 0, ~8% label 1)
2. Oversample minority class to target ratio (default 50%)
3. Train FastText supervised classifier:
   - Learning rate: 0.5, Epochs: 50, Word n-grams: 3, Embedding dim: 1000
   - Loss: softmax, Min word count: 3
4. Evaluate with precision/recall/F1, find optimal classification threshold
5. Output: `fasttext_models/refusal_model.bin`

### Inference (`src/refusal_model.py`)

- `classify_reasoning_trace()`: Splits text into sentences, classifies each with 3-sentence sliding window context. Returns index of first refusal sentence (mode `firstonly`) or full DataFrame (mode `all`).
- `chop_at_refusal()`: Truncates text at the character position of the refusal sentence, preserving original formatting.

## Target Models

Configured in `config/target_models.yaml`:

| Model | Provider |
|-------|----------|
| DeepSeek-V3.2 | DeepInfra |
| Qwen3-235B-A22B-Thinking-2507 | DeepInfra |
| Qwen3-Next-80B-A3B-Thinking | Together.AI |
| Kimi-K2-Thinking | DeepInfra |
| GLM-4.6 | DeepInfra |

## Architect Model

Configured in `config/architect_models.yaml`:

- **OpenThinker3-7B** (`open-thoughts/OpenThinker3-7B`): A Qwen-based reasoning model run locally via vLLM. Generates adversarial reasoning seeds.

## Project Structure

```
inception/
├── config/
│   ├── target_models.yaml          # Target model definitions and API providers
│   └── architect_models.yaml       # Architect model definition
├── core/
│   ├── .env                        # API keys (DEEPINFRA_API_KEY, etc.)
│   ├── llm_engine.py               # vLLM wrapper (OpenLMEngine) with Ray data parallelism
│   ├── openaiapi.py                # OpenAI-compatible API client (multi-provider)
│   └── openai_engine.py            # Batch processing engine (JSONL prep, parallel execution, caching)
├── src/
│   ├── main.py                     # InceptionEngine -- main attack pipeline
│   ├── refusal_model.py            # Refusal detection (classify + chop)
│   ├── benchmark.py                # Baseline evaluation (no jailbreaking)
│   └── utils/
│       └── chunk.py                # Text chunking utilities
├── prepare_dataset/
│   └── generate_wmdp_inquiries.py  # WMDP question -> adversarial inquiry generation
├── fasttext_models/
│   ├── refusal_model.bin           # Trained FastText refusal classifier
│   └── train/
│       ├── prepare_data.py         # Label reasoning traces for refusal training
│       └── train_fasttext.py       # Train FastText classifier
├── scripts/
│   └── scheduler.py                # Multi-GPU job scheduler
├── notebook/inception/
│   ├── prototype.ipynb             # Prototyping and prompt testing
│   └── wmd_eval.ipynb              # Evaluation notebook
├── pyproject.toml                  # Dependencies (Python 3.12, vLLM, Ray, FastText, etc.)
└── README.md
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 3 | Number of architect-target-refusal loops |
| `architect_initial_max_tokens` | 256 | Architect tokens on first iteration |
| `architect_reiterate_max_tokens` | 64 | Architect tokens on subsequent iterations |
| `max_tokens` | 32768 | Target model max output tokens |
| `min_reasoning_tokens` | 2048 | Minimum reasoning length for graduation |
| `temperature` | 0.6 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling parameter |
| `refusal_threshold` | 0.5 | FastText confidence threshold for refusal detection |

## Installation

### Setup uv virtual environment with dependencies
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Configure API keys
In `core/`, create a `.env` file with your API keys:
```
DEEPINFRA_API_KEY=your_key_here
TOGETHERAI_API_KEY=your_key_here
```

### Ensure NVIDIA GPU is available
```bash
uv run -- python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Run the inception pipeline
```bash
python src/main.py \
    --target_model_name "deepseek-ai/DeepSeek-V3.2" \
    --target_nick_name "DeepSeek-V3.2" \
    --architect_model_name "open-thoughts/OpenThinker3-7B" \
    --dataset_name "aochongoliverli/wmdp_biochem_inquiries_800" \
    --split_name "test" \
    --results_dir ./results \
    --max_iterations 3 \
    --architect_initial_max_tokens 256 \
    --architect_reiterate_max_tokens 64 \
    --min_reasoning_tokens 2048 \
    --max_tokens 32768 \
    --temperature 0.6 \
    --client_name "deepinfra" \
    --overwrite
```

### Run baseline benchmark (no jailbreaking)
```bash
python src/benchmark.py \
    --model_name "deepseek-ai/DeepSeek-V3.2" \
    --nick_name "DeepSeek-V3.2" \
    --tokenizer_name "deepseek-ai/DeepSeek-V3.2" \
    --dataset_name_or_path "aochongoliverli/wmdp_biochem_inquiries_800" \
    --split_name "test" \
    --client_name "deepinfra"
```

### Generate WMDP inquiries
```bash
python prepare_dataset/generate_wmdp_inquiries.py
```

### Train refusal classifier
```bash
# Step 1: Generate labeled training data
python fasttext_models/train/prepare_data.py

# Step 2: Train FastText model
python fasttext_models/train/train_fasttext.py
```

### Multi-GPU scheduling
```bash
python scripts/scheduler.py \
    --models-yaml config/target_models.yaml \
    --benchmark-script your_script.sh \
    --poll-interval 5
```

## Output

Results are saved as pickle files under `results/inception/max_iterations_{N}/`:
- `{target_nick_name}.pickle`: DataFrame containing graduated samples with columns for inquiry, accumulated reasoning, and per-iteration architect/target outputs.
- API response caches are stored under `results/inception/max_iterations_{N}/api/`.
