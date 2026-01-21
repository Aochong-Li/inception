<h1 align="center">🧠 Core LLM Inference & Batch Engine</h1>

<p align="center">
  
Modular wrappers for LLM inference with:
- 🖥️ Local models via vLLM (`llm_engine.py`)
- 🌐 API-based models (OpenAI, DeepSeek, TogetherAI) via `openaiapi.py`
- 📦 Prompt orchestration & batch execution (`openai_engine.py`)

---

## llm_engine.py — vLLM Wrapper

```python
from llm_engine import OpenLMEngine, ModelConfig

config = ModelConfig(model_name="your_model_name")
engine = OpenLMEngine(config)
output = engine.generate(["Your prompt here"])
```


## openai_engine.py — Prompt + Batch Runner
```python
from openai_engine import OpenAI_Engine

engine = OpenAI_Engine(input_df, prompt_template="{question}")
engine.run_model()
results = engine.retrieve_outputs()
```

