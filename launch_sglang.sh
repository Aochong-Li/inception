#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

cd /home/al2644/research/codebase/reasoning/inception
/home/al2644/research/codebase/reasoning/inception/deployment/vllm-judge/.venv/bin/python -u -m sglang.launch_server \
  --model-path /share/goyal/md2292/models/Qwen3.5-122B-A10B-FP8/ \
  --tp 2 --port 8000 --host 0.0.0.0 \
  --served-model-name Qwen3.5-122B-A10B-FP8 \
  --mem-fraction-static 0.80 \
  --context-length 16384 \
  --attention-backend triton
