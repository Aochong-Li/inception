set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="deepmath"

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25_deepmath_sft_stage0.yaml

## Push checkpoints to huggingface
# python3 push_to_hf/experiments.py --project_name deepmath --run_name Qwen2.5-1.5B-DeepMath-level1-4-14k-sft-stage0
