set -x

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="inception"

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_deepscaler/qwen25_3b_deepscaler.yaml
