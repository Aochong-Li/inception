set -x

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="math8k"

# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_qwq.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_am.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_qwen3.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_qwq_limo.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_limo/qwen25_3b_math8k_limo.yaml
