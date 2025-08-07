# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES=0,1
MAX_TOKENS="4096"

STEP_LIST=(489 2445 4890)
for STEP in "${STEP_LIST[@]}";do
    MODEL_NAME_OR_PATH="/home/al2644/research/codebase/reasoning/rlvr/sft/LLaMA-Factory/outputs/llamafactory-sft/Qwen2.5-Math-1.5B-DeepMath-Hard-SFT/checkpoint-${STEP}"
    OUTPUT_DIR="./results/deepmath-hard-4096-sft/Qwen2.5-Math-1.5B-DeepMath-Hard-SFT-checkpoint-${STEP}"
    echo "======== Evaluating checkpoint at Global Step: ${STEP} ========"
    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
    echo "======== Saving results to: ${OUTPUT_DIR} ========"

done
3705123