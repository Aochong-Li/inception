# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES="1"
MAX_TOKENS="4096"

EPOCHS_LIST=(4 9 14 19)
for EPOCH in "${EPOCHS_LIST[@]}";do
    echo "======== Evaluating checkpoint at epoch: ${EPOCH} ========"
    MODEL_NAME_OR_PATH="/share/goyal/lio/reasoning/model/math_12k/Qwen2.5-Math-1.5B_grpo_math_12k_rollout_8_max_length_3000/actor/epoch_${EPOCH}"
    OUTPUT_DIR="/share/goyal/lio/reasoning/eval/benchmarks/r1/Qwen2.5-Math-1.5B_grpo_math_12k_rollout_8_max_length_3000/epoch_${EPOCH}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
done