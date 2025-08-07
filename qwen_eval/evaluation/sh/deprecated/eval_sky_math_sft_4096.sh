# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES="0,1"
MAX_TOKENS="4096"

GLOBAL_STEP_LIST=(64 640 1270)
for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}";do
    echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
    MODEL_NAME_OR_PATH="/share/goyal/lio/reasoning/model/sky_math8k/sft/Qwen2.5_Math_1.5B_sky_math8k_max_length_4096_bsz_32_epochs_10/checkpoint-${GLOBAL_STEP}"
    OUTPUT_DIR="/share/goyal/lio/reasoning/eval/benchmarks/sft/Qwen2.5_Math_1.5B_sky_math8k_max_length_4096_bsz_32_epochs_10/global_step_${GLOBAL_STEP}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
done
