# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-openthought"
export CUDA_VISIBLE_DEVICES="0,1"
MAX_TOKENS="16384"

GLOBAL_STEP_LIST=(1017)
for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}";do
    echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
    if [ $GLOBAL_STEP -eq 0 ]; then
        MODEL_NAME_OR_PATH="/share/goyal/lio/reasoning/model/openthought-16384/coldstart/Qwen2.5_1.5B_openthought-hard_coldstart_max_length_16384_bsz_32/checkpoint-62"
    else
        MODEL_NAME_OR_PATH="/share/goyal/lio/reasoning/model/openthought-16384/distill/Qwen2.5_1.5B_openthought-hard_coldstart_max_length_16384_bsz_32_checkpoint-62_openthought-hard_distill_max_length_16384_bsz_32/checkpoint-${GLOBAL_STEP}"
    fi        
    OUTPUT_DIR="/share/goyal/lio/reasoning/eval/benchmarks/distill/Qwen2.5_1.5B_openthought-hard_coldstart_max_length_16384_bsz_32_checkpoint-62_openthought-hard_distill_max_length_16384_bsz_32/global_step_${GLOBAL_STEP}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
done



####### pi1 #######
# PROJECT_NAME="verl_few_shot"
# EXPERIMENT_NAME="Qwen2.5-Math-1.5B-pi1_r128"
# GLOBAL_STEP_LIST=($(seq 20 20 2000))

# # # Loop through each step in the list
# for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
#     echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
#     MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}/actor
#     OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
#     bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
# done


####### DSR-sub #######
# PROJECT_NAME="verl_few_shot"
# EXPERIMENT_NAME="Qwen2.5-Math-1.5B-dsr_sub"
# GLOBAL_STEP_LIST=($(seq 20 20 2000))

# # # Loop through each step in the list
# for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
#     echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
#     MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}/actor
#     OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
#     bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
# done

