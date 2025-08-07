# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES=0
MAX_TOKENS="4096"

MODEL_LIST=(
    # "aochongoliverli/Qwen2.5-Math-1.5B-DeepMath-Hard-SFT-continue-4890"
    # "aochongoliverli/Qwen2.5-Math-1.5B-DeepMath-Hard-SFT-continue-2934"
    "aochongoliverli/Qwen2.5-Math-1.5B-DeepMath-Hard-SFT-continue-3912"
)
for MODEL in "${MODEL_LIST[@]}";do
    echo "======== Evaluating checkpoint at epoch: ${MODEL} ========"
    OUTPUT_DIR="./results/DeepMath-Hard-SFT-continue/${MODEL}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL $MAX_TOKENS $OUTPUT_DIR
done

# MODEL_LIST=(
#     "aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_1800"
# )
# for MODEL in "${MODEL_LIST[@]}";do
#     echo "======== Evaluating checkpoint at epoch: ${MODEL} ========"
#     OUTPUT_DIR="./results/deepmath-hard-4096-rollout-8/${MODEL}"

#     mkdir -p $OUTPUT_DIR

#     bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL $MAX_TOKENS $OUTPUT_DIR
# done