#!/usr/bin/env bash
set -euo pipefail

MODELS_YAML="./config/local_models.yaml"
GPUS=(0 1 2 3 4 5 6 7)   # edit if you want a subset

# -------- load (model,nick) pairs into an array --------
mapfile -t MODEL_LINES < <(
python - <<'PY' "$MODELS_YAML"
import sys, yaml
for m in yaml.safe_load(open(sys.argv[1]))['models']:
    print(m['model_name'], m['nick_name'])
PY
)

# -------- round-robin launch --------
next=0
for line in "${MODEL_LINES[@]}"; do
    read -r MODEL NICK <<<"$line"

    # if every GPU slot is busy with a job, wait for one to finish
    while (( $(jobs -pr | wc -l) >= ${#GPUS[@]} )); do
        sleep 1
    done

    # Find the next available GPU
    while true; do
        GPU=${GPUS[$next]}
        # Check if nvidia-smi reports any processes on the GPU
        if ! nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | grep -q .; then
            # GPU is free, break the find-gpu loop
            break
        fi
        # GPU is busy, try next one
        next=$(( (next + 1) % ${#GPUS[@]} ))
        # small sleep to avoid busy-looping too fast if all are busy
        sleep 1
    done
    
    # TODO: change to shell script of your choice 
    ./scripts/countdown/stress-test/multi-gpu/inject_distractor.sh "$GPU" "$MODEL" "$NICK" &

    # Start search for next job from the next GPU
    next=$(( (next + 1) % ${#GPUS[@]} ))
done

wait
echo "✔ all models evaluated."
