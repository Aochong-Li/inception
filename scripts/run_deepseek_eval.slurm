#!/bin/bash
#SBATCH --job-name=deepseek-eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --constraint=avx2

cd /home/md2292/inception-eval
source rlvr_eval/bin/activate

# Unbuffered output so tail -f shows progress in real time
export PYTHONUNBUFFERED=1

python evaluation/scripts/run_deepseek_judge_full.py
