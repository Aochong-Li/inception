#!/usr/bin/env python3
import yaml
import subprocess
import time
import os
import argparse
from pathlib import Path
from collections import deque
import sys

def load_models(yaml_path):
    """Loads model configurations from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Models file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    models = []
    if 'models' in config and config['models']:
        for m in config['models']:
            if m and 'model_name' in m and 'nick_name' in m:
                models.append({
                    'model_name': m['model_name'],
                    'nick_name': m['nick_name'],
                    'gpu_num': m.get('gpu_num', 1)
                })
    return models

def get_available_gpus():
    """Determines the list of available GPU IDs."""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        val = os.environ['CUDA_VISIBLE_DEVICES']
        if not val:
            return []
        return [int(g) for g in val.split(',')]

    try:
        sp = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
        num_gpus = len(sp.stdout.strip().split('\n'))
        print(f"INFO: CUDA_VISIBLE_DEVICES not set. Detected {num_gpus} GPUs.", file=sys.stderr)
        return list(range(num_gpus))
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARNING: 'nvidia-smi -L' failed. Cannot detect GPUs.", file=sys.stderr)
        print("Please set CUDA_VISIBLE_DEVICES environment variable.", file=sys.stderr)
        return []

def main():
    """
    python scripts/scheduler.py \
        --models-yaml config/control_study.yaml \
        --benchmark-script scripts/math/stress-test/multi-gpu/benchmark_eval.sh \
        --poll-interval 5
    ps aux | grep benchmark_eval.sh | grep -v grep
    """
    parser = argparse.ArgumentParser(description="GPU job scheduler for model benchmarks.")
    parser.add_argument('--models-yaml', type=Path, default=Path('config/market_models.yaml'),
                        help="Path to the models YAML file.")
    parser.add_argument('--benchmark-script', type=Path, 
                        default=Path('scripts/math/stress-test/multi-gpu/benchmark_eval.sh'),
                        help="Path to the benchmark evaluation script.")
    parser.add_argument('--poll-interval', type=int, default=5,
                        help="Interval in seconds to poll for finished jobs and free GPUs.")

    args = parser.parse_args()

    if not args.models_yaml.is_file():
        print(f"Error: Models file not found at '{args.models_yaml}'", file=sys.stderr)
        sys.exit(1)
    
    if not args.benchmark_script.is_file() or not os.access(args.benchmark_script, os.X_OK):
        print(f"Error: Benchmark script not found or not executable at '{args.benchmark_script}'", file=sys.stderr)
        sys.exit(1)

    all_gpus = get_available_gpus()
    if not all_gpus:
        print("Error: No GPUs available. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scheduler starting with {len(all_gpus)} GPUs: {all_gpus}", file=sys.stderr)

    models = load_models(args.models_yaml)
    models.sort(key=lambda m: m.get('gpu_num', 1), reverse=True)
    models_to_run = deque(models)

    if not models_to_run:
        print("No models found to run in the YAML file.", file=sys.stderr)
        return
        
    running_jobs = {}  # pid -> {gpus: [0, 1], nick_name: "..."}
    free_gpus = set(all_gpus)

    try:
        while models_to_run or running_jobs:
            # 1. Check for finished jobs and reclaim GPUs
            finished_pids = []
            for pid, job_info in running_jobs.items():
                if pid.poll() is not None:
                    print(f"Job for '{job_info['nick_name']}' (PID: {pid.pid}) finished with code {pid.returncode}.", file=sys.stderr)
                    if pid.returncode != 0:
                        print(f"WARNING: Job for '{job_info['nick_name']}' failed. Check logs for details.", file=sys.stderr)
                    free_gpus.update(job_info['gpus'])
                    finished_pids.append(pid)
            
            for pid in finished_pids:
                del running_jobs[pid]

            # 2. Try to schedule new jobs from the queue
            scheduled_on_pass = False
            for _ in range(len(models_to_run)):
                model = models_to_run[0]
                needed_gpus = model['gpu_num']

                if len(free_gpus) >= needed_gpus:
                    models_to_run.popleft()  # Remove from queue
                    
                    gpus_to_assign = sorted(list(free_gpus))[:needed_gpus]
                    free_gpus.difference_update(gpus_to_assign)
                    gpu_ids_str = ",".join(map(str, gpus_to_assign))
                    
                    print(f"Launching '{model['nick_name']}' requesting {model['gpu_num']} GPU(s). Assigning: {gpu_ids_str}", file=sys.stderr)
                    
                    cmd = [
                        str(args.benchmark_script),
                        gpu_ids_str,
                        model['model_name'],
                        model['nick_name']
                    ]
                    
                    proc = subprocess.Popen(cmd)
                    running_jobs[proc] = {
                        'gpus': gpus_to_assign,
                        'nick_name': model['nick_name']
                    }
                    scheduled_on_pass = True
                else:
                    # Model needs more GPUs than are free, move to back of queue
                    models_to_run.rotate(-1)

            if not models_to_run and not running_jobs:
                break
            
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt. Terminating running jobs...", file=sys.stderr)
        for pid, job_info in running_jobs.items():
            print(f"  Terminating job for '{job_info['nick_name']}' (PID: {pid.pid})", file=sys.stderr)
            pid.terminate()
        # Wait for them to die
        for pid in running_jobs:
            pid.wait()
        print("All jobs terminated.", file=sys.stderr)

    print("✔ All models evaluated.", file=sys.stderr)

if __name__ == "__main__":
    main() 