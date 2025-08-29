from huggingface_hub import HfApi, upload_folder
from datasets import Dataset, DatasetDict, load_from_disk
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

api = HfApi()
USERNAME = "aochongoliverli"

def push_model_to_hf(model_name: str, local_checkpoint_dir: str, username: str=USERNAME, new_system_prompt: str=None, upload_folder: bool=False):
    repo_id = f"{username}/{model_name}"
    
    # Check if repo already exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Repository {repo_id} already exists, skipping creation.")
    except Exception:
        print(f"Creating new repository {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)

    tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_dir)
    tokenizer.name_or_path = repo_id
    tokenizer.save_pretrained(local_checkpoint_dir)

    if new_system_prompt is not None:
        old_template = tokenizer.chat_template
        new_template = old_template.replace("You are a helpful assistant.", new_system_prompt)
        tokenizer.chat_template = new_template
        tokenizer.save_pretrained(local_checkpoint_dir)
    
    # Upload folder to the repository with progress tracking
    if upload_folder:
        upload_folder(
            folder_path=local_checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(local_checkpoint_dir)
        model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id, private=False)

def push_dataset_to_hf(dataset_name: str, local_dataset_dir: str, username: str=USERNAME):
    dataset = load_from_disk(local_dataset_dir)
    
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})
    elif isinstance(dataset, DatasetDict):
        pass
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
    
    repo_id = f"{username}/{dataset_name}"
    dataset.push_to_hub(repo_id, private=False)

def update_base_model_tokenizer(model_name: str, local_checkpoint_dir: str, username: str=USERNAME, replacement={}):
    repo_id = f"{username}/{model_name}"
    
    model = AutoModelForCausalLM.from_pretrained(local_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_dir)

    template = tokenizer.chat_template
    for old, new in replacement.items():
        template = template.replace(old, new)
    tokenizer.chat_template = template
    tokenizer.name_or_path = repo_id
    

    model.push_to_hub(repo_id, private=False)
    tokenizer.push_to_hub(repo_id, private=False)

if __name__ == "__main__":
    """
    Example usage:
    python push_to_hf/push.py --model_name "Qwen2.5-1.5B-DeepMath-level5-grpo-initial-checkpoint" --local_checkpoint_dir "outputs/deepmath/Qwen2.5-1.5B-DeepMath-level1-5-117k-sft-5epochs-5e-5lr/checkpoint-4570"
    python push_to_hf/push.py --model_name "Qwen2.5-1.5B-DeepMath-level5-grpo-cold-start-level1-4-40k" --local_checkpoint_dir "outputs/deepmath/Qwen2.5-1.5B-DeepMath-level1-4-40k-all_rollouts-sft-stage0/checkpoint-600"
    python push_to_hf/push.py --dataset_name "deepmath_4096" --local_dataset_dir "/share/goyal/lio/reasoning/data/deepmath_4096"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--local_checkpoint_dir", type=str, required=False)
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--local_dataset_dir", type=str, required=False)
    parser.add_argument("--task", type=str, required=True)
    
    args = parser.parse_args()

    replacement = {
        "You are a helpful assistant.": r"Please reason step by step. Think through the problem in depth before answering. Finally, put your final answer within \\boxed{}.",
        # "<|im_start|>assistant\\n": "<|im_start|>assistant\\n<think>"
    }
    if args.task == "push_local_model":
        push_model_to_hf(args.model_name, args.local_checkpoint_dir)
    elif args.task == "push_local_dataset":
        push_dataset_to_hf(args.dataset_name, args.local_dataset_dir)
    elif args.task == "update_base_model_tokenizer":
        update_base_model_tokenizer(args.model_name, args.local_checkpoint_dir, replacement=replacement)
    else:
        raise ValueError(f"Invalid task: {args.task}")
