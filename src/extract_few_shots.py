import os
import pandas as pd
from datasets import load_dataset, Dataset

model_name = 'OpenThinker3-7B-Qwen'
local_dir = "/home/al2644/research/codebase/reasoning/inception/results/wmdp_shot_examples"
hf_dataset_name = "aochongoliverli/wmdp_shot_examples_256"

# Load the Hugging Face dataset and get the test split as a DataFrame
hf_ds = load_dataset(hf_dataset_name, split="test")
hf_df = hf_ds.to_pandas()

# Load local results DataFrame and retain only question and response columns
df = pd.read_pickle(os.path.join(local_dir, f"{model_name}.pickle"))
df_sel = df[["question", "response"]]
df_sel['response'] = df_sel["response"].apply(lambda x: x.split("</think>")[-1].strip("\n") if "</think>" in x else "")
df_sel = df_sel.rename(columns={'response': f'{model_name}_example'})

# Merge on question
merged = pd.merge(hf_df, df_sel, on="question", how="left").reset_index(drop=True)

# Convert to Dataset
new_ds = Dataset.from_pandas(merged)

# Push back to HuggingFace Hub
new_ds.push_to_hub(hf_dataset_name, split="test")
