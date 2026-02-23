"""
Run DeepSeek V3.2 evaluation on ablation experiments using existing SafetyEvaluator.

Evaluates 20 samples (10 bio + 10 chem) from each ablation length.
"""

import os
import sys
import asyncio
import pandas as pd
import importlib.util
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent to path
_script_dir = Path(__file__).parent.resolve()
_parent_dir = _script_dir.parent.resolve()
sys.path.insert(0, str(_parent_dir))

# Import safety_judge module directly to avoid vllm import issues
def _import_module_directly(module_name: str, module_path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load safety_judge module
_safety_judge = _import_module_directly("safety_judge", _script_dir / "safety-judge.py")
SafetyEvaluator = _safety_judge.SafetyEvaluator


def load_and_prepare_ablation(ablation_tokens: int, n_bio: int = 10, n_chem: int = 10, seed: int = 42):
    """Load ablation data and prepare for evaluation."""

    # Load pickle file
    file_path = f"max_iterations_1/think/architect_initial_max_tokens_{ablation_tokens}/DeepSeek-V3.2.pickle"
    df = pd.read_pickle(file_path)

    print(f"\nLoaded ablation {ablation_tokens}: {df.shape[0]} samples")

    # Sample stratified by category
    bio_samples = df[df['category'] == 'wmdp-bio'].sample(n=n_bio, random_state=seed)
    chem_samples = df[df['category'] == 'wmdp-chem'].sample(n=n_chem, random_state=seed)

    sampled_df = pd.concat([bio_samples, chem_samples]).sort_index()
    print(f"Sampled {len(sampled_df)} samples: {len(bio_samples)} bio + {len(chem_samples)} chem")

    # Create reasoning_traces column (combine architect + target)
    sampled_df['reasoning_traces'] = (
        "=== ARCHITECT REASONING ===\n" +
        sampled_df['architect_iteration_0'].astype(str) +
        "\n\n=== TARGET RESPONSE ===\n" +
        sampled_df['target_iteration_0'].astype(str)
    )

    # Map category to SafetyEvaluator labels
    # wmdp-bio -> bioterrorism, wmdp-chem -> chemical
    sampled_df['eval_category'] = sampled_df['category'].map({
        'wmdp-bio': 'bioterrorism',
        'wmdp-chem': 'chemical'
    })

    return sampled_df


async def evaluate_ablation(ablation_tokens: int, eval_model: str, client_name: str, overwrite: bool = False):
    """Evaluate a single ablation using SafetyEvaluator."""

    print(f"\n{'='*80}")
    print(f"Evaluating ablation: {ablation_tokens} tokens")
    print(f"{'='*80}")

    # Load and prepare data
    df = load_and_prepare_ablation(ablation_tokens)

    # Create output directory for this ablation
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"evaluation/results/ablation_{ablation_tokens}_{timestamp}"

    # Initialize SafetyEvaluator
    evaluator = SafetyEvaluator(
        input_df=df,
        inquiry_col="inquiry",
        reasoning_trace_col="reasoning_traces",
        category_col="eval_category",
        chemical_label="chemical",
        bioterrorism_label="bioterrorism",
        output_dir=output_dir,
        nick_name=f"ablation_{ablation_tokens}",
        eval_model=eval_model,
        client_name=client_name,
        temperature=0.0,
        max_tokens=4096,
        requests_per_second=5.0
    )

    # Run evaluation
    print(f"\nRunning evaluation with {eval_model} via {client_name}...")
    results_df = await evaluator.run(overwrite=overwrite)

    print(f"\n✓ Completed ablation {ablation_tokens}")
    print(f"  Results saved to: {output_dir}")
    print(f"  Total samples evaluated: {len(results_df)}")

    return results_df, output_dir


async def main():
    """Main execution."""

    print("="*80)
    print("DeepSeek V3.2 Ablation Evaluation")
    print("="*80)

    # Configuration
    ablation_tokens = [128, 512, 768, 1024]
    eval_model = "deepseek-chat"  # DeepSeek V3.2
    client_name = "deepseek"
    overwrite = False

    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n❌ ERROR: DEEPSEEK_API_KEY environment variable not set")
        sys.exit(1)

    print(f"\n✓ Using model: {eval_model}")
    print(f"✓ API provider: {client_name}")
    print(f"✓ Ablations to process: {ablation_tokens}")

    # Process each ablation
    all_results = {}

    for tokens in ablation_tokens:
        try:
            results_df, output_dir = await evaluate_ablation(
                ablation_tokens=tokens,
                eval_model=eval_model,
                client_name=client_name,
                overwrite=overwrite
            )
            all_results[tokens] = {
                'output_dir': output_dir,
                'num_samples': len(results_df)
            }
        except Exception as e:
            print(f"\n❌ Error processing ablation {tokens}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*80}")
    print("Evaluation Complete")
    print(f"{'='*80}")

    for tokens, info in all_results.items():
        print(f"\n{tokens} tokens:")
        print(f"  - Output: {info['output_dir']}")
        print(f"  - Samples: {info['num_samples']}")

    print(f"\n✓ All evaluations complete!")


if __name__ == "__main__":
    asyncio.run(main())
