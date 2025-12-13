"""
Model engine initialization and management.
"""

import os
import sys
import fasttext

# Add project root directory to path for imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.llm_engine import OpenLMEngine, ModelConfig
from pipeline.config import REFUSAL_MODEL_PATH


# Note: OSS model now always uses API - no local engine creation needed
# This function is kept for compatibility but is no longer used
def create_oss_engine(
    model_name: str, 
    tokenizer_name: str, 
    **kwargs
) -> None:
    """
    OSS model now always uses API - this function is deprecated.
    
    Args:
        model_name: Not used (kept for compatibility)
        tokenizer_name: Not used (kept for compatibility)
        **kwargs: Not used (kept for compatibility)
        
    Returns:
        None (OSS model uses API directly via openaiapi.generate_completions)
    """
    return None


def create_architect_engine(model_name: str, tokenizer_name: str, **kwargs) -> OpenLMEngine:
    """
    Create and initialize architect model engine.
    
    Args:
        model_name: Name/path of the architect model
        tokenizer_name: Name/path of the tokenizer
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized OpenLMEngine instance
    """
    config = ModelConfig(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        max_tokens=2048,  # Reduced from 32768 - blank slate reasoning doesn't need that much, and continuation only uses 30 tokens
        temperature=0.6,
        **kwargs
    )
    return OpenLMEngine(config=config)


def load_refusal_model(path: str = None) -> fasttext.FastText:
    """
    Load FastText refusal detection model.
    
    Args:
        path: Path to the FastText model file. If None, uses default from config.
            Can be absolute path or relative to the injection_pipeline directory.
        
    Returns:
        Loaded FastText model
        
    Raises:
        FileNotFoundError: If the model file cannot be found
    """
    if path is None:
        path = REFUSAL_MODEL_PATH
    
    # If path is not absolute, try to resolve it
    if not os.path.isabs(path):
        # First try: relative to injection_pipeline directory (where config.py is)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(config_dir, path)
        if os.path.exists(local_path):
            path = local_path
        elif os.path.exists(path):
            # Path is already correct
            pass
        else:
            # Try relative to project root as fallback
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            fallback_path = os.path.join(project_root, path.lstrip('/'))
            if os.path.exists(fallback_path):
                path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Refusal model not found. Tried:\n"
                    f"  - {local_path}\n"
                    f"  - {path}\n"
                    f"  - {fallback_path}"
                )
    
    # Verify file exists before loading
    if not os.path.exists(path):
        raise FileNotFoundError(f"Refusal model file not found: {path}")
    
    return fasttext.load_model(path)

