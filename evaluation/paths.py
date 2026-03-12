"""Central path definitions for the evaluation pipeline."""
from pathlib import Path

EVAL_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = EVAL_DIR.parent
INCEPTION_DATA = EVAL_DIR / "inception_data"

MAX_ITER_5_THINK = INCEPTION_DATA / "max_iterations_5" / "think"
MAX_ITER_5_INSTRUCT = INCEPTION_DATA / "max_iterations_5" / "instruct"
MAX_ITER_1 = INCEPTION_DATA / "max_iterations_1"
