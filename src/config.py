"""Central configuration for the fine-tuning project."""

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_SPLIT = "train"
DEFAULT_EVALUATION_SPLIT = "test"
DEFAULT_SEED = 42
REQUIRED_COLUMNS = frozenset({"question", "answer"})
