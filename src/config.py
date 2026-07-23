"""Central configuration for the fine-tuning project."""

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_SPLIT = "train"
DEFAULT_EVALUATION_SPLIT = "test"
DEFAULT_SEED = 42
REQUIRED_COLUMNS = frozenset({"question", "answer"})
DEFAULT_VALIDATION_SIZE = 0.1
SYSTEM_PROMPT = (
    "Solve the math problem step by step and finish with the final answer "
    "in the format #### number."
)
