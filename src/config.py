"""Central project configuration."""

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Stable model, dataset and evaluation settings."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_revision: str = "740312add88f781978c0658806c59bc2815b9866d"
    default_split: str = "train"
    evaluation_split: str = "test"
    seed: int = 42
    required_columns: frozenset[str] = frozenset({"question", "answer"})
    validation_size: float = 0.15
    dataset_path: str = "data/gsm8k_train_validation15_test_seed42"
    num_train_epochs: float = 30.0
    train_batch_size: int = 24
    gradient_accumulation_steps: int = 3
    drop_incomplete_train_batch: bool = True
    eval_batch_size: int = 32
    periodic_eval_batch_size: int = 300
    generation_batch_size: int = 300
    max_new_tokens: int = 768
    prompt_version: str = "2.4.0"
    prompt_path: Path = Path(__file__).resolve().parent / "training" / "prompt.md"

    @cached_property
    def system_prompt(self) -> str:
        """Load the frozen prompt only when it is needed."""
        try:
            prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Prompt file not found: {self.prompt_path}"
            ) from exc
        if not prompt:
            raise ValueError(f"Prompt file is empty: {self.prompt_path}")
        return prompt

    @cached_property
    def system_prompt_sha256(self) -> str:
        """Identify the exact frozen prompt used by an experiment."""
        return sha256(self.system_prompt.encode("utf-8")).hexdigest()


CONFIG = Config()
