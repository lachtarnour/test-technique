"""Validated scientific configuration for the A1 control experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

CAUSAL_LANGUAGE_MODELING = "causal_language_modeling"


@dataclass(frozen=True)
class ExperimentConfig:
    """Scientific choices, kept separate from infrastructure parameters."""

    experiment_id: str
    objective: str = CAUSAL_LANGUAGE_MODELING

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty.")
        if self.objective != CAUSAL_LANGUAGE_MODELING:
            raise ValueError("A1 supports only the causal_language_modeling objective.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentTrainingArguments(TrainingArguments):
    """Trainer arguments extended with epoch-based experiment frequencies."""

    validation_every_epochs: int = field(
        default=1,
        metadata={"help": "Validate and checkpoint every N epochs."},
    )
    log_every_epochs: int = field(
        default=1,
        metadata={"help": "Log training metrics every N epochs."},
    )
    eval_every: int = field(
        default=2,
        metadata={"help": "Run fixed-sample generation metrics every N epochs."},
    )

    def __post_init__(self) -> None:
        for name in (
            "validation_every_epochs",
            "log_every_epochs",
            "eval_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a strictly positive integer.")
        super().__post_init__()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the explicit JSON configuration for a training run."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Experiment config not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON experiment config: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The experiment config must be a JSON object.")
    try:
        return ExperimentConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"Invalid fields in experiment config: {source}") from exc
