"""Validated scientific and infrastructure configuration for experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

LANGUAGE_LOSS = "language"
RESULT_LOSS = "result"
EXECUTION_LOSS = "execution"
OPERATOR_LOSS = "operator"
DEPENDENCY_LOSS = "dependency"
STRUCTURED_ACTION_LOSS = "structured_action"
FINAL_ANSWER_LOSS = "final_answer"
COMPOSITION_LOSS = "composition"
STANDARD_LANGUAGE_LOSS = "standard"
MATH_WEIGHTED_LANGUAGE_LOSS = "math_weighted"
SUPPORTED_LANGUAGE_MODES = frozenset(
    {STANDARD_LANGUAGE_LOSS, MATH_WEIGHTED_LANGUAGE_LOSS}
)


@dataclass(frozen=True)
class LossTermConfig:
    """One enabled loss and its scientific coefficient."""

    name: str
    weight: float = 1.0
    mode: str | None = None
    math_token_weight: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Loss names must not be empty.")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("Loss weights must be finite and strictly positive.")
        object.__setattr__(self, "weight", float(self.weight))

        if self.name == LANGUAGE_LOSS:
            mode = self.mode or STANDARD_LANGUAGE_LOSS
            if mode not in SUPPORTED_LANGUAGE_MODES:
                raise ValueError(f"Unsupported language-loss mode: {mode!r}")
            object.__setattr__(self, "mode", mode)

            math_weight = self.math_token_weight
            if mode == STANDARD_LANGUAGE_LOSS:
                if math_weight not in (None, 1, 1.0):
                    raise ValueError(
                        "Standard language loss cannot weight math tokens."
                    )
                object.__setattr__(self, "math_token_weight", 1.0)
            else:
                if (
                    math_weight is None
                    or isinstance(math_weight, bool)
                    or not isinstance(math_weight, (int, float))
                    or not math.isfinite(math_weight)
                    or math_weight <= 0
                ):
                    raise ValueError(
                        "Math-weighted language loss requires a finite, "
                        "strictly positive math_token_weight."
                    )
                object.__setattr__(
                    self,
                    "math_token_weight",
                    float(math_weight),
                )
        elif self.mode is not None or self.math_token_weight is not None:
            raise ValueError(
                f"Loss {self.name!r} does not accept language-specific options."
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Declarative scientific recipe, independent of Trainer mechanics."""

    experiment_id: str
    losses: tuple[LossTermConfig, ...] = (LossTermConfig(name=LANGUAGE_LOSS),)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty.")
        if not self.losses:
            raise ValueError("A training experiment requires at least one loss.")
        if not all(isinstance(loss, LossTermConfig) for loss in self.losses):
            raise ValueError("losses must contain only LossTermConfig values.")
        names = tuple(loss.name for loss in self.losses)
        if len(set(names)) != len(names):
            raise ValueError("Each loss may appear at most once in an experiment.")
        if LANGUAGE_LOSS not in names:
            raise ValueError("Every training experiment must include language loss.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "losses": [asdict(loss) for loss in self.losses],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Build and validate one recipe from its JSON representation."""
        normalized = dict(payload)
        raw_losses = normalized.pop("losses", None)
        if raw_losses is None:
            raw_losses = [{"name": LANGUAGE_LOSS}]
        if not isinstance(raw_losses, list):
            raise ValueError("losses must be a JSON array.")
        try:
            losses = tuple(
                item if isinstance(item, LossTermConfig) else LossTermConfig(**item)
                for item in raw_losses
            )
        except TypeError as exc:
            raise ValueError("Every loss must be a JSON object.") from exc
        try:
            return cls(losses=losses, **normalized)
        except TypeError as exc:
            raise ValueError("Invalid experiment configuration fields.") from exc


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

    def __post_init__(self) -> None:
        for name in (
            "validation_every_epochs",
            "log_every_epochs",
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
        return ExperimentConfig.from_dict(payload)
    except ValueError as exc:
        raise ValueError(f"Invalid experiment config {source}: {exc}") from exc
