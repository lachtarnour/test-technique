"""The narrow loss interface that future experiments can extend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ObjectiveContext:
    """Minimal Trainer state exposed to an objective."""

    global_step: int
    is_training: bool
    normalization_count: int | float | torch.Tensor | None = None


@dataclass
class LossOutput:
    """Loss contract shared by the Trainer and future objectives."""

    total_loss: torch.Tensor
    components: dict[str, torch.Tensor]


class TrainingObjective(ABC):
    """Compute scientific losses without owning any Trainer mechanics."""

    @abstractmethod
    def compute_loss(
        self,
        *,
        model_outputs: Any,
        batch: dict[str, Any],
        context: ObjectiveContext,
    ) -> LossOutput:
        """Return the total loss and named components."""


class CausalLanguageModelingObjective(TrainingObjective):
    """A1: standard completion-only causal cross-entropy."""

    def compute_loss(
        self,
        *,
        model_outputs: Any,
        batch: dict[str, Any],
        context: ObjectiveContext,
    ) -> LossOutput:
        logits = model_outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = batch["labels"][:, 1:].contiguous()
        language_loss_sum = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        normalization_count = context.normalization_count
        if normalization_count is None:
            normalization_count = int(shifted_labels.ne(-100).sum().item())
        if isinstance(normalization_count, torch.Tensor):
            if normalization_count.numel() != 1:
                raise ValueError("normalization_count must be a scalar.")
            denominator = normalization_count.to(
                device=language_loss_sum.device,
                dtype=language_loss_sum.dtype,
            )
        else:
            if normalization_count <= 0:
                raise ValueError("normalization_count must be strictly positive.")
            denominator = language_loss_sum.new_tensor(normalization_count)
        language_loss = language_loss_sum / denominator
        return LossOutput(
            total_loss=language_loss,
            components={"language_loss": language_loss},
        )
