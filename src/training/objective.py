"""Composable scientific losses, independent of Trainer mechanics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from src.training.arguments import (
    LANGUAGE_LOSS,
    STANDARD_LANGUAGE_LOSS,
    LossTermConfig,
)

ScalarNormalizer = int | float | torch.Tensor


@dataclass(frozen=True)
class ObjectiveModelOutputs:
    """Minimal differentiable outputs consumed by scientific loss terms."""

    logits: torch.Tensor
    last_hidden_state: torch.Tensor


@dataclass(frozen=True)
class ObjectiveContext:
    """Minimal runtime state exposed to scientific loss terms."""

    normalizers: Mapping[str, ScalarNormalizer] = field(default_factory=dict)
    auxiliary_heads: Mapping[str, nn.Module] = field(default_factory=dict)

    def normalizer(self, loss_name: str) -> ScalarNormalizer | None:
        """Return the term-specific normalizer."""
        return self.normalizers.get(loss_name)

    def head(self, name: str) -> nn.Module:
        """Return one plan-required head or fail with an actionable error."""
        try:
            return self.auxiliary_heads[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Objective requires missing auxiliary head {name!r}."
            ) from exc


@dataclass(frozen=True)
class LossStatistics:
    """Additive statistics used for exact per-component aggregation."""

    numerator: torch.Tensor
    denominator: torch.Tensor


@dataclass
class LossOutput:
    """Loss contract shared by every experiment and the common Trainer."""

    total_loss: torch.Tensor
    statistics: dict[str, LossStatistics] = field(default_factory=dict)


@dataclass(frozen=True)
class TermLoss:
    """One unweighted loss value plus its additive logging statistics."""

    value: torch.Tensor
    statistics: LossStatistics


class LossTerm(ABC):
    """One independently testable scientific term."""

    def __init__(self, config: LossTermConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def component_name(self) -> str:
        return f"{self.name}_loss"

    @property
    def weight(self) -> float:
        return self.config.weight

    def supervision_count(self, batch: dict[str, Any]) -> ScalarNormalizer:
        """Count valid elements used by this term in one batch."""
        raise NotImplementedError(
            f"{type(self).__name__} must define supervision_count()."
        )

    @abstractmethod
    def compute(
        self,
        *,
        model_outputs: Any,
        batch: dict[str, Any],
        context: ObjectiveContext,
    ) -> TermLoss:
        """Return this term before its experiment coefficient is applied."""


def _scalar_denominator(
    value: ScalarNormalizer,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} normalizer must be a scalar.")
        denominator = value.to(device=reference.device, dtype=reference.dtype)
    else:
        denominator = reference.new_tensor(value)
    if not torch.isfinite(denominator).item() or denominator.item() <= 0:
        raise ValueError(f"{name} normalizer must be finite and strictly positive.")
    return denominator


class StandardLanguageLossTerm(LossTerm):
    """Completion-only causal cross-entropy used by the A1 control."""

    def __init__(self, config: LossTermConfig | None = None) -> None:
        resolved = config or LossTermConfig(name=LANGUAGE_LOSS)
        if resolved.name != LANGUAGE_LOSS:
            raise ValueError("StandardLanguageLossTerm requires language config.")
        if resolved.mode != STANDARD_LANGUAGE_LOSS:
            raise ValueError("StandardLanguageLossTerm requires mode='standard'.")
        super().__init__(resolved)

    def supervision_count(self, batch: dict[str, Any]) -> ScalarNormalizer:
        labels = batch["labels"]
        if isinstance(labels, torch.Tensor):
            return labels[..., 1:].ne(-100).sum()
        if not labels:
            return 0
        rows = [labels] if isinstance(labels[0], int) else labels
        return sum(int(label) != -100 for row in rows for label in row[1:])

    def compute(
        self,
        *,
        model_outputs: Any,
        batch: dict[str, Any],
        context: ObjectiveContext,
    ) -> TermLoss:
        logits = model_outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = batch["labels"][:, 1:].contiguous()
        numerator = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        natural_count = shifted_labels.ne(-100).sum()
        if natural_count.item() <= 0:
            raise ValueError("language loss requires at least one supervised token.")
        statistics_denominator = natural_count.to(
            device=numerator.device,
            dtype=numerator.dtype,
        )
        optimization_count = context.normalizer(self.name)
        if optimization_count is None:
            denominator = statistics_denominator
        else:
            denominator = _scalar_denominator(
                optimization_count,
                reference=numerator,
                name=self.name,
            )
        return TermLoss(
            value=numerator / denominator,
            statistics=LossStatistics(
                numerator=numerator,
                denominator=statistics_denominator,
            ),
        )


class TrainingObjective(ABC):
    """Compute scientific losses without owning any Trainer mechanics."""

    def normalization_counts(
        self,
        batches: list[dict[str, Any]],
    ) -> dict[str, ScalarNormalizer]:
        """Return accumulation-window counts for each scientific term."""
        del batches
        return {}

    @abstractmethod
    def compute_loss(
        self,
        *,
        model_outputs: Any,
        batch: dict[str, Any],
        context: ObjectiveContext,
    ) -> LossOutput:
        """Return the weighted total and additive logging statistics."""


class CompositeObjective(TrainingObjective):
    """Combine registered terms according to one experiment recipe."""

    def __init__(self, terms: tuple[LossTerm, ...]) -> None:
        if not terms:
            raise ValueError("CompositeObjective requires at least one loss term.")
        names = tuple(term.name for term in terms)
        if len(set(names)) != len(names):
            raise ValueError("CompositeObjective loss names must be unique.")
        self.terms = terms

    def normalization_counts(
        self,
        batches: list[dict[str, Any]],
    ) -> dict[str, ScalarNormalizer]:
        counts: dict[str, ScalarNormalizer] = {}
        for term in self.terms:
            count: ScalarNormalizer = 0
            for batch in batches:
                count = count + term.supervision_count(batch)
            counts[term.name] = count
        return counts

    def compute_loss(
        self,
        *,
        model_outputs: Any,
        batch: dict[str, Any],
        context: ObjectiveContext,
    ) -> LossOutput:
        total_loss: torch.Tensor | None = None
        statistics: dict[str, LossStatistics] = {}
        for term in self.terms:
            term_loss = term.compute(
                model_outputs=model_outputs,
                batch=batch,
                context=context,
            )
            weighted = term_loss.value * term.weight
            total_loss = weighted if total_loss is None else total_loss + weighted
            statistics[term.component_name] = term_loss.statistics
        if total_loss is None:  # Defensive: constructor already rejects this state.
            raise RuntimeError("CompositeObjective has no loss terms.")
        return LossOutput(
            total_loss=total_loss,
            statistics=statistics,
        )
