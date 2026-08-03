"""Ablation recipes, dynamic requirements and loss computation in one place."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from src.data.features import (
    POSTFIX_PROGRAM,
    STEP_MASK,
    STEP_POSITIONS,
    STEP_TARGET_SCALES,
    STEP_TARGETS,
    TOKEN_LOSS_WEIGHTS,
)
from src.model.heads import (
    ACTION_OPERATOR_HEAD,
    COMPOSITION_HEAD,
    NUMERIC_RESULT_HEAD,
    OPERAND_REFERENCE_HEAD,
)

ScalarNormalizer = int | float | torch.Tensor


def count_language_tokens(batch: dict[str, Any]) -> ScalarNormalizer:
    """Count completion tokens, or their exact A2 weights when available."""
    labels = batch["labels"]
    weights = batch.get(TOKEN_LOSS_WEIGHTS)
    if isinstance(labels, torch.Tensor):
        mask = labels[..., 1:].ne(-100)
        if weights is None:
            return mask.sum()
        if not isinstance(weights, torch.Tensor) or weights.shape != labels.shape:
            raise ValueError("token_loss_weights must have the same shape as labels.")
        selected = weights[..., 1:].masked_select(mask)
        if not torch.isfinite(selected).all().item() or selected.le(0).any().item():
            raise ValueError("Supervised token weights must be finite and positive.")
        return selected.sum()
    if not labels:
        return 0
    label_rows = [labels] if isinstance(labels[0], int) else labels
    if weights is None:
        return sum(int(label) != -100 for row in label_rows for label in row[1:])
    if not isinstance(weights, list) or not weights:
        raise ValueError("token_loss_weights must be a non-empty list.")
    weight_rows = [weights] if isinstance(weights[0], (int, float)) else weights
    if len(weight_rows) != len(label_rows):
        raise ValueError("token_loss_weights must have the same shape as labels.")
    total = 0.0
    for label_row, weight_row in zip(label_rows, weight_rows, strict=True):
        if len(weight_row) != len(label_row):
            raise ValueError("token_loss_weights must have the same shape as labels.")
        for label, weight in zip(label_row[1:], weight_row[1:], strict=True):
            if int(label) == -100:
                continue
            value = float(weight)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Supervised token weights must be finite and positive."
                )
            total += value
    return total


def _positive_scalar(
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


def language_loss(
    model_outputs: Mapping[str, torch.Tensor],
    batch: dict[str, Any],
    normalizer: ScalarNormalizer | None,
    auxiliary_heads: Mapping[str, nn.Module],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Compute A1 cross-entropy or A2 token-weighted cross-entropy."""
    del auxiliary_heads
    logits = model_outputs["logits"]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = batch["labels"][:, 1:].contiguous()
    token_losses = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(-100)
    raw_weights = batch.get(TOKEN_LOSS_WEIGHTS)
    if raw_weights is None:
        weights = mask.to(dtype=token_losses.dtype)
    else:
        if (
            not isinstance(raw_weights, torch.Tensor)
            or raw_weights.shape != batch["labels"].shape
        ):
            raise ValueError("token_loss_weights must have the same shape as labels.")
        weights = raw_weights[:, 1:].to(
            device=token_losses.device,
            dtype=token_losses.dtype,
        )
        selected = weights.masked_select(mask)
        if not torch.isfinite(selected).all().item() or selected.le(0).any().item():
            raise ValueError("Supervised token weights must be finite and positive.")
        weights = weights * mask
    numerator = (token_losses * weights).sum()
    natural_count = weights.sum()
    if natural_count.item() <= 0:
        raise ValueError("language loss requires at least one supervised token.")
    statistics_denominator = natural_count.to(
        device=numerator.device,
        dtype=numerator.dtype,
    )
    denominator = (
        statistics_denominator
        if normalizer is None
        else _positive_scalar(
            normalizer,
            reference=numerator,
            name="language",
        )
    )
    return numerator / denominator, (numerator, statistics_denominator)


# Each loss is: (weight, required features, required heads, compute, count).
LOSSES: dict[str, tuple[float, frozenset[str], frozenset[str], Any, Any]] = {
    "language": (
        1.0,
        frozenset(),
        frozenset(),
        language_loss,
        count_language_tokens,
    ),
    "language:math": (
        1.0,
        frozenset({TOKEN_LOSS_WEIGHTS}),
        frozenset(),
        language_loss,
        count_language_tokens,
    ),
    "result": (
        1.0,
        frozenset({STEP_POSITIONS, STEP_TARGETS, STEP_TARGET_SCALES, STEP_MASK}),
        frozenset({NUMERIC_RESULT_HEAD}),
        None,
        None,
    ),
    "execution": (
        1.0,
        frozenset({STEP_POSITIONS, STEP_TARGET_SCALES, STEP_MASK, POSTFIX_PROGRAM}),
        frozenset({NUMERIC_RESULT_HEAD}),
        None,
        None,
    ),
    "operator": (
        1.0,
        frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        frozenset({ACTION_OPERATOR_HEAD}),
        None,
        None,
    ),
    "dependency": (
        1.0,
        frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        frozenset({OPERAND_REFERENCE_HEAD}),
        None,
        None,
    ),
    "structured_action": (
        1.0,
        frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        frozenset({ACTION_OPERATOR_HEAD, OPERAND_REFERENCE_HEAD}),
        None,
        None,
    ),
    "composition": (
        1.0,
        frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        frozenset({COMPOSITION_HEAD}),
        None,
        None,
    ),
}

ABLATIONS = {
    "A1": ("language",),
    "A2": ("language:math",),
    "A3": ("language:math", "result"),
    "A4": ("language:math", "result", "execution"),
    "A5": ("language:math", "result", "execution", "operator", "dependency"),
    "A6": ("language:math", "result", "execution", "structured_action"),
    "A7": (
        "language:math",
        "result",
        "execution",
        "structured_action",
        "composition",
    ),
}


def compile_experiment(
    name: str,
    *,
    math_token_weight: float = 2.0,
    require_implemented: bool = False,
) -> dict[str, Any]:
    """Return the complete data/model plan for one named training ablation."""
    normalized = name.strip().upper()
    if normalized in {"A0", "A8"}:
        raise ValueError(f"{normalized} is an evaluation strategy, not training.")
    try:
        recipe = ABLATIONS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training ablation {name!r}; choose A1 to A7."
        ) from exc

    missing = [loss for loss in recipe if LOSSES[loss][3] is None]
    if require_implemented and missing:
        raise NotImplementedError(f"Losses not implemented yet: {missing}")
    if "language:math" in recipe and (
        isinstance(math_token_weight, bool)
        or not isinstance(math_token_weight, (int, float))
        or not math.isfinite(math_token_weight)
        or math_token_weight <= 0
    ):
        raise ValueError("math_token_weight must be finite and strictly positive.")

    specifications = [LOSSES[loss] for loss in recipe]
    return {
        "id": normalized,
        "losses": {loss: LOSSES[loss][0] for loss in recipe},
        "features": sorted(
            frozenset().union(*(features for _, features, _, _, _ in specifications))
        ),
        "heads": sorted(
            frozenset().union(*(heads for _, _, heads, _, _ in specifications))
        ),
        "math_token_weight": (
            float(math_token_weight) if "language:math" in recipe else 1.0
        ),
    }


def normalization_counts(
    losses: Mapping[str, float],
    batches: list[dict[str, Any]],
) -> dict[str, ScalarNormalizer]:
    """Count each active loss target over one accumulation window."""
    counts: dict[str, ScalarNormalizer] = {}
    for name in losses:
        count = LOSSES[name][4]
        if count is None:
            raise NotImplementedError(f"Loss not implemented yet: {name}")
        total: ScalarNormalizer = 0
        for batch in batches:
            total = total + count(batch)
        counts[name] = total
    return counts


def compute_objective(
    losses: Mapping[str, float],
    *,
    model_outputs: Mapping[str, torch.Tensor],
    batch: dict[str, Any],
    normalizers: Mapping[str, ScalarNormalizer] | None = None,
    auxiliary_heads: Mapping[str, nn.Module] | None = None,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Compute and combine every active loss in the selected ablation."""
    resolved_normalizers = normalizers or {}
    resolved_heads = auxiliary_heads or {}
    total_loss: torch.Tensor | None = None
    statistics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, weight in losses.items():
        compute = LOSSES[name][3]
        if compute is None:
            raise NotImplementedError(f"Loss not implemented yet: {name}")
        value, term_statistics = compute(
            model_outputs,
            batch,
            resolved_normalizers.get(name),
            resolved_heads,
        )
        weighted = value * weight
        total_loss = weighted if total_loss is None else total_loss + weighted
        statistics[f"{name.partition(':')[0]}_loss"] = term_statistics
    if total_loss is None:
        raise ValueError("An experiment requires at least one loss.")
    return total_loss, statistics
