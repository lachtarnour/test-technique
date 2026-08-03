"""Token-aligned supervision features for structured training ablations.

This module is the only bridge between character-level GSM8K annotations and
token-level model inputs.  It deliberately keeps graph construction independent
from Hugging Face and keeps experiment planning independent from data parsing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any

from src.data.graph.builder import build_calculation_graph
from src.data.graph.parser import parse_math_steps
from src.data.graph.postfix import compile_calculation_graph
from src.data.graph.schemas import MathStep, SourceSpan

TOKEN_LOSS_WEIGHTS = "token_loss_weights"
STEP_POSITIONS = "step_positions"
STEP_TARGETS = "step_targets"
STEP_TARGET_SCALES = "step_target_scales"
STEP_MASK = "step_mask"
POSTFIX_PROGRAM = "postfix_program"

BASE_MODEL_COLUMNS = frozenset({"input_ids", "attention_mask", "labels"})
TOKEN_FEATURE_COLUMNS = frozenset({TOKEN_LOSS_WEIGHTS})
STEP_FEATURE_COLUMNS = frozenset(
    {
        STEP_POSITIONS,
        STEP_TARGETS,
        STEP_TARGET_SCALES,
        STEP_MASK,
        POSTFIX_PROGRAM,
    }
)
STRUCTURED_FEATURE_COLUMNS = STEP_FEATURE_COLUMNS
SUPPORTED_FEATURE_COLUMNS = TOKEN_FEATURE_COLUMNS | STRUCTURED_FEATURE_COLUMNS

Offset = tuple[int, int]


def _validate_offsets(
    offsets: Sequence[Sequence[int]],
    *,
    token_count: int,
) -> tuple[Offset, ...]:
    if len(offsets) != token_count:
        raise ValueError(
            "offset_mapping must contain exactly one span per input token."
        )

    normalized: list[Offset] = []
    previous_start = 0
    for raw_offset in offsets:
        if len(raw_offset) != 2:
            raise ValueError("Every token offset must contain a start and an end.")
        start, end = (int(raw_offset[0]), int(raw_offset[1]))
        if start < 0 or end < start:
            raise ValueError("Token offsets must satisfy 0 <= start <= end.")
        if end > start and start < previous_start:
            raise ValueError("Non-empty token offsets must be ordered.")
        if end > start:
            previous_start = start
        normalized.append((start, end))
    return tuple(normalized)


def _token_overlaps(offset: Offset, span: SourceSpan) -> bool:
    start, end = offset
    return end > start and end > span.start and start < span.end


def _span_is_visible(offsets: Sequence[Offset], span: SourceSpan) -> bool:
    """Return whether non-special visible tokens cover the complete span."""
    covered_until = span.start
    for start, end in offsets:
        if end <= span.start or start >= span.end or end <= start:
            continue
        if start > covered_until:
            return False
        covered_until = max(covered_until, end)
        if covered_until >= span.end:
            return True
    return False


def _causal_position(offsets: Sequence[Offset], character_position: int) -> int | None:
    """Return the last token strictly before text beginning at a character."""
    for token_index, (start, end) in enumerate(offsets):
        if end > start and end > character_position:
            return token_index - 1 if token_index > 0 else None
    return None


def _step_context_start(answer: str, steps: Sequence[MathStep], index: int) -> int:
    """Find the earliest text belonging to a step, excluding prior steps."""
    annotation_start = steps[index].annotation_span.start
    line_start = answer.rfind("\n", 0, annotation_start) + 1
    previous_annotation_end = steps[index - 1].annotation_span.end if index > 0 else 0
    start = max(line_start, previous_annotation_end)
    while start < annotation_start and answer[start].isspace():
        start += 1
    return start


def _as_finite_float(value: Fraction | None) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _finite_float(value: Fraction | None, *, fallback: float) -> float:
    converted = _as_finite_float(value)
    return converted if converted is not None else fallback


def _target_scale(value: Fraction | None) -> float:
    if value is None:
        return 1.0
    return _finite_float(max(abs(value), Fraction(1)), fallback=1.0)


def _math_token_weights(
    *,
    labels: Sequence[int],
    offsets: Sequence[Offset],
    annotation_spans: Iterable[SourceSpan],
    annotation_offset: int,
    math_token_weight: float,
) -> list[float]:
    if (
        isinstance(math_token_weight, bool)
        or not isinstance(math_token_weight, (int, float))
        or not math.isfinite(math_token_weight)
        or math_token_weight <= 0
    ):
        raise ValueError("math_token_weight must be finite and strictly positive.")

    global_spans = tuple(
        SourceSpan(
            annotation_offset + span.start,
            annotation_offset + span.end,
        )
        for span in annotation_spans
    )
    weights: list[float] = []
    for label, offset in zip(labels, offsets, strict=True):
        if label == -100:
            weights.append(0.0)
        elif any(_token_overlaps(offset, span) for span in global_spans):
            weights.append(float(math_token_weight))
        else:
            weights.append(1.0)
    return weights


def _build_step_features(
    *,
    question: str,
    answer: str,
    answer_offset: int,
    offsets: Sequence[Offset],
    requested_columns: frozenset[str],
    steps: Sequence[MathStep],
) -> dict[str, Any]:
    graph = build_calculation_graph(question, list(steps))
    program = compile_calculation_graph(graph)
    if len(program.steps) != len(steps):
        raise RuntimeError("Math steps and postfix program steps are not aligned.")

    positions: list[int] = []
    targets: list[float] = []
    scales: list[float] = []
    masks: list[bool] = []
    for step, program_step in zip(steps, program.steps, strict=True):
        context_start = answer_offset + _step_context_start(
            answer,
            steps,
            step.index,
        )
        annotation_span = SourceSpan(
            answer_offset + step.annotation_span.start,
            answer_offset + step.annotation_span.end,
        )
        position = _causal_position(offsets, context_start)
        target = program_step.target_result
        finite_target = _as_finite_float(target)
        target_value = finite_target if finite_target is not None else 0.0
        scale = _target_scale(target)
        usable = (
            step.valid
            and program_step.valid
            and target is not None
            and finite_target is not None
            and position is not None
            and _span_is_visible(offsets, annotation_span)
        )

        positions.append(position if position is not None else 0)
        targets.append(target_value)
        scales.append(scale)
        masks.append(usable)

    candidates: dict[str, Any] = {
        STEP_POSITIONS: positions,
        STEP_TARGETS: targets,
        STEP_TARGET_SCALES: scales,
        STEP_MASK: masks,
        # JSON keeps the Arrow schema stable despite heterogeneous operand
        # reference kinds.  The collator decodes it before objective use.
        POSTFIX_PROGRAM: json.dumps(
            program.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    return {
        name: value for name, value in candidates.items() if name in requested_columns
    }


def build_training_features(
    *,
    question: str,
    answer: str,
    input_ids: Sequence[int],
    labels: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
    answer_offset: int,
    requested_columns: Iterable[str],
    math_token_weight: float = 1.0,
) -> dict[str, Any]:
    """Build only the features required by the selected experiment.

    ``answer_offset`` is the start of the assistant answer in the fully rendered
    chat text. Every emitted hidden-state position is causal and precedes the
    complete target-step text.
    """
    requested = frozenset(requested_columns)
    unknown = requested - SUPPORTED_FEATURE_COLUMNS
    if unknown:
        raise ValueError(f"Unsupported training feature columns: {sorted(unknown)}")
    if not requested:
        return {}
    if not isinstance(question, str) or not isinstance(answer, str):
        raise TypeError("question and answer must be strings.")
    if answer_offset < 0:
        raise ValueError("answer_offset must be non-negative.")
    if len(labels) != len(input_ids):
        raise ValueError("labels and input_ids must have identical lengths.")

    offsets = _validate_offsets(offset_mapping, token_count=len(input_ids))
    steps = parse_math_steps(answer)
    features: dict[str, Any] = {}

    if TOKEN_LOSS_WEIGHTS in requested:
        features[TOKEN_LOSS_WEIGHTS] = _math_token_weights(
            labels=labels,
            offsets=offsets,
            annotation_spans=(step.annotation_span for step in steps),
            annotation_offset=answer_offset,
            math_token_weight=math_token_weight,
        )
    if requested & STEP_FEATURE_COLUMNS:
        features.update(
            _build_step_features(
                question=question,
                answer=answer,
                answer_offset=answer_offset,
                offsets=offsets,
                requested_columns=requested,
                steps=steps,
            )
        )
    if set(features) != set(requested):
        missing = sorted(requested - features.keys())
        raise RuntimeError(f"Feature construction omitted required columns: {missing}")
    return features
