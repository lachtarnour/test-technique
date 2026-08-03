"""Functional dynamic padding for language and structured training features."""

from __future__ import annotations

import json
from typing import Any

import torch

from src.data.features import (
    BASE_MODEL_COLUMNS,
    POSTFIX_PROGRAM,
    STEP_MASK,
    STEP_POSITIONS,
    STEP_TARGET_SCALES,
    STEP_TARGETS,
    SUPPORTED_FEATURE_COLUMNS,
    TOKEN_LOSS_WEIGHTS,
)

_STEP_SEQUENCE_COLUMNS = frozenset(
    {STEP_POSITIONS, STEP_TARGETS, STEP_TARGET_SCALES, STEP_MASK}
)


def _pad(values: list[int], size: int, fill: int) -> list[int]:
    return [*values, *([fill] * (size - len(values)))]


def _feature_columns(features: list[dict[str, Any]]) -> frozenset[str]:
    expected = frozenset(features[0]) - BASE_MODEL_COLUMNS
    unknown = expected - SUPPORTED_FEATURE_COLUMNS
    if unknown:
        raise ValueError(f"Unsupported collator columns: {sorted(unknown)}")
    for row_index, feature in enumerate(features[1:], start=1):
        actual = frozenset(feature) - BASE_MODEL_COLUMNS
        if actual != expected:
            raise ValueError(
                "Every row in a batch must expose the same feature columns; "
                f"row {row_index} differs."
            )
    return expected


def _decode_program(value: Any) -> dict[str, Any]:
    try:
        program = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError("postfix_program is not valid JSON.") from exc
    if not isinstance(program, dict) or not isinstance(program.get("steps"), list):
        raise ValueError("postfix_program must contain a list of steps.")
    return program


def _step_count(
    feature: dict[str, Any],
    columns: frozenset[str],
    program: dict[str, Any] | None,
) -> int:
    counts = {len(feature[name]) for name in columns & _STEP_SEQUENCE_COLUMNS}
    if program is not None:
        counts.add(len(program["steps"]))
    if len(counts) > 1:
        raise ValueError("Step-aligned feature columns have inconsistent lengths.")
    return next(iter(counts), 0)


def collate_completion_only(
    features: list[dict[str, Any]],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int | None = 8,
) -> dict[str, Any]:
    """Pad one batch while keeping prompt labels and missing targets masked."""
    if not features:
        raise ValueError("Cannot collate an empty batch.")
    columns = _feature_columns(features)
    sequence_length = max(len(feature["input_ids"]) for feature in features)
    if pad_to_multiple_of:
        sequence_length = (
            (sequence_length + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of
    batch: dict[str, Any] = {
        "input_ids": torch.tensor(
            [
                _pad(feature["input_ids"], sequence_length, pad_token_id)
                for feature in features
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                _pad(feature["attention_mask"], sequence_length, 0)
                for feature in features
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [_pad(feature["labels"], sequence_length, -100) for feature in features],
            dtype=torch.long,
        ),
    }

    if TOKEN_LOSS_WEIGHTS in columns:
        for feature in features:
            if len(feature[TOKEN_LOSS_WEIGHTS]) != len(feature["input_ids"]):
                raise ValueError(
                    "token_loss_weights must align exactly with input_ids."
                )
        batch[TOKEN_LOSS_WEIGHTS] = torch.tensor(
            [
                [
                    *feature[TOKEN_LOSS_WEIGHTS],
                    *([0.0] * (sequence_length - len(feature[TOKEN_LOSS_WEIGHTS]))),
                ]
                for feature in features
            ],
            dtype=torch.float32,
        )

    programs = (
        [_decode_program(feature[POSTFIX_PROGRAM]) for feature in features]
        if POSTFIX_PROGRAM in columns
        else [None] * len(features)
    )
    step_counts = [
        _step_count(feature, columns, program)
        for feature, program in zip(features, programs, strict=True)
    ]
    max_steps = max(step_counts, default=0)
    step_specs = {
        STEP_POSITIONS: (0, torch.long),
        STEP_TARGETS: (0.0, torch.float32),
        STEP_TARGET_SCALES: (1.0, torch.float32),
        STEP_MASK: (False, torch.bool),
    }
    for name, (fill, dtype) in step_specs.items():
        if name not in columns:
            continue
        batch[name] = torch.tensor(
            [
                [*feature[name], *([fill] * (max_steps - len(feature[name])))]
                for feature in features
            ],
            dtype=dtype,
        )

    if POSTFIX_PROGRAM in columns:
        batch[POSTFIX_PROGRAM] = programs
    return batch
