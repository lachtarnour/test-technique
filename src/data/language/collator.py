"""Dynamic padding for tokenized language-model examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _pad(values: list[int], size: int, fill: int) -> list[int]:
    return [*values, *([fill] * (size - len(values)))]


@dataclass
class CompletionOnlyDataCollator:
    """Pad inputs normally and prompt-masked labels with ``-100``."""

    pad_token_id: int
    pad_to_multiple_of: int | None = 8

    def _sequence_length(self, features: list[dict[str, Any]]) -> int:
        length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            length = ((length + multiple - 1) // multiple) * multiple
        return length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        sequence_length = self._sequence_length(features)
        return {
            "input_ids": torch.tensor(
                [
                    _pad(feature["input_ids"], sequence_length, self.pad_token_id)
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
                [
                    _pad(feature["labels"], sequence_length, -100)
                    for feature in features
                ],
                dtype=torch.long,
            ),
        }
