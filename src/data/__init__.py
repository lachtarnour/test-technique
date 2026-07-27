"""Training-data preparation for the baseline and structured V1 pipeline."""

from __future__ import annotations

from typing import Any

__all__ = ["prepare_tokenized_dataset"]


def __getattr__(name: str) -> Any:
    """Keep the package import lightweight while preserving the A1 public API."""
    if name == "prepare_tokenized_dataset":
        from src.data.dataset import prepare_tokenized_dataset

        return prepare_tokenized_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
