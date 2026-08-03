"""End-to-end construction of the stable tokenized language dataset."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from src.config import CONFIG
from src.data.loading import load_frozen_gsm8k_split

from .formatting import format_training_example
from .tokenization import tokenize_dataset_split

LOGGER = logging.getLogger(__name__)


def _prepare_split(
    dataset: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    feature_columns: Iterable[str] = (),
    math_token_weight: float = 1.0,
) -> Dataset:
    formatted = dataset.map(
        format_training_example,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
        desc="Formatting GSM8K for training",
    )
    tokenized = tokenize_dataset_split(
        formatted,
        tokenizer=tokenizer,
        max_length=max_length,
        feature_columns=feature_columns,
        math_token_weight=math_token_weight,
    )
    LOGGER.info("Prepared %s language-training examples.", f"{len(tokenized):,}")
    return tokenized


def prepare_tokenized_dataset(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path = CONFIG.dataset_path,
    max_length: int = 1024,
    train_subset_size: int | None = None,
    validation_subset_size: int | None = None,
    seed: int = CONFIG.seed,
    feature_columns: Iterable[str] = (),
    math_token_weight: float = 1.0,
) -> DatasetDict:
    """Prepare train/validation features without exposing test to training."""
    requested_features = frozenset(feature_columns)
    raw_train = load_frozen_gsm8k_split(
        "train",
        dataset_path=dataset_path,
        subset_size=train_subset_size,
        seed=seed,
    )
    raw_validation = load_frozen_gsm8k_split(
        "validation",
        dataset_path=dataset_path,
        subset_size=validation_subset_size,
        seed=seed,
    )
    return DatasetDict(
        {
            "train": _prepare_split(
                raw_train,
                tokenizer=tokenizer,
                max_length=max_length,
                feature_columns=requested_features,
                math_token_weight=math_token_weight,
            ),
            "validation": _prepare_split(
                raw_validation,
                tokenizer=tokenizer,
                max_length=max_length,
                feature_columns=requested_features,
                math_token_weight=math_token_weight,
            ),
        }
    )
