"""End-to-end construction of the explicit A1 tokenized dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from src.config import CONFIG
from src.data.formatting import format_training_example
from src.data.tokenization import tokenize_dataset_split
from src.load_data import load_frozen_gsm8k_split

LOGGER = logging.getLogger(__name__)


def _prepare_split(
    dataset: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    formatted = dataset.map(
        format_training_example,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
        desc="Formatting GSM8K for A1-control",
    )
    tokenized = tokenize_dataset_split(
        formatted,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    LOGGER.info("Prepared %s completion-only examples.", f"{len(tokenized):,}")
    return tokenized


def prepare_tokenized_dataset(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path = CONFIG.dataset_path,
    max_length: int = 1024,
    train_subset_size: int | None = None,
    validation_subset_size: int | None = None,
    seed: int = CONFIG.seed,
) -> DatasetDict:
    """Prepare train/validation features without exposing test to training."""
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
            ),
            "validation": _prepare_split(
                raw_validation,
                tokenizer=tokenizer,
                max_length=max_length,
            ),
        }
    )
