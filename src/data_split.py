"""Create the frozen GSM8K train/validation/test dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.config import CONFIG

MANIFEST_NAME = "split_manifest.json"


def _example_key(question: str, answer: str) -> str:
    payload = f"{question}\0{answer}".encode()
    return hashlib.sha256(payload).hexdigest()


def _ordered_split_hash(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for question, answer in zip(
        dataset["question"],
        dataset["answer"],
        strict=True,
    ):
        digest.update(_example_key(question, answer).encode())
    return digest.hexdigest()


def _validate_raw_dataset(dataset: Dataset) -> None:
    missing_columns = CONFIG.required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if len(dataset) < 2:
        raise ValueError("At least two examples are required in every source split.")


def _split_keys(dataset: Dataset) -> set[str]:
    return {
        _example_key(question, answer)
        for question, answer in zip(
            dataset["question"],
            dataset["answer"],
            strict=True,
        )
    }


def _validate_disjoint_splits(splits: Mapping[str, Dataset]) -> None:
    split_keys = {name: _split_keys(dataset) for name, dataset in splits.items()}
    split_names = list(split_keys)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_keys[left_name] & split_keys[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name!r} and {right_name!r} contain "
                    f"{len(overlap)} overlapping examples."
                )


def create_frozen_split(
    output_path: str | Path = CONFIG.dataset_path,
    *,
    validation_size: float = CONFIG.validation_size,
    seed: int = CONFIG.seed,
) -> dict[str, Any]:
    """Split the official train and preserve the untouched official test."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between zero and one.")

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"Frozen split already exists at {destination}. Refusing to overwrite it."
        )

    official_train = load_dataset(
        CONFIG.dataset_name,
        CONFIG.dataset_config,
        split="train",
        revision=CONFIG.dataset_revision,
    )
    official_test = load_dataset(
        CONFIG.dataset_name,
        CONFIG.dataset_config,
        split="test",
        revision=CONFIG.dataset_revision,
    )
    _validate_raw_dataset(official_train)
    _validate_raw_dataset(official_test)

    split = official_train.train_test_split(
        test_size=validation_size,
        seed=seed,
    )
    frozen = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": official_test,
        }
    )
    _validate_disjoint_splits(frozen)

    manifest = {
        "dataset": CONFIG.dataset_name,
        "config": CONFIG.dataset_config,
        "revision": CONFIG.dataset_revision,
        "source_train_split": "train",
        "source_test_split": "test",
        "seed": seed,
        "validation_size": validation_size,
        "total_examples": len(official_train) + len(official_test),
        "official_train_examples": len(official_train),
        "train_examples": len(frozen["train"]),
        "validation_examples": len(frozen["validation"]),
        "test_examples": len(frozen["test"]),
        "train_sha256": _ordered_split_hash(frozen["train"]),
        "validation_sha256": _ordered_split_hash(frozen["validation"]),
        "test_sha256": _ordered_split_hash(frozen["test"]),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    frozen.save_to_disk(str(destination))
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
