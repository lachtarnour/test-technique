"""Load original or frozen GSM8K dataset splits."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src.config import CONFIG


def _validate_schema(dataset: Dataset) -> None:
    missing_columns = CONFIG.required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def _select_subset(
    dataset: Dataset,
    *,
    subset_size: int | None,
    seed: int,
) -> Dataset:
    if subset_size is None:
        return dataset
    if subset_size <= 0:
        raise ValueError("subset_size must be strictly positive.")
    sample_size = min(subset_size, len(dataset))
    shuffled = dataset.shuffle(seed=seed, keep_in_memory=True)
    return shuffled.select(range(sample_size), keep_in_memory=True)


def load_gsm8k_dataset(
    split: str = CONFIG.default_split,
    subset_size: int | None = None,
    seed: int = CONFIG.seed,
) -> Dataset:
    """Load GSM8K, validate its schema, and optionally sample it."""
    dataset = load_dataset(
        CONFIG.dataset_name,
        CONFIG.dataset_config,
        split=split,
        revision=CONFIG.dataset_revision,
    )
    _validate_schema(dataset)
    return _select_subset(dataset, subset_size=subset_size, seed=seed)


def load_frozen_gsm8k_split(
    split: str,
    *,
    dataset_path: str | Path = CONFIG.dataset_path,
    subset_size: int | None = None,
    seed: int = CONFIG.seed,
) -> Dataset:
    """Load train, validation or test from the frozen local DatasetDict."""
    source = Path(dataset_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Frozen dataset not found at {source}. "
            "Run python3 script/create_data_split.py first."
        )

    dataset = load_from_disk(str(source))
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected a DatasetDict at {source}.")
    if split not in dataset:
        raise ValueError(
            f"Unknown frozen split {split!r}. Available splits: {sorted(dataset)}"
        )

    selected = dataset[split]
    _validate_schema(selected)
    return _select_subset(
        selected,
        subset_size=subset_size,
        seed=seed,
    )
