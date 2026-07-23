"""Load and inspect the raw GSM8K dataset before fine-tuning."""

from __future__ import annotations

import argparse
import logging

from datasets import Dataset, load_dataset

from src.config import (
    DATASET_CONFIG,
    DATASET_NAME,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    REQUIRED_COLUMNS,
)

LOGGER = logging.getLogger(__name__)


def load_gsm8k_dataset(
    split: str = DEFAULT_SPLIT,
    subset_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Load GSM8K, validate its schema, and optionally sample it."""
    if subset_size is not None and subset_size <= 0:
        raise ValueError("subset_size must be strictly positive.")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
    )

    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if subset_size is not None:
        sample_size = min(subset_size, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return dataset


def inspect_dataset(dataset: Dataset) -> None:
    """Log the dataset schema and a representative example."""
    if len(dataset) == 0:
        raise ValueError("Cannot inspect an empty dataset.")

    non_empty_questions = sum(
        bool(question.strip()) for question in dataset["question"]
    )
    non_empty_answers = sum(
        bool(answer.strip()) for answer in dataset["answer"]
    )
    example = dataset[0]

    LOGGER.info("Number of examples: %s", f"{len(dataset):,}")
    LOGGER.info("Columns: %s", dataset.column_names)
    LOGGER.info("Features: %s", dataset.features)
    LOGGER.info(
        "Non-empty questions: %s/%s",
        f"{non_empty_questions:,}",
        f"{len(dataset):,}",
    )
    LOGGER.info(
        "Non-empty answers: %s/%s",
        f"{non_empty_answers:,}",
        f"{len(dataset):,}",
    )
    LOGGER.info("First question:\n%s", example["question"])
    LOGGER.info("First answer:\n%s", example["answer"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and inspect the GSM8K dataset."
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional number of shuffled examples to inspect.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Run dataset loading and inspection from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    args = parse_args()
    dataset = load_gsm8k_dataset(
        split=args.split,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    inspect_dataset(dataset)


if __name__ == "__main__":
    main()
