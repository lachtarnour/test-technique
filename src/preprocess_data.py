"""Prepare GSM8K examples for supervised fine-tuning."""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any

from datasets import Dataset, DatasetDict

from src.config import (
    DEFAULT_SEED,
    DEFAULT_VALIDATION_SIZE,
    SYSTEM_PROMPT,
)
from src.evaluation import extract_final_answer
from src.load_data import load_gsm8k_dataset


EQUATION_PATTERN = re.compile(r"<<([^<>]+)>>")
LOGGER = logging.getLogger(__name__)


def extract_equations(answer: str) -> list[str]:
    """Extract GSM8K calculation annotations from an answer."""
    return EQUATION_PATTERN.findall(answer)


def format_example(example: dict[str, str]) -> dict[str, Any]:
    """Convert one raw GSM8K example to conversational prompt/completion."""
    question = example["question"].strip()
    answer = example["answer"].strip()
    final_answer = extract_final_answer(answer)

    if not question:
        raise ValueError("The question must not be empty.")
    if not answer:
        raise ValueError("The answer must not be empty.")
    if final_answer is None:
        raise ValueError("The answer does not contain a valid #### marker.")

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": answer},
        ],
        "equations": extract_equations(answer),
        "final_answer": final_answer,
    }


def preprocess_split(dataset: Dataset) -> Dataset:
    """Format one raw split and remove its original text columns."""
    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting GSM8K examples",
    )


def prepare_gsm8k_dataset(
    *,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    train_subset_size: int | None = None,
    test_subset_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> DatasetDict:
    """Create formatted train, validation, and official test splits."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between 0 and 1.")

    raw_train = load_gsm8k_dataset(
        split="train",
        subset_size=train_subset_size,
        seed=seed,
    )
    raw_test = load_gsm8k_dataset(
        split="test",
        subset_size=test_subset_size,
        seed=seed,
    )

    train_validation = raw_train.train_test_split(
        test_size=validation_size,
        seed=seed,
    )

    return DatasetDict(
        {
            "train": preprocess_split(train_validation["train"]),
            "validation": preprocess_split(train_validation["test"]),
            "test": preprocess_split(raw_test),
        }
    )


def inspect_preprocessed_dataset(dataset: DatasetDict) -> None:
    """Log split sizes, schema, and one formatted training example."""
    LOGGER.info("Prepared dataset: %s", dataset)
    for split_name, split in dataset.items():
        LOGGER.info("%s examples: %s", split_name, f"{len(split):,}")

    example = dataset["train"][0]
    LOGGER.info("Columns: %s", dataset["train"].column_names)
    LOGGER.info("Prompt: %s", example["prompt"])
    LOGGER.info("Completion: %s", example["completion"])
    LOGGER.info("Equations: %s", example["equations"])
    LOGGER.info("Final answer: %s", example["final_answer"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess GSM8K for supervised fine-tuning."
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
    )
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--test-subset-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Run preprocessing and inspect the resulting structure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    args = parse_args()
    dataset = prepare_gsm8k_dataset(
        validation_size=args.validation_size,
        train_subset_size=args.train_subset_size,
        test_subset_size=args.test_subset_size,
        seed=args.seed,
    )
    inspect_preprocessed_dataset(dataset)


if __name__ == "__main__":
    main()
