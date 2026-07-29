"""Prompt/completion formatting for language-model training."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Final

from src.config import CONFIG

_TRIVIAL_IDENTITY_PATTERN: Final = re.compile(
    r"<<\s*([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"
    r"\s*=\s*"
    r"([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))\s*>>"
)


def remove_trivial_identity_annotations(answer: str) -> str:
    """Remove ``<<value=value>>`` while preserving the visible result."""

    def replace_identity(match: re.Match[str]) -> str:
        try:
            left = Decimal(match.group(1).replace(",", ""))
            right = Decimal(match.group(2).replace(",", ""))
        except InvalidOperation:
            return match.group(0)
        return "" if left == right else match.group(0)

    return _TRIVIAL_IDENTITY_PATTERN.sub(replace_identity, answer)


def format_training_example(example: dict[str, str]) -> dict[str, object]:
    """Normalize one raw row into the prompt/completion contract."""
    question = example["question"].strip()
    answer = remove_trivial_identity_annotations(example["answer"]).strip()
    if not question:
        raise ValueError("The question must not be empty.")
    if not answer:
        raise ValueError("The answer must not be empty.")

    return {
        "prompt": [
            {"role": "system", "content": CONFIG.system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [{"role": "assistant", "content": answer}],
    }
