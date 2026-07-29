"""Extract normalized final answers from generated GSM8K text."""

from __future__ import annotations

import re

from src.evaluation.numeric import (
    NUMERIC_OR_FRACTION_PATTERN,
    normalize_numeric_value,
)

_VALUE_BOUNDARY = r"(?![\w/-]|\.\d)"
_TERMINAL_ANSWER_PATTERN = re.compile(rf"####\s*({NUMERIC_OR_FRACTION_PATTERN})\s*\Z")
_STANDALONE_VALUE_PATTERN = re.compile(
    rf"(?<![\w.-])({NUMERIC_OR_FRACTION_PATTERN}){_VALUE_BOUNDARY}"
)


def extract_final_answer(text: str, *, require_marker: bool = True) -> str | None:
    """Extract a terminal marked answer or the last standalone numeric value."""
    terminal_match = _TERMINAL_ANSWER_PATTERN.search(text)
    if terminal_match is not None:
        return normalize_numeric_value(terminal_match.group(1))
    if require_marker:
        return None

    matches = _STANDALONE_VALUE_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_numeric_value(matches[-1])
