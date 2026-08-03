"""Extract normalized final answers from generated GSM8K text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.evaluation.numeric import (
    NUMERIC_OR_FRACTION_PATTERN,
    normalize_numeric_value,
)

_VALUE_BOUNDARY = r"(?![\w/-]|\.\d)"
_TERMINAL_ANSWER_PATTERN = re.compile(rf"####\s*({NUMERIC_OR_FRACTION_PATTERN})\s*\Z")
_STANDALONE_VALUE_PATTERN = re.compile(
    rf"(?<![\w.-])({NUMERIC_OR_FRACTION_PATTERN}){_VALUE_BOUNDARY}"
)


@dataclass(frozen=True)
class TerminalAnswer:
    """Normalized terminal answer and its numeric span in the source text."""

    value: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("A terminal-answer span must satisfy 0 <= start < end.")


def extract_terminal_answer(text: str) -> TerminalAnswer | None:
    """Extract the strict ``#### number`` target without losing its source span."""
    match = _TERMINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None

    normalized = normalize_numeric_value(match.group(1))
    if normalized is None:
        return None
    start, end = match.span(1)
    return TerminalAnswer(value=normalized, start=start, end=end)


def extract_final_answer(text: str, *, require_marker: bool = True) -> str | None:
    """Extract a terminal marked answer or the last standalone numeric value."""
    terminal_answer = extract_terminal_answer(text)
    if terminal_answer is not None:
        return terminal_answer.value
    if require_marker:
        return None

    matches = _STANDALONE_VALUE_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_numeric_value(matches[-1])
