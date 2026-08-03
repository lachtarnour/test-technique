"""Prompt/completion formatting for language-model training."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Final

from src.config import CONFIG
from src.evaluation.arithmetic import FORMULA_PATTERN
from src.evaluation.numeric import (
    NUMERIC_OR_FRACTION_PATTERN,
    parse_numeric_fraction,
)

_TRIVIAL_IDENTITY_PATTERN: Final = re.compile(
    r"<<\s*([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"
    r"\s*=\s*"
    r"([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))\s*>>"
)
_VISIBLE_RESULT_PATTERN: Final = re.compile(
    rf"(?P<leading>[ \t]*(?:[$€£][ \t]*)?)"
    rf"(?P<value>{NUMERIC_OR_FRACTION_PATTERN})"
)
_CALCULATION_CUE_PATTERN: Final = re.compile(
    r"(?:=|equals?)[ \t]*[$€£]?[ \t]*$",
    re.IGNORECASE,
)
_DANGLING_OPERATOR_PATTERN: Final = re.compile(r"[+\-*/(][ \t]*$")
_WORD_OPERATOR_PATTERNS: Final = (
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    (re.compile(r"\bx\b", re.IGNORECASE), "*"),
)
_CHARACTER_OPERATORS: Final = {
    "×": "*",
    "✕": "*",
    "·": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
}
_MATH_CHARACTERS: Final = frozenset("0123456789.+-*/()")


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


def _math_characters_with_positions(text: str) -> tuple[str, tuple[int, ...]]:
    """Normalize visible arithmetic while retaining source character positions."""
    replacements: dict[int, str] = {}
    blocked_positions: set[int] = set()
    for pattern, operator in _WORD_OPERATOR_PATTERNS:
        for match in pattern.finditer(text):
            replacements[match.start()] = operator
            blocked_positions.update(range(match.start() + 1, match.end()))

    characters: list[str] = []
    positions: list[int] = []
    for index, raw_character in enumerate(text):
        if index in blocked_positions:
            continue
        if index in replacements:
            characters.append(replacements[index])
            positions.append(index)
            continue

        character = _CHARACTER_OPERATORS.get(raw_character, raw_character)
        if character == "/":
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and text[next_index].isalpha():
                # ``hours/person`` and ``$/hour`` are units, not DIV actions.
                continue
        if character in _MATH_CHARACTERS:
            characters.append(character)
            positions.append(index)
    return "".join(characters), tuple(positions)


def _visible_expression_start(local_text: str, expression: str) -> int | None:
    """Locate a repeated displayed expression at the end of one step prefix."""
    expected, _ = _math_characters_with_positions(expression)
    visible, positions = _math_characters_with_positions(local_text)
    if not expected or not visible.endswith(expected):
        return None
    return positions[len(visible) - len(expected)]


def _numeric_values(text: str) -> tuple[Fraction, ...]:
    values: list[Fraction] = []
    for match in re.finditer(NUMERIC_OR_FRACTION_PATTERN, text):
        value = parse_numeric_fraction(match.group(0))
        if value is not None:
            values.append(value)
    return tuple(values)


def _merged_spans(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def canonicalize_calculation_annotations(answer: str) -> str:
    """Keep every calculation exactly once, inside its ``<<...>>`` annotation.

    GSM8K usually displays an expression before an annotation and repeats its
    numerical result immediately afterwards.  Exact suffixes are removed while
    preserving surrounding prose.  Ambiguous calculation prefixes ending in
    ``=`` or ``equals`` fall back to the annotation-only representation instead
    of risking target leakage.
    """
    matches = list(FORMULA_PATTERN.finditer(answer))
    deletions: list[tuple[int, int]] = []
    previous_annotation_end = 0

    for match in matches:
        expression, separator, claimed_text = match.group(1).rpartition("=")
        if not separator or match.group(2) != ">>":
            previous_annotation_end = match.end()
            continue
        target = parse_numeric_fraction(claimed_text.strip())

        line_start = answer.rfind("\n", 0, match.start()) + 1
        context_start = max(line_start, previous_annotation_end)
        while context_start < match.start() and answer[context_start] in {" ", "\t"}:
            context_start += 1
        local_prefix = answer[context_start : match.start()]
        relative_expression_start = _visible_expression_start(
            local_prefix,
            expression,
        )
        if relative_expression_start is not None:
            expression_start = context_start + relative_expression_start
            preceding_text = answer[context_start:expression_start]
            expected_math, _ = _math_characters_with_positions(expression)
            preceding_math, _ = _math_characters_with_positions(preceding_text)
            if (
                (expected_math and expected_math in preceding_math)
                or (target is not None and target in _numeric_values(preceding_text))
                or _CALCULATION_CUE_PATTERN.search(preceding_text)
                or _DANGLING_OPERATOR_PATTERN.search(preceding_text)
            ):
                # The prose already states the calculation or its target before
                # demonstrating it in the authoritative annotation.
                expression_start = context_start
            deletions.append((expression_start, match.start()))
        elif (
            re.search(r"\d", local_prefix)
            and _CALCULATION_CUE_PATTERN.search(local_prefix)
        ) or (
            _DANGLING_OPERATOR_PATTERN.search(local_prefix)
            and expression.lstrip().startswith(("+", "-", "*", "/", "("))
        ):
            # Units, percentages or reordered operands made exact alignment
            # unsafe.  Retain the authoritative annotation and drop the prefix.
            deletions.append((context_start, match.start()))

        visible_result = _VISIBLE_RESULT_PATTERN.match(answer, match.end())
        if visible_result is not None and target is not None:
            repeated_value = parse_numeric_fraction(visible_result.group("value"))
            leading = visible_result.group("leading")
            if repeated_value == target and not any(
                symbol in leading for symbol in "$€£"
            ):
                deletions.append(
                    (
                        visible_result.start("value"),
                        visible_result.end("value"),
                    )
                )

        previous_annotation_end = match.end()

    chunks: list[str] = []
    cursor = 0
    for start, end in _merged_spans(deletions):
        chunks.append(answer[cursor:start])
        cursor = end
    chunks.append(answer[cursor:])
    canonical = "".join(chunks)
    canonical = re.sub(r">>(?=[A-Za-z0-9$€£])", ">> ", canonical)
    canonical = canonical.replace(">><<", ">>\n<<")
    return re.sub(r">>[ \t]+(?=[.,;:!?])", ">>", canonical)


def format_training_example(example: dict[str, str]) -> dict[str, object]:
    """Normalize one raw row into the prompt/completion contract."""
    question = example["question"].strip()
    answer = canonicalize_calculation_annotations(
        remove_trivial_identity_annotations(example["answer"])
    ).strip()
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
