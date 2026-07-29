"""Build conservative non-semantic calculation graphs.

The module follows graph construction in order:

1. extract number occurrences from the question;
2. resolve operand provenance and build graph steps;
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from text_to_num import find_numbers

from src.evaluation.numeric import (
    NUMERIC_OR_FRACTION_PATTERN,
    parse_numeric_fraction,
)

from .schemas import (
    ArithmeticOperator,
    CalculationGraph,
    ExpressionNode,
    GraphExpressionNode,
    GraphStep,
    LiteralNode,
    MathStep,
    NumberNode,
    OperationNode,
    ProblemNumberNode,
    ReferenceNode,
    SourceSpan,
    UnresolvedNode,
)

_PROBLEM_NUMBER_PATTERN = re.compile(
    rf"(?<![\w.])({NUMERIC_OR_FRACTION_PATTERN})(?!\w)"
)
_WORD_PATTERN = re.compile(r"\b[A-Za-z]+\b")
_SPECIAL_NUMBER_WORDS = {
    "half": Fraction(1, 2),
    "twice": Fraction(2),
    "double": Fraction(2),
    "thrice": Fraction(3),
    "triple": Fraction(3),
    "dozen": Fraction(12),
}
_TEXT2NUM_BLOCKED_WORDS = {
    "billions",
    "hundreds",
    "millions",
    "o",
    "ones",
    "thousands",
}
_FRACTION_DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
}
_FRACTION_NUMERATORS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_ORDINAL_CONTEXT_NOUNS = {
    "book",
    "car",
    "class",
    "customer",
    "day",
    "friend",
    "grade",
    "hour",
    "inning",
    "lap",
    "month",
    "person",
    "place",
    "player",
    "position",
    "quarter",
    "round",
    "set",
    "step",
    "student",
    "time",
    "wall",
    "week",
    "year",
}
_MIXED_FRACTION_PATTERN = re.compile(
    r"\s+and\s+"
    r"(?:(?P<numerator>a|an|one|two|three|four|five|six|seven|eight|nine)"
    r"[\s-]+)?"
    r"(?P<denominator>half|halves|thirds?|quarters?|fourths?|fifths?|sixths?|"
    r"sevenths?|eighths?|ninths?|tenths?)\b",
    re.IGNORECASE,
)
_HALF_DOZEN_PATTERN = re.compile(r"\bhalf(?:\s+a)?\s+dozen\b", re.IGNORECASE)
_TIME_AND_A_HALF_PATTERN = re.compile(
    r"\btime[\s-]+and[\s-]+a[\s-]+half\b",
    re.IGNORECASE,
)
_COUPLE_PATTERN = re.compile(
    r"\ba\s+couple\s+of\b",
    re.IGNORECASE,
)
_ONCE_PATTERN = re.compile(
    r"\bonce(?=\s+(?:a|an|every|per)\b)",
    re.IGNORECASE,
)


# Problem-number extraction


@dataclass(frozen=True)
class _WordToken:
    value: str
    start: int
    end: int
    source: str

    def text(self) -> str:
        return self.value

    def nt_separated(self, previous: _WordToken) -> bool:
        separator = self.source[previous.end : self.start]
        return re.fullmatch(r"[\s-]+", separator) is None

    def not_a_number_part(self) -> bool:
        return self.value.lower() in _TEXT2NUM_BLOCKED_WORDS


def _problem_number(
    question: str,
    *,
    start: int,
    end: int,
    value: Fraction,
) -> ProblemNumberNode:
    span = SourceSpan(start, end)
    return ProblemNumberNode(
        value=value,
        source_span=span,
        source_text=span.extract(question),
    )


def _extract_cardinal_problem_numbers(
    question: str,
    tokens: list[_WordToken],
) -> list[ProblemNumberNode]:
    problem_numbers: list[ProblemNumberNode] = []
    for occurrence in find_numbers(tokens, "en", threshold=0):
        if occurrence.is_ordinal:
            continue
        selected_tokens = tokens[occurrence.start : occurrence.end]
        if not selected_tokens:
            continue
        numeric_value = Fraction(str(occurrence.value))
        if numeric_value.denominator != 1:
            continue
        problem_numbers.append(
            _problem_number(
                question,
                start=selected_tokens[0].start,
                end=selected_tokens[-1].end,
                value=numeric_value,
            )
        )
    return problem_numbers


def _next_word(
    question: str,
    *,
    after: int,
) -> str | None:
    match = _WORD_PATTERN.search(question, after)
    if match is None:
        return None
    separator = question[after : match.start()]
    if re.fullmatch(r"[\s-]+", separator) is None:
        return None
    return match.group(0).lower()


def _extract_fraction_problem_numbers(
    question: str,
    tokens: list[_WordToken],
    cardinals: list[ProblemNumberNode],
) -> list[ProblemNumberNode]:
    problem_numbers: list[ProblemNumberNode] = []
    for index, token in enumerate(tokens):
        denominator_word = token.value.lower()
        denominator = _FRACTION_DENOMINATORS.get(denominator_word)
        if denominator is None:
            continue

        numerator = 1
        start = token.start
        has_explicit_numerator = False
        numerator_separator: str | None = None
        if index > 0:
            previous = tokens[index - 1]
            separator = question[previous.end : token.start]
            previous_word = previous.value.lower()
            if (
                previous_word in {"a", "an"}
                and re.fullmatch(r"[\s-]+", separator) is not None
            ):
                start = previous.start
                has_explicit_numerator = True
                numerator_separator = separator
            else:
                matching_cardinal = next(
                    (
                        cardinal
                        for cardinal in reversed(cardinals)
                        if cardinal.source_span.end == previous.end
                        and cardinal.source_span.start == previous.start
                        and re.fullmatch(r"[\s-]+", separator) is not None
                        and cardinal.value.denominator == 1
                        and 1 <= cardinal.value <= 9
                    ),
                    None,
                )
                if matching_cardinal is not None:
                    numerator = matching_cardinal.value.numerator
                    start = matching_cardinal.source_span.start
                    has_explicit_numerator = True
                    numerator_separator = separator

        next_word = _next_word(question, after=token.end)
        is_safe_bare_fraction = denominator_word == "half"
        if not has_explicit_numerator and not is_safe_bare_fraction:
            continue
        if (
            has_explicit_numerator
            and denominator == 4
            and numerator > 1
            and numerator_separator is not None
            and "-" not in numerator_separator
            and next_word not in {"as", "of"}
        ):
            continue
        if (
            has_explicit_numerator
            and denominator_word not in {"half", "quarter"}
            and next_word in _ORDINAL_CONTEXT_NOUNS
        ):
            continue

        problem_numbers.append(
            _problem_number(
                question,
                start=start,
                end=token.end,
                value=Fraction(numerator, denominator),
            )
        )
    return problem_numbers


def _extract_mixed_fraction_problem_numbers(
    question: str,
    whole_numbers: list[ProblemNumberNode],
) -> list[ProblemNumberNode]:
    problem_numbers: list[ProblemNumberNode] = []
    for whole_number in whole_numbers:
        if whole_number.value.denominator != 1 or whole_number.value < 0:
            continue
        match = _MIXED_FRACTION_PATTERN.match(
            question,
            whole_number.source_span.end,
        )
        if match is None:
            continue
        numerator_word = (match.group("numerator") or "one").lower()
        numerator = _FRACTION_NUMERATORS[numerator_word]
        denominator = _FRACTION_DENOMINATORS[match.group("denominator").lower()]
        problem_numbers.append(
            _problem_number(
                question,
                start=whole_number.source_span.start,
                end=match.end(),
                value=whole_number.value + Fraction(numerator, denominator),
            )
        )
    return problem_numbers


def _extract_contextual_problem_numbers(
    question: str,
) -> list[ProblemNumberNode]:
    problem_numbers: list[ProblemNumberNode] = []
    for match in _TIME_AND_A_HALF_PATTERN.finditer(question):
        problem_numbers.append(
            _problem_number(
                question,
                start=match.start(),
                end=match.end(),
                value=Fraction(3, 2),
            )
        )
    for match in _HALF_DOZEN_PATTERN.finditer(question):
        preceding = question[max(0, match.start() - 8) : match.start()]
        if re.search(r"\band(?:\s+a)?\s*$", preceding, re.IGNORECASE):
            continue
        problem_numbers.append(
            _problem_number(
                question,
                start=match.start(),
                end=match.end(),
                value=Fraction(6),
            )
        )
    for match in _COUPLE_PATTERN.finditer(question):
        problem_numbers.append(
            _problem_number(
                question,
                start=match.start(),
                end=match.end(),
                value=Fraction(2),
            )
        )
    for match in _ONCE_PATTERN.finditer(question):
        problem_numbers.append(
            _problem_number(
                question,
                start=match.start(),
                end=match.end(),
                value=Fraction(1),
            )
        )
    return problem_numbers


def _extract_special_problem_numbers(
    question: str,
    tokens: list[_WordToken],
) -> list[ProblemNumberNode]:
    problem_numbers: list[ProblemNumberNode] = []
    for token in tokens:
        value = _SPECIAL_NUMBER_WORDS.get(token.value.lower())
        if value is None:
            continue
        problem_numbers.append(
            _problem_number(
                question,
                start=token.start,
                end=token.end,
                value=value,
            )
        )
    return problem_numbers


def _select_non_overlapping_problem_numbers(
    candidates: list[tuple[int, ProblemNumberNode]],
) -> list[ProblemNumberNode]:
    selected: list[ProblemNumberNode] = []
    for _, candidate in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            -(item[1].source_span.end - item[1].source_span.start),
            item[1].source_span.start,
        ),
    ):
        candidate_span = candidate.source_span
        if any(
            candidate_span.start < selected_number.source_span.end
            and selected_number.source_span.start < candidate_span.end
            for selected_number in selected
        ):
            continue
        selected.append(candidate)
    return sorted(
        selected,
        key=lambda number: (
            number.source_span.start,
            number.source_span.end,
        ),
    )


def _extract_word_problem_numbers(
    question: str,
    numeric_problem_numbers: list[ProblemNumberNode],
) -> list[ProblemNumberNode]:
    tokens = [
        _WordToken(
            value=match.group(0),
            start=match.start(),
            end=match.end(),
            source=question,
        )
        for match in _WORD_PATTERN.finditer(question)
    ]
    cardinals = _extract_cardinal_problem_numbers(question, tokens)
    fractions = _extract_fraction_problem_numbers(
        question,
        tokens,
        cardinals,
    )
    contextual = _extract_contextual_problem_numbers(question)
    special = _extract_special_problem_numbers(question, tokens)
    mixed = _extract_mixed_fraction_problem_numbers(
        question,
        [*numeric_problem_numbers, *cardinals],
    )
    return _select_non_overlapping_problem_numbers(
        [
            *((50, number) for number in mixed),
            *((45, number) for number in contextual),
            *((40, number) for number in fractions),
            *((35, number) for number in numeric_problem_numbers),
            *((30, number) for number in cardinals),
            *((20, number) for number in special),
        ]
    )


def extract_problem_numbers(question: str) -> tuple[ProblemNumberNode, ...]:
    """Extract exact numeric occurrences from the problem without semantics."""
    problem_numbers: list[ProblemNumberNode] = []
    for match in _PROBLEM_NUMBER_PATTERN.finditer(question):
        source_text = match.group(1)
        value = parse_numeric_fraction(source_text)
        if value is None:
            continue
        problem_numbers.append(
            ProblemNumberNode(
                value=value,
                source_span=SourceSpan(*match.span(1)),
                source_text=source_text,
            )
        )
    return tuple(_extract_word_problem_numbers(question, problem_numbers))


# Provenance resolution and graph construction


def _provenance_candidates(
    value: Fraction,
    *,
    problem_numbers: tuple[ProblemNumberNode, ...],
    previous_steps: tuple[GraphStep, ...],
) -> tuple[ProblemNumberNode | ReferenceNode, ...]:
    problem_candidates = tuple(
        problem_number
        for problem_number in problem_numbers
        if problem_number.value == value
    )
    step_candidates = tuple(
        ReferenceNode(step_index=step.index, value=value)
        for step in previous_steps
        if step.valid and step.target_result == value
    )
    return (*problem_candidates, *step_candidates)


def _direct_fraction_parts(
    node: OperationNode,
) -> tuple[Fraction, Fraction, Fraction] | None:
    if node.operator is not ArithmeticOperator.DIVIDE:
        return None
    left, right = node.operands
    if not isinstance(left, NumberNode) or not isinstance(right, NumberNode):
        return None
    if left.value.denominator != 1 or right.value.denominator != 1 or right.value == 0:
        return None
    value = left.value / right.value
    if value.denominator == 1:
        return None
    return left.value, right.value, value


def _fraction_source_parts(
    problem_number: ProblemNumberNode,
) -> tuple[Fraction, Fraction] | None:
    source_text = problem_number.source_text
    if "/" in source_text:
        numerator_text, denominator_text = source_text.split("/", maxsplit=1)
        numerator = parse_numeric_fraction(numerator_text)
        denominator = parse_numeric_fraction(denominator_text)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator, denominator
    if any(character.isalpha() for character in source_text):
        value = problem_number.value
        return Fraction(value.numerator), Fraction(value.denominator)
    return None


def _fraction_provenance_candidates(
    *,
    numerator: Fraction,
    denominator: Fraction,
    value: Fraction,
    problem_numbers: tuple[ProblemNumberNode, ...],
    previous_steps: tuple[GraphStep, ...],
) -> tuple[ProblemNumberNode | ReferenceNode, ...]:
    problem_candidates = tuple(
        problem_number
        for problem_number in problem_numbers
        if problem_number.value == value
        and _fraction_source_parts(problem_number) == (numerator, denominator)
    )
    if not problem_candidates:
        return ()
    step_candidates = tuple(
        ReferenceNode(step_index=step.index, value=value)
        for step in previous_steps
        if step.valid and step.target_result == value
    )
    return (*problem_candidates, *step_candidates)


def _resolve_expression(
    node: ExpressionNode,
    *,
    problem_numbers: tuple[ProblemNumberNode, ...],
    previous_steps: tuple[GraphStep, ...],
) -> GraphExpressionNode:
    if isinstance(node, NumberNode):
        candidates = _provenance_candidates(
            node.value,
            problem_numbers=problem_numbers,
            previous_steps=previous_steps,
        )
        if not candidates:
            return LiteralNode(node.value)
        if len(candidates) == 1:
            return candidates[0]
        return UnresolvedNode(value=node.value, candidates=candidates)

    if isinstance(node, OperationNode):
        fraction_parts = _direct_fraction_parts(node)
        if fraction_parts is not None:
            numerator, denominator, fraction_value = fraction_parts
            candidates = _fraction_provenance_candidates(
                numerator=numerator,
                denominator=denominator,
                value=fraction_value,
                problem_numbers=problem_numbers,
                previous_steps=previous_steps,
            )
            if candidates:
                if len(candidates) == 1:
                    return candidates[0]
                return UnresolvedNode(
                    value=fraction_value,
                    candidates=candidates,
                )
        return OperationNode(
            operator=node.operator,
            operands=tuple(
                _resolve_expression(
                    operand,
                    problem_numbers=problem_numbers,
                    previous_steps=previous_steps,
                )
                for operand in node.operands
            ),
        )

    raise TypeError("A syntactic tree must contain only numbers and operations.")


def walk_expression_tree(node: ExpressionNode) -> Iterator[ExpressionNode]:
    """Yield a node and all descendants in deterministic pre-order."""
    yield node
    if isinstance(node, OperationNode):
        for operand in node.operands:
            yield from walk_expression_tree(operand)


def build_calculation_graph(
    question: str,
    steps: list[MathStep],
) -> CalculationGraph:
    """Resolve high-confidence operand provenance for ordered math steps."""
    problem_numbers = extract_problem_numbers(question)
    graph_steps: list[GraphStep] = []

    for step in steps:
        if not step.valid or step.expression_tree is None:
            graph_steps.append(
                GraphStep(
                    index=step.index,
                    expression=step.expression,
                    target_result=step.target_result,
                    expression_tree=None,
                    dependencies=(),
                    unresolved_operand_count=0,
                    valid=False,
                    error=step.error or "invalid_math_step",
                    is_final=False,
                )
            )
            continue

        resolved_tree = _resolve_expression(
            step.expression_tree,
            problem_numbers=problem_numbers,
            previous_steps=tuple(graph_steps),
        )
        nodes = tuple(walk_expression_tree(resolved_tree))
        dependencies = tuple(
            sorted(
                {node.step_index for node in nodes if isinstance(node, ReferenceNode)}
            )
        )
        unresolved_count = sum(isinstance(node, UnresolvedNode) for node in nodes)
        graph_steps.append(
            GraphStep(
                index=step.index,
                expression=step.expression,
                target_result=step.target_result,
                expression_tree=resolved_tree,
                dependencies=dependencies,
                unresolved_operand_count=unresolved_count,
                valid=True,
                error=None,
                is_final=step.is_final,
            )
        )

    return CalculationGraph(
        problem_numbers=problem_numbers,
        steps=tuple(graph_steps),
    )
