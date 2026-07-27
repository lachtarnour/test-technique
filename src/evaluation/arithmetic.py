"""Safe parsing and execution of annotated arithmetic formulas."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, DecimalException
from fractions import Fraction
from typing import Any

from src.evaluation.numeric import (
    normalize_fraction,
    numeric_prediction_matches_fraction,
    parse_numeric_fraction,
)

FORMULA_PATTERN = re.compile(r"<<(.*?)(>>|$)", re.DOTALL)
THOUSANDS_VALUE_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
DECIMAL_LITERAL_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")

MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 64
MAX_LITERAL_DIGITS = 50
MAX_RESULT_DIGITS = 256

SUPPORTED_OPERATOR_SYMBOLS = ("+", "-", "*", "/", "//")
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
)


class FormulaParseError(ValueError):
    """Raised when an annotated formula is outside the supported grammar."""


@dataclass(frozen=True)
class FormulaAnalysis:
    """Structured result for one ``<<expression=result>>`` annotation."""

    raw: str
    expression: str | None
    claimed_result: str | None
    evaluated_result: str | None
    parse_success: bool
    execution_success: bool
    arithmetic_correct: bool
    error: str | None

    @property
    def is_correct(self) -> bool:
        """Return whether the formula is parsable, executable and correct."""
        return self.parse_success and self.execution_success and self.arithmetic_correct

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            **asdict(self),
            "is_correct": self.is_correct,
        }


def _remove_thousands_separators(expression: str) -> str:
    return THOUSANDS_VALUE_PATTERN.sub(
        lambda match: match.group(0).replace(",", ""),
        expression,
    )


def _parse_expression(expression: str) -> tuple[str, ast.Expression]:
    normalized_expression = _remove_thousands_separators(expression.strip())
    if not normalized_expression:
        raise FormulaParseError("empty expression")
    if len(normalized_expression) > MAX_EXPRESSION_LENGTH:
        raise FormulaParseError("expression is too long")

    try:
        tree = ast.parse(normalized_expression, mode="eval")
    except SyntaxError as error:
        raise FormulaParseError("invalid expression syntax") from error

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise FormulaParseError("expression is too complex")
    if any(not isinstance(node, ALLOWED_AST_NODES) for node in nodes):
        raise FormulaParseError("unsupported expression element")
    if not any(isinstance(node, ast.BinOp) for node in nodes):
        raise FormulaParseError("expression must contain an arithmetic operation")

    for node in nodes:
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaParseError("only decimal numeric literals are supported")

        literal = ast.get_source_segment(normalized_expression, node)
        if literal is None or DECIMAL_LITERAL_PATTERN.fullmatch(literal) is None:
            raise FormulaParseError("invalid numeric literal")
        digits = sum(character.isdigit() for character in literal)
        if digits > MAX_LITERAL_DIGITS:
            raise FormulaParseError("numeric literal is too long")

    return normalized_expression, tree


def _ensure_bounded_result(value: Fraction) -> Fraction:
    numerator_digits = len(str(abs(value.numerator)))
    denominator_digits = len(str(value.denominator))
    if max(numerator_digits, denominator_digits) > MAX_RESULT_DIGITS:
        raise ArithmeticError("result is too large")
    return value


def _evaluate_node(node: ast.AST, expression: str) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, expression)

    if isinstance(node, ast.Constant):
        literal = ast.get_source_segment(expression, node)
        if literal is None:
            raise ArithmeticError("missing numeric literal")
        return Fraction(Decimal(literal))

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, expression)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _ensure_bounded_result(-operand)
        raise ArithmeticError("unsupported unary operator")

    if not isinstance(node, ast.BinOp):
        raise ArithmeticError("unsupported expression")

    left = _evaluate_node(node.left, expression)
    right = _evaluate_node(node.right, expression)

    if isinstance(node.op, ast.Add):
        result = left + right
    elif isinstance(node.op, ast.Sub):
        result = left - right
    elif isinstance(node.op, ast.Mult):
        result = left * right
    elif isinstance(node.op, ast.Div):
        result = left / right
    elif isinstance(node.op, ast.FloorDiv):
        result = Fraction(left // right)
    else:
        raise ArithmeticError("unsupported binary operator")

    return _ensure_bounded_result(result)


def _analyze_formula(raw: str, content: str) -> FormulaAnalysis:
    if "=" not in content:
        return FormulaAnalysis(
            raw=raw,
            expression=None,
            claimed_result=None,
            evaluated_result=None,
            parse_success=False,
            execution_success=False,
            arithmetic_correct=False,
            error="missing_equals",
        )

    expression, claimed_text = (part.strip() for part in content.rsplit("=", 1))
    claimed_number = parse_numeric_fraction(claimed_text)
    if not expression or claimed_number is None:
        return FormulaAnalysis(
            raw=raw,
            expression=expression or None,
            claimed_result=None,
            evaluated_result=None,
            parse_success=False,
            execution_success=False,
            arithmetic_correct=False,
            error="invalid_claimed_result" if expression else "empty_expression",
        )

    claimed_result = normalize_fraction(claimed_number)
    try:
        normalized_expression, tree = _parse_expression(expression)
    except FormulaParseError:
        return FormulaAnalysis(
            raw=raw,
            expression=expression,
            claimed_result=claimed_result,
            evaluated_result=None,
            parse_success=False,
            execution_success=False,
            arithmetic_correct=False,
            error="invalid_expression",
        )

    try:
        evaluated_number = _evaluate_node(tree, normalized_expression)
        evaluated_result = normalize_fraction(evaluated_number)
    except (ArithmeticError, DecimalException, ZeroDivisionError):
        return FormulaAnalysis(
            raw=raw,
            expression=expression,
            claimed_result=claimed_result,
            evaluated_result=None,
            parse_success=True,
            execution_success=False,
            arithmetic_correct=False,
            error="execution_error",
        )

    arithmetic_correct = numeric_prediction_matches_fraction(
        claimed_text,
        evaluated_number,
    )
    return FormulaAnalysis(
        raw=raw,
        expression=expression,
        claimed_result=claimed_result,
        evaluated_result=evaluated_result,
        parse_success=True,
        execution_success=True,
        arithmetic_correct=arithmetic_correct,
        error=None if arithmetic_correct else "incorrect_result",
    )


def parse_annotated_formulas(text: str) -> list[FormulaAnalysis]:
    """Parse every ``<<expression=result>>`` annotation in ``text`` safely."""
    formulas: list[FormulaAnalysis] = []
    for match in FORMULA_PATTERN.finditer(text):
        if match.group(2) != ">>":
            formulas.append(
                FormulaAnalysis(
                    raw=match.group(0),
                    expression=None,
                    claimed_result=None,
                    evaluated_result=None,
                    parse_success=False,
                    execution_success=False,
                    arithmetic_correct=False,
                    error="unclosed_annotation",
                )
            )
        else:
            formulas.append(_analyze_formula(match.group(0), match.group(1)))
    return formulas
