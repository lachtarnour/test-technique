"""Build exact, non-semantic V1 math steps from GSM8K annotations."""

from __future__ import annotations

import ast
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

from src.evaluation.answers import extract_final_answer
from src.evaluation.arithmetic import (
    FORMULA_PATTERN,
    THOUSANDS_VALUE_PATTERN,
    parse_annotated_formulas,
)
from src.evaluation.numeric import (
    normalize_fraction,
    numeric_prediction_matches_fraction,
    parse_numeric_fraction,
)

from .schemas import (
    ArithmeticOperator,
    ExpressionNode,
    MathStep,
    NumberNode,
    OperationNode,
    SourceSpan,
    SyntacticExpressionNode,
)

_BINARY_OPERATORS: dict[type[ast.operator], ArithmeticOperator] = {
    ast.Add: ArithmeticOperator.ADD,
    ast.Sub: ArithmeticOperator.SUBTRACT,
    ast.Mult: ArithmeticOperator.MULTIPLY,
    ast.Div: ArithmeticOperator.DIVIDE,
    ast.FloorDiv: ArithmeticOperator.FLOOR_DIVIDE,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], ArithmeticOperator] = {
    ast.UAdd: ArithmeticOperator.POSITIVE,
    ast.USub: ArithmeticOperator.NEGATE,
}


class MathStructureError(ValueError):
    """Raised when a validated AST cannot be represented by the V1 schema."""


def _normalize_expression(expression: str) -> str:
    return THOUSANDS_VALUE_PATTERN.sub(
        lambda match: match.group(0).replace(",", ""),
        expression.strip(),
    )


def _convert_ast_node(node: ast.AST, expression: str) -> ExpressionNode:
    if isinstance(node, ast.Expression):
        return _convert_ast_node(node.body, expression)

    if isinstance(node, ast.Constant):
        literal = ast.get_source_segment(expression, node)
        if literal is None:
            raise MathStructureError("missing numeric literal")
        return NumberNode(Fraction(Decimal(literal)))

    if isinstance(node, ast.UnaryOp):
        operator = _UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise MathStructureError("unsupported unary operator")
        operand = _convert_ast_node(node.operand, expression)
        if isinstance(operand, NumberNode):
            return NumberNode(
                operand.value
                if operator is ArithmeticOperator.POSITIVE
                else -operand.value
            )
        return OperationNode(
            operator=operator,
            operands=(operand,),
        )

    if isinstance(node, ast.BinOp):
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise MathStructureError("unsupported binary operator")
        return OperationNode(
            operator=operator,
            operands=(
                _convert_ast_node(node.left, expression),
                _convert_ast_node(node.right, expression),
            ),
        )

    raise MathStructureError(f"unsupported AST node: {type(node).__name__}")


def parse_expression_tree(expression: str) -> SyntacticExpressionNode:
    """Convert a previously validated arithmetic expression into a V1 tree."""
    normalized_expression = _normalize_expression(expression)
    try:
        tree = ast.parse(normalized_expression, mode="eval")
    except SyntaxError as error:
        raise MathStructureError("invalid expression syntax") from error
    return _convert_ast_node(tree, normalized_expression)


def evaluate_expression_tree(node: SyntacticExpressionNode) -> Fraction:
    """Execute a V1 expression tree exactly with ``Fraction``."""
    if isinstance(node, NumberNode):
        return node.value
    if not isinstance(node, OperationNode):
        raise MathStructureError("A syntactic tree contains a resolved graph node.")

    values = tuple(evaluate_expression_tree(operand) for operand in node.operands)
    operator = node.operator
    if operator is ArithmeticOperator.POSITIVE:
        return values[0]
    if operator is ArithmeticOperator.NEGATE:
        return -values[0]
    if operator is ArithmeticOperator.ADD:
        return values[0] + values[1]
    if operator is ArithmeticOperator.SUBTRACT:
        return values[0] - values[1]
    if operator is ArithmeticOperator.MULTIPLY:
        return values[0] * values[1]
    if operator is ArithmeticOperator.DIVIDE:
        return values[0] / values[1]
    if operator is ArithmeticOperator.FLOOR_DIVIDE:
        return Fraction(values[0] // values[1])
    raise MathStructureError(f"unsupported operator: {operator.value}")


def _trimmed_span(text: str, *, absolute_start: int) -> SourceSpan | None:
    leading = len(text) - len(text.lstrip())
    trailing = len(text) - len(text.rstrip())
    start = absolute_start + leading
    end = absolute_start + len(text) - trailing
    return SourceSpan(start, end) if end > start else None


def _annotation_fields(
    content: str,
    *,
    content_start: int,
) -> tuple[SourceSpan | None, SourceSpan | None, str | None]:
    if "=" not in content:
        return _trimmed_span(content, absolute_start=content_start), None, None

    equals_offset = content.rfind("=")
    expression_text = content[:equals_offset]
    claimed_text = content[equals_offset + 1 :]
    expression_span = _trimmed_span(
        expression_text,
        absolute_start=content_start,
    )
    claimed_span = _trimmed_span(
        claimed_text,
        absolute_start=content_start + equals_offset + 1,
    )
    stripped_claimed_text = claimed_text.strip() or None
    return expression_span, claimed_span, stripped_claimed_text


def parse_math_steps(answer: str) -> list[MathStep]:
    """Extract every annotation as an ordered, non-semantic ``MathStep``."""
    analyses = parse_annotated_formulas(answer)
    matches = list(FORMULA_PATTERN.finditer(answer))
    if len(analyses) != len(matches):
        raise RuntimeError("Formula extraction and structured parsing diverged.")

    steps: list[MathStep] = []
    for index, (match, analysis) in enumerate(zip(matches, analyses, strict=True)):
        annotation_span = SourceSpan(*match.span())
        expression_span, claimed_span, claimed_text = _annotation_fields(
            match.group(1),
            content_start=match.start(1),
        )
        claimed_result = (
            parse_numeric_fraction(claimed_text) if claimed_text is not None else None
        )
        expression_tree: SyntacticExpressionNode | None = None
        target_result: Fraction | None = None
        structure_error: str | None = None

        if analysis.parse_success and analysis.expression is not None:
            try:
                expression_tree = parse_expression_tree(analysis.expression)
            except MathStructureError:
                structure_error = "structure_error"
            if expression_tree is not None:
                try:
                    target_result = evaluate_expression_tree(expression_tree)
                except (ArithmeticError, ZeroDivisionError):
                    target_result = None

        if (
            target_result is not None
            and analysis.evaluated_result is not None
            and normalize_fraction(target_result) != analysis.evaluated_result
        ):
            structure_error = "evaluation_mismatch"

        valid = (
            analysis.is_correct
            and structure_error is None
            and claimed_result is not None
            and target_result is not None
            and expression_tree is not None
        )
        error = None if valid else structure_error or analysis.error
        if not valid and error is None:
            error = "incomplete_structure"

        steps.append(
            MathStep(
                index=index,
                raw_annotation=match.group(0),
                annotation_span=annotation_span,
                expression=analysis.expression,
                expression_span=expression_span,
                claimed_result_text=claimed_text,
                claimed_result_span=claimed_span,
                claimed_result=claimed_result,
                target_result=target_result,
                expression_tree=expression_tree,
                valid=valid,
                error=error,
            )
        )

    final_answer = extract_final_answer(answer)
    if final_answer is not None and steps:
        final_step = steps[-1]
        if (
            final_step.valid
            and final_step.target_result is not None
            and numeric_prediction_matches_fraction(
                final_answer,
                final_step.target_result,
            )
        ):
            steps[-1] = replace(final_step, is_final=True)

    return steps
