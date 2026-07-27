"""Build and execute conservative non-semantic calculation graphs."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from src.data.schemas import (
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
from src.evaluation.numeric import (
    NUMERIC_OR_FRACTION_PATTERN,
    parse_numeric_fraction,
)

_PROBLEM_NUMBER_PATTERN = re.compile(
    rf"(?<![\w.])({NUMERIC_OR_FRACTION_PATTERN})(?!\w)"
)


class GraphExecutionError(ValueError):
    """Raised when a partial graph cannot be executed safely."""


def _fraction_to_dict(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


@dataclass(frozen=True)
class GraphStepEvaluation:
    """Exact execution result for one graph step."""

    index: int
    result: Fraction | None
    target_result: Fraction | None
    matches_target: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "result": _fraction_to_dict(self.result),
            "target_result": _fraction_to_dict(self.target_result),
            "matches_target": self.matches_target,
            "error": self.error,
        }


@dataclass(frozen=True)
class GraphEvaluation:
    """Execution report for every step of one calculation graph."""

    steps: tuple[GraphStepEvaluation, ...]

    @property
    def all_steps_executable(self) -> bool:
        """Return whether every graph step produced an exact result."""
        return bool(self.steps) and all(step.result is not None for step in self.steps)

    @property
    def all_steps_match_targets(self) -> bool:
        """Return whether every graph step reproduces its exact target."""
        return bool(self.steps) and all(step.matches_target for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "all_steps_executable": self.all_steps_executable,
            "all_steps_match_targets": self.all_steps_match_targets,
        }


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
    return tuple(problem_numbers)


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
                {
                    node.step_index
                    for node in nodes
                    if isinstance(node, ReferenceNode)
                }
            )
        )
        unresolved_count = sum(
            isinstance(node, UnresolvedNode) for node in nodes
        )
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


def _apply_operator(
    operator: ArithmeticOperator,
    values: tuple[Fraction, ...],
) -> Fraction:
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
    raise GraphExecutionError(f"Unsupported graph operator: {operator.value}")


def evaluate_graph_expression(
    node: ExpressionNode,
    *,
    previous_results: tuple[Fraction | None, ...],
) -> Fraction:
    """Execute one resolved graph expression exactly."""
    if isinstance(node, (ProblemNumberNode, LiteralNode, UnresolvedNode)):
        return node.value
    if isinstance(node, ReferenceNode):
        if node.step_index >= len(previous_results):
            raise GraphExecutionError("Reference points to a future step.")
        result = previous_results[node.step_index]
        if result is None:
            raise GraphExecutionError("Reference points to an invalid step.")
        return result
    if isinstance(node, NumberNode):
        raise GraphExecutionError("Unresolved syntactic number in graph.")
    if not isinstance(node, OperationNode):
        raise GraphExecutionError("Unsupported graph node.")

    values = tuple(
        evaluate_graph_expression(
            operand,
            previous_results=previous_results,
        )
        for operand in node.operands
    )
    return _apply_operator(node.operator, values)


def evaluate_calculation_graph(graph: CalculationGraph) -> GraphEvaluation:
    """Execute every graph step in topological order and compare its target."""
    results: list[Fraction | None] = []
    evaluations: list[GraphStepEvaluation] = []

    for step in graph.steps:
        if not step.valid or step.expression_tree is None:
            results.append(None)
            evaluations.append(
                GraphStepEvaluation(
                    index=step.index,
                    result=None,
                    target_result=step.target_result,
                    matches_target=False,
                    error=step.error or "invalid_graph_step",
                )
            )
            continue

        try:
            result = evaluate_graph_expression(
                step.expression_tree,
                previous_results=tuple(results),
            )
        except (ArithmeticError, GraphExecutionError, ZeroDivisionError) as error:
            results.append(None)
            evaluations.append(
                GraphStepEvaluation(
                    index=step.index,
                    result=None,
                    target_result=step.target_result,
                    matches_target=False,
                    error=str(error),
                )
            )
            continue

        matches_target = result == step.target_result
        results.append(result)
        evaluations.append(
            GraphStepEvaluation(
                index=step.index,
                result=result,
                target_result=step.target_result,
                matches_target=matches_target,
                error=None if matches_target else "target_mismatch",
            )
        )

    return GraphEvaluation(steps=tuple(evaluations))
