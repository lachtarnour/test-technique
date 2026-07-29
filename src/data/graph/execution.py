"""Exact execution of resolved calculation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .schemas import (
    ArithmeticOperator,
    CalculationGraph,
    ExpressionNode,
    LiteralNode,
    NumberNode,
    OperationNode,
    ProblemNumberNode,
    ReferenceNode,
    UnresolvedNode,
    fraction_to_dict,
)


class GraphExecutionError(ValueError):
    """Raised when a resolved graph cannot be executed safely."""


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
            "result": fraction_to_dict(self.result),
            "target_result": fraction_to_dict(self.target_result),
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
