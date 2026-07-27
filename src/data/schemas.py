"""Non-semantic arithmetic schemas used by the V1 training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any


class ArithmeticOperator(str, Enum):
    """Operators supported by the safe GSM8K arithmetic grammar."""

    ADD = "add"
    SUBTRACT = "sub"
    MULTIPLY = "mul"
    DIVIDE = "div"
    FLOOR_DIVIDE = "floor_div"
    POSITIVE = "pos"
    NEGATE = "neg"

    @property
    def arity(self) -> int:
        """Return the number of operands required by this operator."""
        return 1 if self in {self.POSITIVE, self.NEGATE} else 2


@dataclass(frozen=True)
class SourceSpan:
    """Half-open character offsets into the original answer."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("A source span must satisfy 0 <= start <= end.")

    def extract(self, text: str) -> str:
        """Return the text covered by this span."""
        return text[self.start : self.end]

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""
        return {"start": self.start, "end": self.end}


def _fraction_to_dict(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


@dataclass(frozen=True)
class NumberNode:
    """One exact number occurrence before provenance resolution."""

    value: Fraction

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": "number",
            "value": _fraction_to_dict(self.value),
        }


@dataclass(frozen=True)
class ProblemNumberNode:
    """A number occurrence found directly in the problem statement."""

    value: Fraction
    source_span: SourceSpan
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": "problem_number",
            "value": _fraction_to_dict(self.value),
            "source_span": self.source_span.to_dict(),
            "source_text": self.source_text,
        }


@dataclass(frozen=True)
class ReferenceNode:
    """A high-confidence reference to one previous calculation step."""

    step_index: int
    value: Fraction

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("A reference step index must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": "reference",
            "step_index": self.step_index,
            "value": _fraction_to_dict(self.value),
        }


ProvenanceCandidate = ProblemNumberNode | ReferenceNode


@dataclass(frozen=True)
class LiteralNode:
    """A fixed number with no identifiable problem or step provenance."""

    value: Fraction

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": "literal",
            "value": _fraction_to_dict(self.value),
        }


@dataclass(frozen=True)
class UnresolvedNode:
    """A number whose provenance has multiple plausible candidates."""

    value: Fraction
    candidates: tuple[ProvenanceCandidate, ...]

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError("An unresolved node requires at least two candidates.")
        if any(candidate.value != self.value for candidate in self.candidates):
            raise ValueError("Every provenance candidate must have the same value.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": "unresolved",
            "value": _fraction_to_dict(self.value),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class OperationNode:
    """One arithmetic operation over non-semantic child nodes."""

    operator: ArithmeticOperator
    operands: tuple[ExpressionNode, ...]

    def __post_init__(self) -> None:
        if len(self.operands) != self.operator.arity:
            raise ValueError(
                f"{self.operator.value} expects {self.operator.arity} operands, "
                f"received {len(self.operands)}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": "operation",
            "operator": self.operator.value,
            "operands": [operand.to_dict() for operand in self.operands],
        }


ExpressionNode = (
    NumberNode
    | ProblemNumberNode
    | ReferenceNode
    | LiteralNode
    | UnresolvedNode
    | OperationNode
)
SyntacticExpressionNode = NumberNode | OperationNode
GraphExpressionNode = (
    ProblemNumberNode
    | ReferenceNode
    | LiteralNode
    | UnresolvedNode
    | OperationNode
)


@dataclass(frozen=True)
class MathStep:
    """One annotated GSM8K calculation prepared for V1 supervision."""

    index: int
    raw_annotation: str
    annotation_span: SourceSpan
    expression: str | None
    expression_span: SourceSpan | None
    claimed_result_text: str | None
    claimed_result_span: SourceSpan | None
    claimed_result: Fraction | None
    target_result: Fraction | None
    expression_tree: SyntacticExpressionNode | None
    valid: bool
    error: str | None
    is_final: bool = False

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("A math-step index must be non-negative.")
        if self.valid:
            required = (
                self.expression,
                self.expression_span,
                self.claimed_result_text,
                self.claimed_result_span,
                self.claimed_result,
                self.target_result,
                self.expression_tree,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "A valid math step must contain all structured fields."
                )
            if self.error is not None:
                raise ValueError("A valid math step cannot contain an error.")
        elif self.error is None:
            raise ValueError("An invalid math step must contain an error.")
        if self.is_final and not self.valid:
            raise ValueError("Only a valid math step can be marked as final.")

    @property
    def operator(self) -> ArithmeticOperator | None:
        """Return the root operator used as the step-level operator label."""
        if isinstance(self.expression_tree, OperationNode):
            return self.expression_tree.operator
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "raw_annotation": self.raw_annotation,
            "annotation_span": self.annotation_span.to_dict(),
            "expression": self.expression,
            "expression_span": (
                self.expression_span.to_dict()
                if self.expression_span is not None
                else None
            ),
            "claimed_result_text": self.claimed_result_text,
            "claimed_result_span": (
                self.claimed_result_span.to_dict()
                if self.claimed_result_span is not None
                else None
            ),
            "claimed_result": _fraction_to_dict(self.claimed_result),
            "target_result": _fraction_to_dict(self.target_result),
            "expression_tree": (
                self.expression_tree.to_dict()
                if self.expression_tree is not None
                else None
            ),
            "operator": self.operator.value if self.operator is not None else None,
            "valid": self.valid,
            "error": self.error,
            "is_final": self.is_final,
        }


@dataclass(frozen=True)
class GraphStep:
    """One calculation step after conservative provenance resolution."""

    index: int
    expression: str | None
    target_result: Fraction | None
    expression_tree: GraphExpressionNode | None
    dependencies: tuple[int, ...]
    unresolved_operand_count: int
    valid: bool
    error: str | None
    is_final: bool

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("A graph-step index must be non-negative.")
        if any(index < 0 or index >= self.index for index in self.dependencies):
            raise ValueError("Graph dependencies must reference previous steps.")
        if tuple(sorted(set(self.dependencies))) != self.dependencies:
            raise ValueError("Graph dependencies must be unique and sorted.")
        if self.unresolved_operand_count < 0:
            raise ValueError("unresolved_operand_count must be non-negative.")
        if self.valid:
            if self.expression_tree is None or self.target_result is None:
                raise ValueError("A valid graph step requires a tree and target.")
            if self.error is not None:
                raise ValueError("A valid graph step cannot contain an error.")
        elif self.error is None:
            raise ValueError("An invalid graph step must contain an error.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "expression": self.expression,
            "target_result": _fraction_to_dict(self.target_result),
            "expression_tree": (
                self.expression_tree.to_dict()
                if self.expression_tree is not None
                else None
            ),
            "dependencies": list(self.dependencies),
            "unresolved_operand_count": self.unresolved_operand_count,
            "valid": self.valid,
            "error": self.error,
            "is_final": self.is_final,
        }


@dataclass(frozen=True)
class CalculationGraph:
    """A partial high-confidence graph for one question and answer."""

    problem_numbers: tuple[ProblemNumberNode, ...]
    steps: tuple[GraphStep, ...]

    @property
    def unresolved_operand_count(self) -> int:
        """Return the total number of ambiguous number occurrences."""
        return sum(step.unresolved_operand_count for step in self.steps)

    @property
    def provenance_complete(self) -> bool:
        """Return whether every operand has one unambiguous provenance."""
        return (
            bool(self.steps)
            and all(step.valid for step in self.steps)
            and self.unresolved_operand_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "problem_numbers": [
                problem_number.to_dict()
                for problem_number in self.problem_numbers
            ],
            "steps": [step.to_dict() for step in self.steps],
            "unresolved_operand_count": self.unresolved_operand_count,
            "provenance_complete": self.provenance_complete,
        }
