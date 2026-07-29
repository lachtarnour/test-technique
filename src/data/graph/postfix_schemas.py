"""Immutable schemas for executable postfix supervision programs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

from .schemas import (
    ArithmeticOperator,
    ProblemNumberNode,
    fraction_to_dict,
)


class PostfixOperator(str, Enum):
    """Operators emitted by the expression-tree compiler."""

    COPY = "copy"
    ADD = "add"
    SUBTRACT = "sub"
    MULTIPLY = "mul"
    DIVIDE = "div"
    FLOOR_DIVIDE = "floor_div"
    POSITIVE = "pos"
    NEGATE = "neg"

    @classmethod
    def from_arithmetic(cls, operator: ArithmeticOperator) -> PostfixOperator:
        """Convert a parser operator without losing execution semantics."""
        return cls(operator.value)

    @property
    def arity(self) -> int:
        """Return the exact number of operands consumed by this action."""
        if self in {self.COPY, self.POSITIVE, self.NEGATE}:
            return 1
        return 2

    @property
    def supervision_operator(self) -> PostfixOperator:
        """Return the operator class exposed to the auxiliary head."""
        if self is self.FLOOR_DIVIDE:
            return self.DIVIDE
        return self


# Ordered operand references


@dataclass(frozen=True)
class ProblemNumberReference:
    """Pointer to one exact numeric occurrence in the problem statement."""

    problem_number_index: int

    def __post_init__(self) -> None:
        if self.problem_number_index < 0:
            raise ValueError("A problem-number index must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "problem_number",
            "problem_number_index": self.problem_number_index,
        }


@dataclass(frozen=True)
class PreviousResultReference:
    """Pointer to the final result of one previous equation step."""

    step_index: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("A previous-result index must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "previous_result",
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class LocalResultReference:
    """Pointer to an earlier action inside the current equation step."""

    action_index: int

    def __post_init__(self) -> None:
        if self.action_index < 0:
            raise ValueError("A local-result index must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "local_result",
            "action_index": self.action_index,
        }


@dataclass(frozen=True)
class LiteralReference:
    """Numeric constant with no problem or previous-step provenance."""

    value: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "literal",
            "value": fraction_to_dict(self.value),
        }


ProvenanceReference = ProblemNumberReference | PreviousResultReference


@dataclass(frozen=True)
class UnresolvedReference:
    """Operand whose exact value is known but whose provenance is ambiguous."""

    value: Fraction
    candidates: tuple[ProvenanceReference, ...]

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError("An unresolved reference requires two candidates.")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Unresolved candidates must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "unresolved",
            "value": fraction_to_dict(self.value),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


OperandReference = (
    ProblemNumberReference
    | PreviousResultReference
    | LocalResultReference
    | LiteralReference
    | UnresolvedReference
)


# Program representation


def _iter_reference_and_candidates(
    reference: OperandReference,
) -> tuple[
    ProblemNumberReference | PreviousResultReference | LocalResultReference,
    ...,
]:
    if isinstance(reference, UnresolvedReference):
        return reference.candidates
    if isinstance(reference, LiteralReference):
        return ()
    return (reference,)


@dataclass(frozen=True)
class PostfixAction:
    """One ordered operation in a postfix/SSA-style equation program."""

    index: int
    operator: PostfixOperator
    operands: tuple[OperandReference, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("An action index must be non-negative.")
        if len(self.operands) != self.operator.arity:
            raise ValueError(
                f"{self.operator.value} expects {self.operator.arity} operands, "
                f"received {len(self.operands)}."
            )
        for operand in self.operands:
            for reference in _iter_reference_and_candidates(operand):
                if (
                    isinstance(reference, LocalResultReference)
                    and reference.action_index >= self.index
                ):
                    raise ValueError("A local result must point to an earlier action.")

    @property
    def operand_mask(self) -> tuple[bool, ...]:
        """Return one supervision bit per ordered operand occurrence."""
        return tuple(
            not isinstance(operand, UnresolvedReference) for operand in self.operands
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "execution_operator": self.operator.value,
            "supervision_operator": self.operator.supervision_operator.value,
            "operands": [operand.to_dict() for operand in self.operands],
            "operand_mask": list(self.operand_mask),
        }


@dataclass(frozen=True)
class PostfixProgramStep:
    """Compiled actions and numerical supervision metadata for one step."""

    index: int
    expression: str | None
    target_result: Fraction | None
    actions: tuple[PostfixAction, ...]
    valid: bool
    error: str | None
    is_final: bool

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("A program-step index must be non-negative.")
        expected_action_indices = tuple(range(len(self.actions)))
        if tuple(action.index for action in self.actions) != expected_action_indices:
            raise ValueError("Postfix action indices must be contiguous and ordered.")

        if self.valid:
            if self.target_result is None or not self.actions:
                raise ValueError(
                    "A valid program step requires a target and at least one action."
                )
            if self.error is not None:
                raise ValueError("A valid program step cannot contain an error.")
        else:
            if self.error is None:
                raise ValueError("An invalid program step must contain an error.")
            if self.actions:
                raise ValueError("An invalid program step cannot contain actions.")
            if self.is_final:
                raise ValueError("An invalid program step cannot be final.")

        for action in self.actions:
            for operand in action.operands:
                for reference in _iter_reference_and_candidates(operand):
                    if (
                        isinstance(reference, PreviousResultReference)
                        and reference.step_index >= self.index
                    ):
                        raise ValueError(
                            "A previous result must point to an earlier step."
                        )

    @property
    def target_scale(self) -> float | None:
        """Scale used by normalized numerical and execution losses."""
        if self.target_result is None:
            return None
        return float(max(abs(self.target_result), Fraction(1)))

    @property
    def operand_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Return action-aligned operand supervision masks."""
        return tuple(action.operand_mask for action in self.actions)

    @property
    def masked_operand_count(self) -> int:
        """Return the number of locally masked ambiguous operand positions."""
        return sum(
            not mask for action_mask in self.operand_mask for mask in action_mask
        )

    @property
    def operand_count(self) -> int:
        """Return the total number of ordered operand positions."""
        return sum(len(action.operands) for action in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "expression": self.expression,
            "target_result": fraction_to_dict(self.target_result),
            "target_scale": self.target_scale,
            "actions": [action.to_dict() for action in self.actions],
            "operand_mask": [list(mask) for mask in self.operand_mask],
            "valid": self.valid,
            "error": self.error,
            "is_final": self.is_final,
        }


@dataclass(frozen=True)
class PostfixProgram:
    """Self-contained postfix program for one calculation graph."""

    problem_numbers: tuple[ProblemNumberNode, ...]
    steps: tuple[PostfixProgramStep, ...]

    def __post_init__(self) -> None:
        if tuple(step.index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("Program-step indices must be contiguous and ordered.")
        for step in self.steps:
            for action in step.actions:
                for operand in action.operands:
                    for reference in _iter_reference_and_candidates(operand):
                        if isinstance(
                            reference, ProblemNumberReference
                        ) and reference.problem_number_index >= len(
                            self.problem_numbers
                        ):
                            raise ValueError(
                                "A problem-number reference is outside the table."
                            )

    @property
    def action_count(self) -> int:
        return sum(len(step.actions) for step in self.steps)

    @property
    def operand_count(self) -> int:
        return sum(step.operand_count for step in self.steps)

    @property
    def masked_operand_count(self) -> int:
        return sum(step.masked_operand_count for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_numbers": [
                problem_number.to_dict() for problem_number in self.problem_numbers
            ],
            "steps": [step.to_dict() for step in self.steps],
            "action_count": self.action_count,
            "operand_count": self.operand_count,
            "masked_operand_count": self.masked_operand_count,
        }


# Execution and validation reports


@dataclass(frozen=True)
class PostfixStepEvaluation:
    """Exact execution result for one compiled program step."""

    index: int
    result: Fraction | None
    target_result: Fraction | None
    matches_target: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "result": fraction_to_dict(self.result),
            "target_result": fraction_to_dict(self.target_result),
            "matches_target": self.matches_target,
            "error": self.error,
        }


@dataclass(frozen=True)
class PostfixProgramEvaluation:
    """Exact execution report for every compiled program step."""

    steps: tuple[PostfixStepEvaluation, ...]

    @property
    def all_steps_executable(self) -> bool:
        return bool(self.steps) and all(step.result is not None for step in self.steps)

    @property
    def all_steps_match_targets(self) -> bool:
        return bool(self.steps) and all(step.matches_target for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "all_steps_executable": self.all_steps_executable,
            "all_steps_match_targets": self.all_steps_match_targets,
        }


@dataclass(frozen=True)
class PostfixVerificationStep:
    """Exact comparison between the old graph and compiled program results."""

    index: int
    graph_result: Fraction | None
    program_result: Fraction | None
    matches_graph: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "graph_result": fraction_to_dict(self.graph_result),
            "program_result": fraction_to_dict(self.program_result),
            "matches_graph": self.matches_graph,
        }


@dataclass(frozen=True)
class PostfixProgramVerification:
    """Step-level equivalence report for graph and postfix execution."""

    steps: tuple[PostfixVerificationStep, ...]

    @property
    def all_steps_match_graph(self) -> bool:
        return all(step.matches_graph for step in self.steps)

    @property
    def executable_step_count(self) -> int:
        return sum(step.graph_result is not None for step in self.steps)

    @property
    def matching_executable_step_count(self) -> int:
        return sum(
            step.graph_result is not None and step.matches_graph for step in self.steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "all_steps_match_graph": self.all_steps_match_graph,
            "executable_step_count": self.executable_step_count,
            "matching_executable_step_count": (self.matching_executable_step_count),
        }


@dataclass(frozen=True)
class PostfixValidationIssue:
    """One actionable structural difference between a graph and its program."""

    code: str
    message: str
    step_index: int | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("A validation issue code must not be empty.")
        if not self.message:
            raise ValueError("A validation issue message must not be empty.")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("A validation issue step index must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class PostfixStructuralValidation:
    """Independent structural audit of a compiled calculation program."""

    checked_step_count: int
    issues: tuple[PostfixValidationIssue, ...]

    def __post_init__(self) -> None:
        if self.checked_step_count < 0:
            raise ValueError("checked_step_count must be non-negative.")

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checked_step_count": self.checked_step_count,
            "issue_count": len(self.issues),
            "issue_counts": self.issue_counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }
