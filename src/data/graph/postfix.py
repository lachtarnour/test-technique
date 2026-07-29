"""Compile, execute and validate postfix supervision programs.

The three stages use independent checks so structural validation can detect a
compiler mistake even when the resulting number happens to be correct.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from .postfix_schemas import (
    LiteralReference,
    LocalResultReference,
    OperandReference,
    PostfixAction,
    PostfixOperator,
    PostfixProgram,
    PostfixProgramEvaluation,
    PostfixProgramStep,
    PostfixProgramVerification,
    PostfixStepEvaluation,
    PostfixStructuralValidation,
    PostfixValidationIssue,
    PostfixVerificationStep,
    PreviousResultReference,
    ProblemNumberReference,
    ProvenanceReference,
    UnresolvedReference,
)
from .schemas import (
    CalculationGraph,
    ExpressionNode,
    LiteralNode,
    NumberNode,
    OperationNode,
    ProblemNumberNode,
    ReferenceNode,
    UnresolvedNode,
)

_ExpressionSignature = tuple[object, ...]


# Compilation


class PostfixCompilationError(ValueError):
    """Raised when a graph tree cannot be compiled without guessing."""


class PostfixExecutionError(ValueError):
    """Raised when a compiled program violates its execution contract."""


class _ExpressionCompiler:
    """Stateful post-order compiler scoped to one expression tree."""

    def __init__(
        self,
        problem_number_indices: dict[ProblemNumberNode, int],
    ) -> None:
        self._problem_number_indices = problem_number_indices
        self._actions: list[PostfixAction] = []

    def compile(self, expression_tree: ExpressionNode) -> tuple[PostfixAction, ...]:
        root_reference = self._compile_node(expression_tree)
        if not isinstance(root_reference, LocalResultReference):
            self._append_action(PostfixOperator.COPY, (root_reference,))
        return tuple(self._actions)

    def _append_action(
        self,
        operator: PostfixOperator,
        operands: tuple[OperandReference, ...],
    ) -> LocalResultReference:
        action_index = len(self._actions)
        self._actions.append(
            PostfixAction(
                index=action_index,
                operator=operator,
                operands=operands,
            )
        )
        return LocalResultReference(action_index=action_index)

    def _compile_node(self, node: ExpressionNode) -> OperandReference:
        if isinstance(node, ProblemNumberNode):
            try:
                problem_number_index = self._problem_number_indices[node]
            except KeyError as error:
                raise PostfixCompilationError(
                    "A ProblemNumber node is absent from the graph table."
                ) from error
            return ProblemNumberReference(
                problem_number_index=problem_number_index,
            )

        if isinstance(node, ReferenceNode):
            return PreviousResultReference(step_index=node.step_index)

        if isinstance(node, LiteralNode):
            return LiteralReference(value=node.value)

        if isinstance(node, UnresolvedNode):
            return UnresolvedReference(
                value=node.value,
                candidates=tuple(
                    self._compile_candidate(candidate) for candidate in node.candidates
                ),
            )

        if isinstance(node, NumberNode):
            raise PostfixCompilationError(
                "A graph tree still contains an unresolved syntactic Number."
            )

        if not isinstance(node, OperationNode):
            raise PostfixCompilationError(
                f"Unsupported graph node: {type(node).__name__}."
            )

        operands = tuple(self._compile_node(operand) for operand in node.operands)
        return self._append_action(
            PostfixOperator.from_arithmetic(node.operator),
            operands,
        )

    def _compile_candidate(
        self,
        node: ProblemNumberNode | ReferenceNode,
    ) -> ProvenanceReference:
        reference = self._compile_node(node)
        if isinstance(
            reference,
            (ProblemNumberReference, PreviousResultReference),
        ):
            return reference
        raise PostfixCompilationError(
            "An unresolved candidate has an unsupported provenance type."
        )


def _problem_number_index(
    problem_numbers: Sequence[ProblemNumberNode],
) -> dict[ProblemNumberNode, int]:
    indices = {
        problem_number: index for index, problem_number in enumerate(problem_numbers)
    }
    if len(indices) != len(problem_numbers):
        raise PostfixCompilationError(
            "The graph problem-number table contains duplicate occurrences."
        )
    return indices


def compile_expression_tree(
    expression_tree: ExpressionNode,
    *,
    problem_numbers: Sequence[ProblemNumberNode],
) -> tuple[PostfixAction, ...]:
    """Compile one tree in deterministic left-to-right post-order.

    Every operation is emitted after the actions that compute its nested
    operands. A leaf-only expression receives one explicit ``COPY`` action so
    that every valid equation step has a program output.
    """
    compiler = _ExpressionCompiler(_problem_number_index(problem_numbers))
    return compiler.compile(expression_tree)


def compile_calculation_graph(graph: CalculationGraph) -> PostfixProgram:
    """Compile every valid graph step while retaining invalid-step alignment."""
    program_steps: list[PostfixProgramStep] = []
    for step in graph.steps:
        if not step.valid or step.expression_tree is None:
            program_steps.append(
                PostfixProgramStep(
                    index=step.index,
                    expression=step.expression,
                    target_result=step.target_result,
                    actions=(),
                    valid=False,
                    error=step.error or "invalid_graph_step",
                    is_final=False,
                )
            )
            continue

        actions = compile_expression_tree(
            step.expression_tree,
            problem_numbers=graph.problem_numbers,
        )
        program_steps.append(
            PostfixProgramStep(
                index=step.index,
                expression=step.expression,
                target_result=step.target_result,
                actions=actions,
                valid=True,
                error=None,
                is_final=step.is_final,
            )
        )

    return PostfixProgram(
        problem_numbers=graph.problem_numbers,
        steps=tuple(program_steps),
    )


# Exact execution


def _apply_operator(
    operator: PostfixOperator,
    values: tuple[Fraction, ...],
) -> Fraction:
    if operator in {PostfixOperator.COPY, PostfixOperator.POSITIVE}:
        return values[0]
    if operator is PostfixOperator.NEGATE:
        return -values[0]
    if operator is PostfixOperator.ADD:
        return values[0] + values[1]
    if operator is PostfixOperator.SUBTRACT:
        return values[0] - values[1]
    if operator is PostfixOperator.MULTIPLY:
        return values[0] * values[1]
    if operator is PostfixOperator.DIVIDE:
        return values[0] / values[1]
    if operator is PostfixOperator.FLOOR_DIVIDE:
        return Fraction(values[0] // values[1])
    raise PostfixExecutionError(f"Unsupported postfix operator: {operator.value}.")


def _resolve_reference(
    reference: OperandReference,
    *,
    problem_numbers: tuple[ProblemNumberNode, ...],
    previous_results: tuple[Fraction | None, ...],
    local_results: tuple[Fraction, ...],
) -> Fraction:
    if isinstance(reference, ProblemNumberReference):
        try:
            return problem_numbers[reference.problem_number_index].value
        except IndexError as error:
            raise PostfixExecutionError(
                "ProblemNumber points outside the program table."
            ) from error

    if isinstance(reference, PreviousResultReference):
        if reference.step_index >= len(previous_results):
            raise PostfixExecutionError(
                "PreviousResult points to a future program step."
            )
        result = previous_results[reference.step_index]
        if result is None:
            raise PostfixExecutionError(
                "PreviousResult points to an invalid program step."
            )
        return result

    if isinstance(reference, LocalResultReference):
        if reference.action_index >= len(local_results):
            raise PostfixExecutionError(
                "LocalResult points to an action that has not executed."
            )
        return local_results[reference.action_index]

    if isinstance(reference, LiteralReference):
        return reference.value

    if isinstance(reference, UnresolvedReference):
        candidate_values = tuple(
            _resolve_reference(
                candidate,
                problem_numbers=problem_numbers,
                previous_results=previous_results,
                local_results=local_results,
            )
            for candidate in reference.candidates
        )
        if any(value != reference.value for value in candidate_values):
            raise PostfixExecutionError(
                "An Unresolved candidate no longer has the expected value."
            )
        return reference.value

    raise PostfixExecutionError(
        f"Unsupported operand reference: {type(reference).__name__}."
    )


def execute_postfix_step(
    step: PostfixProgramStep,
    *,
    problem_numbers: tuple[ProblemNumberNode, ...],
    previous_results: tuple[Fraction | None, ...],
) -> Fraction:
    """Execute one valid postfix step with exact rational arithmetic."""
    if not step.valid:
        raise PostfixExecutionError("An invalid program step cannot execute.")

    local_results: list[Fraction] = []
    for action in step.actions:
        values = tuple(
            _resolve_reference(
                operand,
                problem_numbers=problem_numbers,
                previous_results=previous_results,
                local_results=tuple(local_results),
            )
            for operand in action.operands
        )
        local_results.append(_apply_operator(action.operator, values))

    if not local_results:
        raise PostfixExecutionError("A valid program step produced no result.")
    return local_results[-1]


def execute_postfix_program(
    program: PostfixProgram,
) -> PostfixProgramEvaluation:
    """Execute all compiled steps in topological order."""
    results: list[Fraction | None] = []
    evaluations: list[PostfixStepEvaluation] = []

    for step in program.steps:
        if not step.valid:
            results.append(None)
            evaluations.append(
                PostfixStepEvaluation(
                    index=step.index,
                    result=None,
                    target_result=step.target_result,
                    matches_target=False,
                    error=step.error or "invalid_program_step",
                )
            )
            continue

        try:
            result = execute_postfix_step(
                step,
                problem_numbers=program.problem_numbers,
                previous_results=tuple(results),
            )
        except (ArithmeticError, PostfixExecutionError, ZeroDivisionError) as error:
            results.append(None)
            evaluations.append(
                PostfixStepEvaluation(
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
            PostfixStepEvaluation(
                index=step.index,
                result=result,
                target_result=step.target_result,
                matches_target=matches_target,
                error=None if matches_target else "target_mismatch",
            )
        )

    return PostfixProgramEvaluation(steps=tuple(evaluations))


# Numerical equivalence with the graph


def verify_postfix_results(
    evaluation: PostfixProgramEvaluation,
    *,
    graph_results: Sequence[Fraction | None],
) -> PostfixProgramVerification:
    """Compare every postfix result with the current graph executor exactly."""
    if len(evaluation.steps) != len(graph_results):
        raise ValueError(
            "Program and graph evaluations must contain the same step count."
        )
    return PostfixProgramVerification(
        steps=tuple(
            PostfixVerificationStep(
                index=step.index,
                graph_result=graph_result,
                program_result=step.result,
                matches_graph=step.result == graph_result,
            )
            for step, graph_result in zip(
                evaluation.steps,
                graph_results,
                strict=True,
            )
        )
    )


# Structural equivalence with the graph


def _graph_expression_signature(
    node: ExpressionNode,
    *,
    problem_number_indices: dict[ProblemNumberNode, int],
) -> _ExpressionSignature:
    if isinstance(node, ProblemNumberNode):
        return ("problem_number", problem_number_indices[node])
    if isinstance(node, ReferenceNode):
        return ("previous_result", node.step_index)
    if isinstance(node, LiteralNode):
        return ("literal", node.value)
    if isinstance(node, UnresolvedNode):
        return (
            "unresolved",
            node.value,
            tuple(
                _graph_expression_signature(
                    candidate,
                    problem_number_indices=problem_number_indices,
                )
                for candidate in node.candidates
            ),
        )
    if isinstance(node, NumberNode):
        return ("invalid_syntactic_number", node.value)
    if isinstance(node, OperationNode):
        return (
            "operation",
            node.operator.value,
            tuple(
                _graph_expression_signature(
                    operand,
                    problem_number_indices=problem_number_indices,
                )
                for operand in node.operands
            ),
        )
    raise TypeError(f"Unsupported graph node: {type(node).__name__}.")


def _graph_postorder_signatures(
    node: ExpressionNode,
    *,
    problem_number_indices: dict[ProblemNumberNode, int],
) -> tuple[_ExpressionSignature, ...]:
    if not isinstance(node, OperationNode):
        return (
            _graph_expression_signature(
                node,
                problem_number_indices=problem_number_indices,
            ),
        )

    child_outputs: list[_ExpressionSignature] = []
    for operand in node.operands:
        if isinstance(operand, OperationNode):
            child_outputs.extend(
                _graph_postorder_signatures(
                    operand,
                    problem_number_indices=problem_number_indices,
                )
            )
    child_outputs.append(
        _graph_expression_signature(
            node,
            problem_number_indices=problem_number_indices,
        )
    )
    return tuple(child_outputs)


def _program_reference_signature(
    reference: OperandReference,
    *,
    local_results: tuple[_ExpressionSignature, ...],
) -> _ExpressionSignature:
    if isinstance(reference, ProblemNumberReference):
        return ("problem_number", reference.problem_number_index)
    if isinstance(reference, PreviousResultReference):
        return ("previous_result", reference.step_index)
    if isinstance(reference, LocalResultReference):
        return local_results[reference.action_index]
    if isinstance(reference, LiteralReference):
        return ("literal", reference.value)
    if isinstance(reference, UnresolvedReference):
        return (
            "unresolved",
            reference.value,
            tuple(
                _program_reference_signature(
                    candidate,
                    local_results=local_results,
                )
                for candidate in reference.candidates
            ),
        )
    raise TypeError(f"Unsupported operand reference: {type(reference).__name__}.")


def _program_action_signatures(
    step: PostfixProgramStep,
) -> tuple[_ExpressionSignature, ...]:
    local_results: list[_ExpressionSignature] = []
    for action in step.actions:
        operand_signatures = tuple(
            _program_reference_signature(
                operand,
                local_results=tuple(local_results),
            )
            for operand in action.operands
        )
        if action.operator is PostfixOperator.COPY:
            action_signature = operand_signatures[0]
        else:
            action_signature = (
                "operation",
                action.operator.value,
                operand_signatures,
            )
        local_results.append(action_signature)
    return tuple(local_results)


def _program_dependencies(step: PostfixProgramStep) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                operand.step_index
                for action in step.actions
                for operand in action.operands
                if isinstance(operand, PreviousResultReference)
            }
        )
    )


def validate_postfix_structure(
    graph: CalculationGraph,
    program: PostfixProgram,
) -> PostfixStructuralValidation:
    """Validate compilation structure independently from numerical execution.

    The comparison covers every post-order subtree, not only the final value.
    It therefore detects reordered operands, changed provenance, missing or
    dead actions, altered operators, incorrect masks, and metadata drift even
    when the final numerical result happens to remain unchanged.
    """
    issues: list[PostfixValidationIssue] = []

    def add_issue(
        code: str,
        message: str,
        *,
        step_index: int | None = None,
    ) -> None:
        issues.append(
            PostfixValidationIssue(
                code=code,
                message=message,
                step_index=step_index,
            )
        )

    if program.problem_numbers != graph.problem_numbers:
        add_issue(
            "problem_number_table_mismatch",
            "The program problem-number table differs from the graph table.",
        )
    if len(program.steps) != len(graph.steps):
        add_issue(
            "step_count_mismatch",
            "The graph and program contain different step counts.",
        )

    problem_number_indices = _problem_number_index(graph.problem_numbers)
    paired_steps = tuple(zip(graph.steps, program.steps, strict=False))
    for graph_step, program_step in paired_steps:
        step_index = graph_step.index
        graph_metadata = (
            graph_step.index,
            graph_step.expression,
            graph_step.target_result,
            graph_step.valid,
            graph_step.error,
            graph_step.is_final,
        )
        program_metadata = (
            program_step.index,
            program_step.expression,
            program_step.target_result,
            program_step.valid,
            program_step.error,
            program_step.is_final,
        )
        if program_metadata != graph_metadata:
            add_issue(
                "step_metadata_mismatch",
                "The compiled step metadata differs from the graph step.",
                step_index=step_index,
            )

        expected_scale = (
            float(max(abs(graph_step.target_result), Fraction(1)))
            if graph_step.target_result is not None
            else None
        )
        if program_step.target_scale != expected_scale or (
            program_step.target_scale is not None
            and not math.isfinite(program_step.target_scale)
        ):
            add_issue(
                "target_scale_mismatch",
                "target_scale is not max(abs(target_result), 1.0).",
                step_index=step_index,
            )

        if not graph_step.valid or graph_step.expression_tree is None:
            if program_step.actions:
                add_issue(
                    "invalid_step_has_actions",
                    "An invalid graph step produced postfix actions.",
                    step_index=step_index,
                )
            continue

        expected_outputs = _graph_postorder_signatures(
            graph_step.expression_tree,
            problem_number_indices=problem_number_indices,
        )
        actual_outputs = _program_action_signatures(program_step)
        if len(actual_outputs) != len(expected_outputs):
            add_issue(
                "action_count_mismatch",
                (
                    f"Expected {len(expected_outputs)} post-order actions, "
                    f"received {len(actual_outputs)}."
                ),
                step_index=step_index,
            )
        if actual_outputs != expected_outputs:
            add_issue(
                "action_sequence_mismatch",
                "Postfix actions do not preserve every ordered graph subtree.",
                step_index=step_index,
            )

        if _program_dependencies(program_step) != graph_step.dependencies:
            add_issue(
                "dependency_mismatch",
                "PreviousResult references differ from graph dependencies.",
                step_index=step_index,
            )

        expected_masks = tuple(
            tuple(
                not isinstance(operand, UnresolvedReference)
                for operand in action.operands
            )
            for action in program_step.actions
        )
        if program_step.operand_mask != expected_masks:
            add_issue(
                "operand_mask_mismatch",
                "An operand mask does not isolate exactly Unresolved references.",
                step_index=step_index,
            )
        if program_step.masked_operand_count != graph_step.unresolved_operand_count:
            add_issue(
                "unresolved_count_mismatch",
                "Masked operands differ from graph Unresolved occurrences.",
                step_index=step_index,
            )

    return PostfixStructuralValidation(
        checked_step_count=len(paired_steps),
        issues=tuple(issues),
    )
