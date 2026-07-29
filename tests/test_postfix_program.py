"""Tests for postfix compilation, local masks, and exact round-trip execution."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from src.data.graph.audit import audit_graph_rows
from src.data.graph.builder import build_calculation_graph
from src.data.graph.execution import evaluate_calculation_graph
from src.data.graph.parser import parse_math_steps
from src.data.graph.postfix import (
    compile_calculation_graph,
    execute_postfix_program,
    validate_postfix_structure,
    verify_postfix_results,
)
from src.data.graph.postfix_schemas import (
    LiteralReference,
    LocalResultReference,
    PostfixAction,
    PostfixOperator,
    PreviousResultReference,
    ProblemNumberReference,
    UnresolvedReference,
)
from src.data.graph.schemas import (
    ArithmeticOperator,
    CalculationGraph,
    GraphExpressionNode,
    GraphStep,
    LiteralNode,
    OperationNode,
)


def _compile(question: str, answer: str):
    graph = build_calculation_graph(question, parse_math_steps(answer))
    program = compile_calculation_graph(graph)
    return graph, program


def test_nested_tree_compiles_in_left_to_right_postfix_order() -> None:
    graph, program = _compile(
        "Compute the requested quantity.",
        "The result is <<1+1/2=3/2>>. \n#### 3/2",
    )

    actions = program.steps[0].actions

    assert actions == (
        PostfixAction(
            index=0,
            operator=PostfixOperator.DIVIDE,
            operands=(
                LiteralReference(Fraction(1)),
                LiteralReference(Fraction(2)),
            ),
        ),
        PostfixAction(
            index=1,
            operator=PostfixOperator.ADD,
            operands=(
                LiteralReference(Fraction(1)),
                LocalResultReference(action_index=0),
            ),
        ),
    )
    assert program.steps[0].target_scale == 1.5
    assert execute_postfix_program(program).steps[0].result == Fraction(3, 2)
    assert evaluate_calculation_graph(graph).steps[0].result == Fraction(3, 2)


def test_leaf_expression_gets_copy_and_problem_occurrence_pointer() -> None:
    graph, program = _compile(
        "A ribbon is 1/3 meter long.",
        "Its length is <<1/3=1/3>> meter.\n#### 1/3",
    )

    step = program.steps[0]

    assert len(graph.problem_numbers) == 1
    assert step.actions == (
        PostfixAction(
            index=0,
            operator=PostfixOperator.COPY,
            operands=(ProblemNumberReference(problem_number_index=0),),
        ),
    )
    assert step.target_scale == 1.0
    assert execute_postfix_program(program).steps[0].result == Fraction(1, 3)


def test_previous_and_repeated_results_keep_order_and_occurrences() -> None:
    _, program = _compile(
        "There are 6 objects and 2 are removed.",
        "First <<6-2=4>>, then subtract it from itself <<4-4=0>>.\n#### 0",
    )

    second_action = program.steps[1].actions[0]

    assert second_action.operator is PostfixOperator.SUBTRACT
    assert second_action.operands == (
        PreviousResultReference(step_index=0),
        PreviousResultReference(step_index=0),
    )
    assert second_action.operand_mask == (True, True)
    assert execute_postfix_program(program).steps[1].result == 0


def test_unresolved_provenance_masks_only_its_operand_position() -> None:
    _, program = _compile(
        "There are 4 red objects among 6 objects, and 2 are removed.",
        "First <<6-2=4>> remain. Then <<4+1=5>> are counted.\n#### 5",
    )

    action = program.steps[1].actions[0]
    ambiguous_operand = action.operands[0]

    assert isinstance(ambiguous_operand, UnresolvedReference)
    assert ambiguous_operand.candidates == (
        ProblemNumberReference(problem_number_index=0),
        PreviousResultReference(step_index=0),
    )
    assert action.operand_mask == (False, True)
    assert program.steps[1].operand_mask == ((False, True),)
    assert program.masked_operand_count == 1
    assert execute_postfix_program(program).steps[1].result == Fraction(5)


def test_floor_division_keeps_execution_operator_but_merges_supervision() -> None:
    graph, program = _compile(
        "Compute the requested quantity.",
        "The result is <<7//2=3>>.\n#### 3",
    )

    action = program.steps[0].actions[0]
    serialized = action.to_dict()

    assert action.operator is PostfixOperator.FLOOR_DIVIDE
    assert serialized["execution_operator"] == "floor_div"
    assert serialized["supervision_operator"] == "div"
    assert execute_postfix_program(program).steps[0].result == Fraction(3)
    assert evaluate_calculation_graph(graph).steps[0].result == Fraction(3)


def test_target_scale_is_derived_from_the_exact_step_target() -> None:
    _, large_program = _compile(
        "Compute the requested quantity.",
        "The result is <<2000*2000=4000000>>.\n#### 4000000",
    )
    _, small_program = _compile(
        "Compute the requested quantity.",
        "The result is <<1/4=0.25>>.\n#### 0.25",
    )

    assert large_program.steps[0].target_scale == 4_000_000.0
    assert small_program.steps[0].target_scale == 1.0
    assert large_program.steps[0].to_dict()["target_scale"] == 4_000_000.0


def test_program_reproduces_every_graph_result_exactly() -> None:
    graph, program = _compile(
        "There are 10 items and 3 are removed.",
        "First <<10-3=7>>. Then <<7/2=3.5>>.\n#### 3.5",
    )
    graph_evaluation = evaluate_calculation_graph(graph)
    program_evaluation = execute_postfix_program(program)
    verification = verify_postfix_results(
        program_evaluation,
        graph_results=tuple(step.result for step in graph_evaluation.steps),
    )
    structure_validation = validate_postfix_structure(graph, program)

    assert verification.all_steps_match_graph is True
    assert structure_validation.is_valid is True
    assert structure_validation.checked_step_count == 2
    assert structure_validation.issues == ()
    assert verification.executable_step_count == 2
    assert verification.matching_executable_step_count == 2
    json.dumps(program.to_dict())
    json.dumps(program_evaluation.to_dict())
    json.dumps(verification.to_dict())


def test_schema_rejects_forward_local_result_pointer() -> None:
    with pytest.raises(
        ValueError,
        match="local result must point to an earlier action",
    ):
        PostfixAction(
            index=0,
            operator=PostfixOperator.ADD,
            operands=(
                LocalResultReference(action_index=0),
                LiteralReference(Fraction(1)),
            ),
        )


def test_structural_validation_detects_operand_reordering() -> None:
    graph, program = _compile(
        "There are 10 items and 3 are removed.",
        "The result is <<10-3=7>>.\n#### 7",
    )
    original_step = program.steps[0]
    original_action = original_step.actions[0]
    reordered_action = replace(
        original_action,
        operands=tuple(reversed(original_action.operands)),
    )
    corrupted_program = replace(
        program,
        steps=(replace(original_step, actions=(reordered_action,)),),
    )

    validation = validate_postfix_structure(graph, corrupted_program)

    assert validation.is_valid is False
    assert validation.issue_counts == {"action_sequence_mismatch": 1}


def test_structural_validation_detects_dead_or_extra_action() -> None:
    graph, program = _compile(
        "There are 10 items and 3 are removed.",
        "The result is <<10-3=7>>.\n#### 7",
    )
    original_step = program.steps[0]
    extra_action = PostfixAction(
        index=1,
        operator=PostfixOperator.COPY,
        operands=(LiteralReference(Fraction(7)),),
    )
    corrupted_program = replace(
        program,
        steps=(
            replace(
                original_step,
                actions=(*original_step.actions, extra_action),
            ),
        ),
    )

    validation = validate_postfix_structure(graph, corrupted_program)

    assert validation.is_valid is False
    assert validation.issue_counts == {
        "action_count_mismatch": 1,
        "action_sequence_mismatch": 1,
    }


def _random_expression(
    generator: random.Random,
    *,
    depth: int,
) -> tuple[GraphExpressionNode, Fraction]:
    if depth == 0 or generator.random() < 0.3:
        value = Fraction(generator.randint(-20, 20), generator.randint(1, 7))
        return LiteralNode(value), value

    operator = generator.choice(tuple(ArithmeticOperator))
    if operator in {ArithmeticOperator.POSITIVE, ArithmeticOperator.NEGATE}:
        operand, operand_value = _random_expression(
            generator,
            depth=depth - 1,
        )
        value = (
            operand_value if operator is ArithmeticOperator.POSITIVE else -operand_value
        )
        return OperationNode(operator=operator, operands=(operand,)), value

    left, left_value = _random_expression(generator, depth=depth - 1)
    right, right_value = _random_expression(generator, depth=depth - 1)
    if (
        operator in {ArithmeticOperator.DIVIDE, ArithmeticOperator.FLOOR_DIVIDE}
        and right_value == 0
    ):
        right = LiteralNode(Fraction(1))
        right_value = Fraction(1)

    if operator is ArithmeticOperator.ADD:
        value = left_value + right_value
    elif operator is ArithmeticOperator.SUBTRACT:
        value = left_value - right_value
    elif operator is ArithmeticOperator.MULTIPLY:
        value = left_value * right_value
    elif operator is ArithmeticOperator.DIVIDE:
        value = left_value / right_value
    else:
        value = Fraction(left_value // right_value)
    return OperationNode(operator=operator, operands=(left, right)), value


def test_seeded_random_trees_compile_and_execute_exactly() -> None:
    generator = random.Random(20260728)
    for sample_index in range(1_000):
        expression_tree, target_result = _random_expression(
            generator,
            depth=generator.randint(0, 5),
        )
        graph = CalculationGraph(
            problem_numbers=(),
            steps=(
                GraphStep(
                    index=0,
                    expression=f"synthetic_{sample_index}",
                    target_result=target_result,
                    expression_tree=expression_tree,
                    dependencies=(),
                    unresolved_operand_count=0,
                    valid=True,
                    error=None,
                    is_final=True,
                ),
            ),
        )

        program = compile_calculation_graph(graph)
        validation = validate_postfix_structure(graph, program)
        evaluation = execute_postfix_program(program)

        assert validation.is_valid is True
        assert evaluation.steps[0].result == target_result


def test_audit_reports_postfix_execution_and_local_masks() -> None:
    rows = [
        {
            "question": "There are 6 objects and 2 are removed.",
            "answer": "<<6-2=4>> remain.\n#### 4",
        },
        {
            "question": ("There are 4 red objects among 6 objects, and 2 are removed."),
            "answer": "<<6-2=4>> remain, then <<4+1=5>>.\n#### 5",
        },
    ]

    report = audit_graph_rows(rows, workers=1)
    summary = report["summary"]

    assert summary["program_execution_rate"] == 1.0
    assert summary["program_graph_equivalence_rate"] == 1.0
    assert summary["program_structure_validity_rate"] == 1.0
    assert summary["program_structure_issues"] == 0
    assert summary["program_steps"] == summary["program_steps_matching_graph"]
    assert summary["program_result_mismatches"] == 0
    assert summary["masked_operands"] == 1
    assert summary["operand_reference_counts"]["unresolved"] == 1
    assert report["examples"][1]["program"]["steps"][1]["operand_mask"] == [
        [False, True]
    ]
    json.dumps(report)
