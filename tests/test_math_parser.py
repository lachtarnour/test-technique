"""Tests for the exact, non-semantic V1 math parser."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from src.data.graph.parser import (
    evaluate_expression_tree,
    parse_expression_tree,
    parse_math_steps,
)
from src.data.graph.schemas import (
    ArithmeticOperator,
    NumberNode,
    OperationNode,
)


def test_parser_builds_the_four_frankie_steps() -> None:
    answer = (
        "He has 6 - 2 = <<6-2=4>>4 cats.\n"
        "He has 4 - 1 = <<4-1=3>>3 parrots.\n"
        "He has 4 + 6 = <<4+6=10>>10 snakes.\n"
        "He has <<2+4+3+10=19>>19 pets.\n"
        "#### 19"
    )

    steps = parse_math_steps(answer)

    assert len(steps) == 4
    assert [step.index for step in steps] == [0, 1, 2, 3]
    assert [step.expression for step in steps] == [
        "6-2",
        "4-1",
        "4+6",
        "2+4+3+10",
    ]
    assert [step.target_result for step in steps] == [
        Fraction(4),
        Fraction(3),
        Fraction(10),
        Fraction(19),
    ]
    assert [step.operator for step in steps] == [
        ArithmeticOperator.SUBTRACT,
        ArithmeticOperator.SUBTRACT,
        ArithmeticOperator.ADD,
        ArithmeticOperator.ADD,
    ]
    assert all(step.valid for step in steps)
    assert [step.is_final for step in steps] == [False, False, False, True]


def test_parser_preserves_exact_targets_for_fractional_claims() -> None:
    steps = parse_math_steps("Exact <<1/3=1/3>> and approximate <<1/3=.3>>.\n#### .3")

    assert len(steps) == 2
    assert steps[0].claimed_result == Fraction(1, 3)
    assert steps[0].target_result == Fraction(1, 3)
    assert steps[1].claimed_result == Fraction(3, 10)
    assert steps[1].target_result == Fraction(1, 3)
    assert all(step.valid for step in steps)
    assert steps[1].is_final is True


def test_nested_expression_tree_is_exact_and_non_semantic() -> None:
    tree = parse_expression_tree("1+1/2")

    assert tree == OperationNode(
        operator=ArithmeticOperator.ADD,
        operands=(
            NumberNode(Fraction(1)),
            OperationNode(
                operator=ArithmeticOperator.DIVIDE,
                operands=(
                    NumberNode(Fraction(1)),
                    NumberNode(Fraction(2)),
                ),
            ),
        ),
    )
    assert evaluate_expression_tree(tree) == Fraction(3, 2)


def test_floor_division_executes_by_floor_but_uses_the_division_label() -> None:
    tree = parse_expression_tree("7//2")
    step = parse_math_steps("Calculate <<7//2=3>>.\n#### 3")[0]

    assert isinstance(tree, OperationNode)
    assert tree.operator is ArithmeticOperator.FLOOR_DIVIDE
    assert evaluate_expression_tree(tree) == Fraction(3)
    assert step.valid is True
    assert step.operator is ArithmeticOperator.DIVIDE
    assert step.to_dict()["operator"] == "div"
    assert step.to_dict()["expression_tree"]["operator"] == "div"


def test_parser_records_exact_source_spans() -> None:
    answer = "Before << 1 + 1/2 = 3/2 >> after.\n#### 3/2"

    step = parse_math_steps(answer)[0]

    assert step.annotation_span.extract(answer) == "<< 1 + 1/2 = 3/2 >>"
    assert step.expression_span is not None
    assert step.expression_span.extract(answer) == "1 + 1/2"
    assert step.claimed_result_span is not None
    assert step.claimed_result_span.extract(answer) == "3/2"


def test_parser_does_not_invent_an_unannotated_final_step() -> None:
    steps = parse_math_steps(
        "The original total is <<15+60=75>>75 apples.\n"
        "Each child has 15, so 75/15 = 5 children.\n"
        "#### 5"
    )

    assert len(steps) == 1
    assert steps[0].valid is True
    assert steps[0].is_final is False


@pytest.mark.parametrize(
    ("annotation", "expected_error"),
    [
        ("<<4=4>>", "invalid_expression"),
        ("<<1/3=2/3>>", "incorrect_result"),
        ("<<1/0=0>>", "execution_error"),
        ("<<2+2>>", "missing_equals"),
        ("<<__import__('os')=0>>", "invalid_expression"),
        ("<<2+2=4", "unclosed_annotation"),
    ],
)
def test_parser_retains_invalid_annotations(
    annotation: str,
    expected_error: str,
) -> None:
    step = parse_math_steps(annotation)[0]

    assert step.valid is False
    assert step.error == expected_error
    assert step.is_final is False


def test_math_step_is_json_serializable() -> None:
    step = parse_math_steps("Calculate <<1+1/2=3/2>>.\n#### 3/2")[0]

    serialized = json.loads(json.dumps(step.to_dict()))

    assert serialized["target_result"] == {"numerator": 3, "denominator": 2}
    assert serialized["operator"] == "add"
    assert serialized["expression_tree"]["kind"] == "operation"
