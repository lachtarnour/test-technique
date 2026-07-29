"""Tests for conservative non-semantic graph construction and execution."""

from __future__ import annotations

import json
from fractions import Fraction

from src.data.graph.audit import audit_graph_rows
from src.data.graph.builder import (
    build_calculation_graph,
    extract_problem_numbers,
    walk_expression_tree,
)
from src.data.graph.execution import evaluate_calculation_graph
from src.data.graph.parser import parse_math_steps
from src.data.graph.schemas import (
    LiteralNode,
    ProblemNumberNode,
    ReferenceNode,
    UnresolvedNode,
)


def _nodes(graph, step_index: int):
    tree = graph.steps[step_index].expression_tree
    assert tree is not None
    return tuple(walk_expression_tree(tree))


def test_problem_numbers_are_extracted_exactly_with_spans() -> None:
    question = (
        "A box has 1,000 items; 3/5 are blue, 2.5 are reserved, "
        "twenty-five are packed twice, and the adjustment is -2."
    )

    numbers = extract_problem_numbers(question)

    assert [number.value for number in numbers] == [
        Fraction(1000),
        Fraction(3, 5),
        Fraction(5, 2),
        Fraction(25),
        Fraction(2),
        Fraction(-2),
    ]
    assert [number.source_span.extract(question) for number in numbers] == [
        "1,000",
        "3/5",
        "2.5",
        "twenty-five",
        "twice",
        "-2",
    ]


def test_adjacent_word_numbers_are_not_merged() -> None:
    cases = {
        "six one day": [("six", Fraction(6)), ("one", Fraction(1))],
        "five and nine pounds": [
            ("five", Fraction(5)),
            ("nine", Fraction(9)),
        ],
        "twenty-five one ounce": [
            ("twenty-five", Fraction(25)),
            ("one", Fraction(1)),
        ],
    }

    for question, expected in cases.items():
        numbers = extract_problem_numbers(question)
        assert [(number.source_text, number.value) for number in numbers] == expected


def test_word_fractions_and_mixed_numbers_are_extracted_exactly() -> None:
    cases = {
        "one-third remains": [("one-third", Fraction(1, 3))],
        "three-fifths remain": [("three-fifths", Fraction(3, 5))],
        "a quarter remains": [("a quarter", Fraction(1, 4))],
        "three and one-fourth pounds": [("three and one-fourth", Fraction(13, 4))],
        "2 and a half pounds": [("2 and a half", Fraction(5, 2))],
        "2 and three-fourths pounds": [("2 and three-fourths", Fraction(11, 4))],
        "half a dozen eggs": [("half a dozen", Fraction(6))],
        "time-and-a-half pay": [("time-and-a-half", Fraction(3, 2))],
    }

    for question, expected in cases.items():
        numbers = extract_problem_numbers(question)
        assert [(number.source_text, number.value) for number in numbers] == expected


def test_contextual_quantities_are_supported_without_ordinal_false_positives() -> None:
    question = "A couple of socks are checked once per week."

    numbers = extract_problem_numbers(question)

    assert [(number.source_text, number.value) for number in numbers] == [
        ("A couple of", Fraction(2)),
        ("once", Fraction(1)),
    ]
    assert (
        extract_problem_numbers(
            "The third day leaves remaining ones; totals are shown in millions. P.T.O."
        )
        == ()
    )
    assert (
        extract_problem_numbers(
            "A pair of shoes is used in the final quarter; "
            "the coins are quarters split into halves and thirds."
        )
        == ()
    )


def test_quarter_counts_are_not_mistaken_for_fraction_values() -> None:
    question = (
        "There are four quarters in a year and six quarters in the purse, "
        "but three quarters of the cake remain."
    )

    numbers = extract_problem_numbers(question)

    assert [(number.source_text, number.value) for number in numbers] == [
        ("four", Fraction(4)),
        ("six", Fraction(6)),
        ("three quarters", Fraction(3, 4)),
    ]


def test_numeric_fraction_is_one_problem_operand_not_two_integer_operands() -> None:
    question = "A tank is 3/8 full and has capacity 32 liters."
    steps = parse_math_steps(
        "An unrelated earlier result is <<16/2=8>>. "
        "The amount is <<3/8*32=12>> liters.\n"
        "#### 12"
    )

    graph = build_calculation_graph(question, steps)
    evaluation = evaluate_calculation_graph(graph)

    fraction_nodes = [
        node
        for node in _nodes(graph, 1)
        if isinstance(node, ProblemNumberNode) and node.value == Fraction(3, 8)
    ]
    references = [node for node in _nodes(graph, 1) if isinstance(node, ReferenceNode)]
    assert len(fraction_nodes) == 1
    assert references == []
    assert graph.steps[1].dependencies == ()
    assert evaluation.all_steps_match_targets is True


def test_equal_fraction_value_does_not_hide_a_real_previous_step_reference() -> None:
    question = (
        "The largest rate is 3. The medium rate is one-half of it, "
        "and the smallest rate is one-third of the medium rate."
    )
    steps = parse_math_steps(
        "The medium rate is <<3/2=1.5>>. The smallest rate is <<1.5/3=.5>>.\n#### .5"
    )

    graph = build_calculation_graph(question, steps)

    references = [node for node in _nodes(graph, 1) if isinstance(node, ReferenceNode)]
    assert references == [ReferenceNode(step_index=0, value=Fraction(3, 2))]
    assert graph.steps[1].dependencies == (0,)


def test_equal_division_results_do_not_create_a_false_reference() -> None:
    graph = build_calculation_graph(
        "The available values are 1, 10, 5, and 50.",
        parse_math_steps(
            "The first ratio is <<1/10=.1>>. The second ratio is <<5/50=.1>>.\n#### .1"
        ),
    )

    assert graph.steps[1].dependencies == ()
    assert not any(isinstance(node, ReferenceNode) for node in _nodes(graph, 1))


def test_identity_fraction_step_uses_the_fraction_from_the_problem() -> None:
    graph = build_calculation_graph(
        "A ribbon is 1/3 meter long.",
        parse_math_steps("Its length is <<1/3=1/3>> meter.\n#### 1/3"),
    )
    evaluation = evaluate_calculation_graph(graph)

    assert isinstance(graph.steps[0].expression_tree, ProblemNumberNode)
    assert graph.steps[0].expression_tree.value == Fraction(1, 3)
    assert evaluation.all_steps_match_targets is True


def test_unique_previous_result_becomes_a_reference() -> None:
    question = "There are 6 objects and 2 are removed."
    steps = parse_math_steps("First <<6-2=4>> remain. Then <<4-1=3>> remain.\n#### 3")

    graph = build_calculation_graph(question, steps)

    references = [node for node in _nodes(graph, 1) if isinstance(node, ReferenceNode)]
    assert references == [ReferenceNode(step_index=0, value=Fraction(4))]
    assert graph.steps[1].dependencies == (0,)
    assert graph.provenance_complete is True


def test_one_step_can_merge_multiple_previous_branches() -> None:
    steps = parse_math_steps(
        "First branch <<5-4=1>>. "
        "Second branch <<6-4=2>>. "
        "Merge both branches <<1+2=3>>.\n"
        "#### 3"
    )

    graph = build_calculation_graph(
        "Compute independent branches and merge their results.",
        steps,
    )
    evaluation = evaluate_calculation_graph(graph)

    assert [step.dependencies for step in graph.steps] == [
        (),
        (),
        (0, 1),
    ]
    references = [node for node in _nodes(graph, 2) if isinstance(node, ReferenceNode)]
    assert references == [
        ReferenceNode(step_index=0, value=Fraction(1)),
        ReferenceNode(step_index=1, value=Fraction(2)),
    ]
    assert evaluation.all_steps_match_targets is True
    assert evaluation.steps[2].result == Fraction(3)


def test_problem_and_step_collision_is_unresolved() -> None:
    question = "There are 4 red objects among 6 objects, and 2 are removed."
    steps = parse_math_steps(
        "First <<6-2=4>> remain. Then <<4+1=5>> are counted.\n#### 5"
    )

    graph = build_calculation_graph(question, steps)

    unresolved = [node for node in _nodes(graph, 1) if isinstance(node, UnresolvedNode)]
    assert len(unresolved) == 1
    assert unresolved[0].value == Fraction(4)
    assert {type(candidate) for candidate in unresolved[0].candidates} == {
        ProblemNumberNode,
        ReferenceNode,
    }
    assert graph.steps[1].dependencies == ()
    assert graph.steps[1].unresolved_operand_count == 1
    assert graph.provenance_complete is False


def test_word_number_and_step_collision_is_also_unresolved() -> None:
    question = "There are four red objects among 6 objects, and 2 are removed."
    steps = parse_math_steps(
        "First <<6-2=4>> remain. Then <<4+1=5>> are counted.\n#### 5"
    )

    graph = build_calculation_graph(question, steps)

    unresolved = [node for node in _nodes(graph, 1) if isinstance(node, UnresolvedNode)]
    assert len(unresolved) == 1
    assert unresolved[0].value == Fraction(4)
    assert {
        candidate.source_text
        for candidate in unresolved[0].candidates
        if isinstance(candidate, ProblemNumberNode)
    } == {"four"}


def test_multiple_previous_results_are_unresolved() -> None:
    question = "The available starting values are 6 and 8."
    steps = parse_math_steps(
        "First <<6-2=4>>, then <<8/2=4>>, finally <<4+1=5>>.\n#### 5"
    )

    graph = build_calculation_graph(question, steps)

    unresolved = [node for node in _nodes(graph, 2) if isinstance(node, UnresolvedNode)]
    assert len(unresolved) == 1
    assert [
        candidate.step_index
        for candidate in unresolved[0].candidates
        if isinstance(candidate, ReferenceNode)
    ] == [0, 1]
    assert graph.steps[2].dependencies == ()


def test_numbers_without_candidates_remain_literals() -> None:
    graph = build_calculation_graph(
        "The problem contains 6.",
        parse_math_steps("Calculate <<6+7=13>>.\n#### 13"),
    )

    nodes = _nodes(graph, 0)

    assert any(
        isinstance(node, ProblemNumberNode) and node.value == 6 for node in nodes
    )
    assert any(isinstance(node, LiteralNode) and node.value == 7 for node in nodes)


def test_frankie_graph_executes_all_reference_dependencies_exactly() -> None:
    question = (
        "Frankie has six more snakes than cats and one less parrot than cats. "
        "Six pets have four legs. He has 2 dogs."
    )
    answer = (
        "<<6-2=4>>4 cats.\n"
        "<<4-1=3>>3 parrots.\n"
        "<<4+6=10>>10 snakes.\n"
        "<<2+4+3+10=19>>19 pets.\n"
        "#### 19"
    )

    graph = build_calculation_graph(question, parse_math_steps(answer))
    evaluation = evaluate_calculation_graph(graph)

    assert [step.dependencies for step in graph.steps] == [
        (),
        (),
        (),
        (1, 2),
    ]
    assert graph.provenance_complete is False
    assert graph.unresolved_operand_count == 5
    assert evaluation.all_steps_executable is True
    assert evaluation.all_steps_match_targets is True
    assert [step.result for step in evaluation.steps] == [
        Fraction(4),
        Fraction(3),
        Fraction(10),
        Fraction(19),
    ]
    json.dumps(graph.to_dict())
    json.dumps(evaluation.to_dict())


def test_unresolved_graph_still_executes_without_supervising_a_false_edge() -> None:
    question = "There are 4 red objects among 6 objects, and 2 are removed."
    steps = parse_math_steps(
        "First <<6-2=4>> remain. Then <<4+1=5>> are counted.\n#### 5"
    )

    graph = build_calculation_graph(question, steps)
    evaluation = evaluate_calculation_graph(graph)

    assert graph.provenance_complete is False
    assert evaluation.all_steps_executable is True
    assert evaluation.all_steps_match_targets is True


def test_graph_audit_reports_every_example() -> None:
    rows = [
        {
            "question": "There are 6 objects and 2 are removed.",
            "answer": "<<6-2=4>>4 remain.\n#### 4",
        },
        {
            "question": "There are 4 red objects among 6; 2 are removed.",
            "answer": "<<6-2=4>> remain, then <<4+1=5>>.\n#### 5",
        },
    ]

    report = audit_graph_rows(rows, workers=1)

    assert report["summary"]["examples"] == 2
    assert report["summary"]["valid_math_step_rate"] == 1.0
    assert report["summary"]["graph_execution_rate"] == 1.0
    assert report["summary"]["unresolved_operands"] == 1
    assert [example["sample_index"] for example in report["examples"]] == [0, 1]
    json.dumps(report)
