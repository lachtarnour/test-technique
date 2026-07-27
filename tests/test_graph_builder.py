"""Tests for conservative non-semantic graph construction and execution."""

from __future__ import annotations

import json
from fractions import Fraction

from src.data.graph_audit import audit_graph_rows
from src.data.graph_builder import (
    build_calculation_graph,
    evaluate_calculation_graph,
    extract_problem_numbers,
    walk_expression_tree,
)
from src.data.math_parser import parse_math_steps
from src.data.schemas import (
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
        "and the adjustment is -2."
    )

    numbers = extract_problem_numbers(question)

    assert [number.value for number in numbers] == [
        Fraction(1000),
        Fraction(3, 5),
        Fraction(5, 2),
        Fraction(-2),
    ]
    assert [number.source_span.extract(question) for number in numbers] == [
        "1,000",
        "3/5",
        "2.5",
        "-2",
    ]


def test_unique_previous_result_becomes_a_reference() -> None:
    question = "There are 6 objects and 2 are removed."
    steps = parse_math_steps(
        "First <<6-2=4>> remain. Then <<4-1=3>> remain.\n#### 3"
    )

    graph = build_calculation_graph(question, steps)

    references = [
        node for node in _nodes(graph, 1) if isinstance(node, ReferenceNode)
    ]
    assert references == [ReferenceNode(step_index=0, value=Fraction(4))]
    assert graph.steps[1].dependencies == (0,)
    assert graph.provenance_complete is True


def test_problem_and_step_collision_is_unresolved() -> None:
    question = "There are 4 red objects among 6 objects, and 2 are removed."
    steps = parse_math_steps(
        "First <<6-2=4>> remain. Then <<4+1=5>> are counted.\n#### 5"
    )

    graph = build_calculation_graph(question, steps)

    unresolved = [
        node for node in _nodes(graph, 1) if isinstance(node, UnresolvedNode)
    ]
    assert len(unresolved) == 1
    assert unresolved[0].value == Fraction(4)
    assert {type(candidate) for candidate in unresolved[0].candidates} == {
        ProblemNumberNode,
        ReferenceNode,
    }
    assert graph.steps[1].dependencies == ()
    assert graph.steps[1].unresolved_operand_count == 1
    assert graph.provenance_complete is False


def test_multiple_previous_results_are_unresolved() -> None:
    question = "The available starting values are 6 and 8."
    steps = parse_math_steps(
        "First <<6-2=4>>, then <<8/2=4>>, finally <<4+1=5>>.\n#### 5"
    )

    graph = build_calculation_graph(question, steps)

    unresolved = [
        node for node in _nodes(graph, 2) if isinstance(node, UnresolvedNode)
    ]
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
        (0,),
        (0,),
        (0, 1, 2),
    ]
    assert graph.provenance_complete is True
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
