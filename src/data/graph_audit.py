"""Parallel dataset audit for V1 calculation-graph construction."""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.data.formatting import remove_trivial_identity_annotations
from src.data.graph_builder import (
    build_calculation_graph,
    evaluate_calculation_graph,
    walk_expression_tree,
)
from src.data.math_parser import parse_math_steps
from src.data.schemas import (
    LiteralNode,
    NumberNode,
    OperationNode,
    ProblemNumberNode,
    ReferenceNode,
    UnresolvedNode,
)


def _example_id(question: str, answer: str) -> str:
    payload = f"{question}\0{answer}".encode()
    return hashlib.sha256(payload).hexdigest()


def _unresolved_reason(node: UnresolvedNode) -> str:
    problem_candidates = sum(
        isinstance(candidate, ProblemNumberNode) for candidate in node.candidates
    )
    reference_candidates = sum(
        isinstance(candidate, ReferenceNode) for candidate in node.candidates
    )
    if problem_candidates and reference_candidates:
        return "problem_and_previous_step"
    if problem_candidates > 1:
        return "multiple_problem_occurrences"
    if reference_candidates > 1:
        return "multiple_previous_steps"
    return "multiple_candidates"


def _audit_example(payload: tuple[int, dict[str, str]]) -> dict[str, Any]:
    sample_index, row = payload
    question = row["question"]
    original_answer = row["answer"]
    answer = remove_trivial_identity_annotations(original_answer)
    math_steps = parse_math_steps(answer)
    graph = build_calculation_graph(question, math_steps)
    evaluation = evaluate_calculation_graph(graph)

    node_counts: Counter[str] = Counter()
    unresolved_reasons: Counter[str] = Counter()
    for step in graph.steps:
        if step.expression_tree is None:
            continue
        for node in walk_expression_tree(step.expression_tree):
            if isinstance(node, OperationNode):
                node_counts["operation"] += 1
            elif isinstance(node, ProblemNumberNode):
                node_counts["problem_number"] += 1
            elif isinstance(node, ReferenceNode):
                node_counts["reference"] += 1
            elif isinstance(node, LiteralNode):
                node_counts["literal"] += 1
            elif isinstance(node, UnresolvedNode):
                node_counts["unresolved"] += 1
                unresolved_reasons[_unresolved_reason(node)] += 1
            elif isinstance(node, NumberNode):
                node_counts["unresolved_syntax_number"] += 1

    valid_math_steps = sum(step.valid for step in math_steps)
    final_step_identified = any(step.is_final for step in graph.steps)
    graph_execution_success = (
        evaluation.all_steps_executable
        and evaluation.all_steps_match_targets
    )
    return {
        "sample_index": sample_index,
        "worker_pid": os.getpid(),
        "example_id": _example_id(question, original_answer),
        "question": question,
        "answer": original_answer,
        "preprocessed_answer": answer,
        "metrics": {
            "math_step_count": len(math_steps),
            "valid_math_step_count": valid_math_steps,
            "all_math_steps_valid": (
                bool(math_steps) and valid_math_steps == len(math_steps)
            ),
            "reference_edge_count": sum(
                len(step.dependencies) for step in graph.steps
            ),
            "reference_operand_count": node_counts["reference"],
            "unresolved_operand_count": graph.unresolved_operand_count,
            "provenance_complete": graph.provenance_complete,
            "final_step_identified": final_step_identified,
            "graph_execution_success": graph_execution_success,
            "fully_resolved_and_executable": (
                graph.provenance_complete and graph_execution_success
            ),
            "node_counts": dict(sorted(node_counts.items())),
            "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        },
        "math_steps": [step.to_dict() for step in math_steps],
        "graph": graph.to_dict(),
        "evaluation": evaluation.to_dict(),
    }


def _aggregate_example_reports(
    reports: list[dict[str, Any]],
    *,
    workers: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    total_steps = sum(
        report["metrics"]["math_step_count"] for report in reports
    )
    valid_steps = sum(
        report["metrics"]["valid_math_step_count"] for report in reports
    )
    node_counts: Counter[str] = Counter()
    unresolved_reasons: Counter[str] = Counter()
    for report in reports:
        node_counts.update(report["metrics"]["node_counts"])
        unresolved_reasons.update(report["metrics"]["unresolved_reasons"])

    example_count = len(reports)
    all_steps_valid = sum(
        report["metrics"]["all_math_steps_valid"] for report in reports
    )
    executable = sum(
        report["metrics"]["graph_execution_success"] for report in reports
    )
    provenance_complete = sum(
        report["metrics"]["provenance_complete"] for report in reports
    )
    fully_resolved = sum(
        report["metrics"]["fully_resolved_and_executable"]
        for report in reports
    )
    final_identified = sum(
        report["metrics"]["final_step_identified"] for report in reports
    )
    return {
        "examples": example_count,
        "workers": workers,
        "worker_process_count": len(
            {report["worker_pid"] for report in reports}
        ),
        "worker_process_ids": sorted(
            {report["worker_pid"] for report in reports}
        ),
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": (
            example_count / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "math_steps": total_steps,
        "valid_math_steps": valid_steps,
        "valid_math_step_rate": valid_steps / total_steps if total_steps else 0.0,
        "examples_with_all_math_steps_valid": all_steps_valid,
        "all_math_steps_valid_rate": all_steps_valid / example_count,
        "examples_with_executable_graph": executable,
        "graph_execution_rate": executable / example_count,
        "examples_with_complete_provenance": provenance_complete,
        "provenance_complete_rate": provenance_complete / example_count,
        "examples_fully_resolved_and_executable": fully_resolved,
        "fully_resolved_and_executable_rate": fully_resolved / example_count,
        "examples_with_identified_final_step": final_identified,
        "final_step_coverage": final_identified / example_count,
        "reference_edges": sum(
            report["metrics"]["reference_edge_count"] for report in reports
        ),
        "reference_operands": node_counts["reference"],
        "unresolved_operands": node_counts["unresolved"],
        "node_counts": dict(sorted(node_counts.items())),
        "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
    }


def audit_graph_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Build and execute every row graph, optionally in parallel."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a strictly positive integer.")
    if not rows:
        raise ValueError("At least one row is required for a graph audit.")

    payloads = [
        (
            index,
            {
                "question": row["question"],
                "answer": row["answer"],
            },
        )
        for index, row in enumerate(rows)
    ]
    started_at = time.perf_counter()
    if workers == 1:
        reports = [_audit_example(payload) for payload in payloads]
    else:
        chunksize = max(1, len(payloads) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            reports = list(
                executor.map(
                    _audit_example,
                    payloads,
                    chunksize=chunksize,
                )
            )
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "summary": _aggregate_example_reports(
            reports,
            workers=workers,
            elapsed_seconds=elapsed_seconds,
        ),
        "examples": reports,
    }
