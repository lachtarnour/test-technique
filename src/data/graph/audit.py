"""Parallel dataset audit for V1 calculation-graph construction."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.data.language.formatting import remove_trivial_identity_annotations

from .builder import (
    build_calculation_graph,
    walk_expression_tree,
)
from .execution import evaluate_calculation_graph
from .parser import parse_math_steps
from .postfix import (
    compile_calculation_graph,
    execute_postfix_program,
    validate_postfix_structure,
    verify_postfix_results,
)
from .postfix_schemas import (
    LiteralReference,
    LocalResultReference,
    OperandReference,
    PreviousResultReference,
    ProblemNumberReference,
    UnresolvedReference,
)
from .schemas import (
    LiteralNode,
    NumberNode,
    OperationNode,
    ProblemNumberNode,
    ReferenceNode,
    UnresolvedNode,
)


def _operand_reference_kind(reference: OperandReference) -> str:
    if isinstance(reference, ProblemNumberReference):
        return "problem_number"
    if isinstance(reference, PreviousResultReference):
        return "previous_result"
    if isinstance(reference, LocalResultReference):
        return "local_result"
    if isinstance(reference, LiteralReference):
        return "literal"
    if isinstance(reference, UnresolvedReference):
        return "unresolved"
    raise TypeError(f"Unknown operand reference: {type(reference).__name__}.")


def _nearest_rank_percentile(
    sorted_values: list[float],
    percentile: float,
) -> float | None:
    if not sorted_values:
        return None
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[rank - 1]


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


# Per-example audit


def _audit_example(payload: tuple[int, dict[str, str]]) -> dict[str, Any]:
    sample_index, row = payload
    question = row["question"]
    original_answer = row["answer"]
    answer = remove_trivial_identity_annotations(original_answer)
    math_steps = parse_math_steps(answer)
    graph = build_calculation_graph(question, math_steps)
    evaluation = evaluate_calculation_graph(graph)
    program = compile_calculation_graph(graph)
    program_evaluation = execute_postfix_program(program)
    structure_validation = validate_postfix_structure(graph, program)
    program_verification = verify_postfix_results(
        program_evaluation,
        graph_results=tuple(step.result for step in evaluation.steps),
    )

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

    execution_operator_counts: Counter[str] = Counter()
    supervision_operator_counts: Counter[str] = Counter()
    operand_reference_counts: Counter[str] = Counter()
    unresolved_candidate_counts: Counter[int] = Counter()
    actions_with_masked_operands = 0
    repeated_previous_result_occurrences = 0
    for step in program.steps:
        previous_result_indices: list[int] = []
        for action in step.actions:
            execution_operator_counts[action.operator.value] += 1
            supervision_operator_counts[action.operator.supervision_operator.value] += 1
            operand_reference_counts.update(
                _operand_reference_kind(operand) for operand in action.operands
            )
            if not all(action.operand_mask):
                actions_with_masked_operands += 1
            for operand in action.operands:
                if isinstance(operand, PreviousResultReference):
                    previous_result_indices.append(operand.step_index)
                elif isinstance(operand, UnresolvedReference):
                    unresolved_candidate_counts[len(operand.candidates)] += 1
        repeated_previous_result_occurrences += len(previous_result_indices) - len(
            set(previous_result_indices)
        )

    valid_math_steps = sum(step.valid for step in math_steps)
    final_step_identified = any(step.is_final for step in graph.steps)
    graph_execution_success = (
        evaluation.all_steps_executable and evaluation.all_steps_match_targets
    )
    program_execution_success = (
        program_evaluation.all_steps_executable
        and program_evaluation.all_steps_match_targets
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
            "reference_edge_count": sum(len(step.dependencies) for step in graph.steps),
            "reference_operand_count": node_counts["reference"],
            "unresolved_operand_count": graph.unresolved_operand_count,
            "provenance_complete": graph.provenance_complete,
            "final_step_identified": final_step_identified,
            "graph_execution_success": graph_execution_success,
            "program_execution_success": program_execution_success,
            "program_structure_valid": structure_validation.is_valid,
            "program_structure_checked_steps": (
                structure_validation.checked_step_count
            ),
            "program_structure_issue_count": len(structure_validation.issues),
            "program_structure_issue_counts": (structure_validation.issue_counts),
            "program_matches_graph": (program_verification.all_steps_match_graph),
            "program_step_count": len(program.steps),
            "program_steps_matching_graph": sum(
                step.matches_graph for step in program_verification.steps
            ),
            "program_result_mismatch_count": sum(
                not step.matches_graph for step in program_verification.steps
            ),
            "postfix_action_count": program.action_count,
            "postfix_operand_count": program.operand_count,
            "masked_operand_count": program.masked_operand_count,
            "actions_with_masked_operands": actions_with_masked_operands,
            "steps_with_masked_operands": sum(
                step.masked_operand_count > 0 for step in program.steps
            ),
            "multi_action_step_count": sum(
                len(step.actions) > 1 for step in program.steps
            ),
            "max_actions_per_step": max(
                (len(step.actions) for step in program.steps),
                default=0,
            ),
            "repeated_previous_result_occurrences": (
                repeated_previous_result_occurrences
            ),
            "unresolved_candidate_counts": {
                str(candidate_count): count
                for candidate_count, count in sorted(
                    unresolved_candidate_counts.items()
                )
            },
            "execution_operator_counts": dict(
                sorted(execution_operator_counts.items())
            ),
            "supervision_operator_counts": dict(
                sorted(supervision_operator_counts.items())
            ),
            "operand_reference_counts": dict(sorted(operand_reference_counts.items())),
            "fully_resolved_and_executable": (
                graph.provenance_complete and graph_execution_success
            ),
            "node_counts": dict(sorted(node_counts.items())),
            "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        },
        "math_steps": [step.to_dict() for step in math_steps],
        "graph": graph.to_dict(),
        "evaluation": evaluation.to_dict(),
        "program": program.to_dict(),
        "program_evaluation": program_evaluation.to_dict(),
        "program_structure_validation": structure_validation.to_dict(),
        "program_verification": program_verification.to_dict(),
    }


# Dataset-level aggregation


def _aggregate_example_reports(
    reports: list[dict[str, Any]],
    *,
    workers: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    total_steps = sum(report["metrics"]["math_step_count"] for report in reports)
    valid_steps = sum(report["metrics"]["valid_math_step_count"] for report in reports)
    node_counts: Counter[str] = Counter()
    unresolved_reasons: Counter[str] = Counter()
    execution_operator_counts: Counter[str] = Counter()
    supervision_operator_counts: Counter[str] = Counter()
    operand_reference_counts: Counter[str] = Counter()
    structure_issue_counts: Counter[str] = Counter()
    unresolved_candidate_counts: Counter[str] = Counter()
    for report in reports:
        node_counts.update(report["metrics"]["node_counts"])
        unresolved_reasons.update(report["metrics"]["unresolved_reasons"])
        execution_operator_counts.update(report["metrics"]["execution_operator_counts"])
        supervision_operator_counts.update(
            report["metrics"]["supervision_operator_counts"]
        )
        operand_reference_counts.update(report["metrics"]["operand_reference_counts"])
        structure_issue_counts.update(
            report["metrics"]["program_structure_issue_counts"]
        )
        unresolved_candidate_counts.update(
            report["metrics"]["unresolved_candidate_counts"]
        )

    example_count = len(reports)
    all_steps_valid = sum(
        report["metrics"]["all_math_steps_valid"] for report in reports
    )
    executable = sum(report["metrics"]["graph_execution_success"] for report in reports)
    program_executable = sum(
        report["metrics"]["program_execution_success"] for report in reports
    )
    program_equivalent = sum(
        report["metrics"]["program_matches_graph"] for report in reports
    )
    structurally_valid_programs = sum(
        report["metrics"]["program_structure_valid"] for report in reports
    )
    provenance_complete = sum(
        report["metrics"]["provenance_complete"] for report in reports
    )
    fully_resolved = sum(
        report["metrics"]["fully_resolved_and_executable"] for report in reports
    )
    final_identified = sum(
        report["metrics"]["final_step_identified"] for report in reports
    )
    program_step_count = sum(
        report["metrics"]["program_step_count"] for report in reports
    )
    matching_program_steps = sum(
        report["metrics"]["program_steps_matching_graph"] for report in reports
    )
    program_result_mismatches = sum(
        report["metrics"]["program_result_mismatch_count"] for report in reports
    )
    structurally_checked_steps = sum(
        report["metrics"]["program_structure_checked_steps"] for report in reports
    )
    structure_issue_count = sum(
        report["metrics"]["program_structure_issue_count"] for report in reports
    )
    postfix_action_count = sum(
        report["metrics"]["postfix_action_count"] for report in reports
    )
    postfix_operand_count = sum(
        report["metrics"]["postfix_operand_count"] for report in reports
    )
    masked_operand_count = sum(
        report["metrics"]["masked_operand_count"] for report in reports
    )
    operand_supervision_rate = (
        1 - masked_operand_count / postfix_operand_count
        if postfix_operand_count
        else 0.0
    )
    target_scales = sorted(
        step["target_scale"]
        for report in reports
        for step in report["program"]["steps"]
        if step["target_scale"] is not None
    )
    examples_without_math_steps = sum(
        report["metrics"]["math_step_count"] == 0 for report in reports
    )
    examples_with_steps_without_final = sum(
        report["metrics"]["math_step_count"] > 0
        and not report["metrics"]["final_step_identified"]
        for report in reports
    )
    return {
        "examples": example_count,
        "workers": workers,
        "worker_process_count": len({report["worker_pid"] for report in reports}),
        "worker_process_ids": sorted({report["worker_pid"] for report in reports}),
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": (
            example_count / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "math_steps": total_steps,
        "valid_math_steps": valid_steps,
        "valid_math_step_rate": valid_steps / total_steps if total_steps else 0.0,
        "examples_without_math_steps": examples_without_math_steps,
        "examples_with_steps_without_identified_final": (
            examples_with_steps_without_final
        ),
        "examples_with_all_math_steps_valid": all_steps_valid,
        "all_math_steps_valid_rate": all_steps_valid / example_count,
        "examples_with_executable_graph": executable,
        "graph_execution_rate": executable / example_count,
        "examples_with_executable_program": program_executable,
        "program_execution_rate": program_executable / example_count,
        "examples_with_program_matching_graph": program_equivalent,
        "program_graph_equivalence_rate": program_equivalent / example_count,
        "examples_with_structurally_valid_program": structurally_valid_programs,
        "program_structure_validity_rate": (
            structurally_valid_programs / example_count
        ),
        "program_structure_checked_steps": structurally_checked_steps,
        "program_structure_issues": structure_issue_count,
        "program_structure_issue_counts": dict(sorted(structure_issue_counts.items())),
        "program_steps": program_step_count,
        "program_steps_matching_graph": matching_program_steps,
        "program_result_mismatches": program_result_mismatches,
        "postfix_actions": postfix_action_count,
        "postfix_operands": postfix_operand_count,
        "masked_operands": masked_operand_count,
        "operand_supervision_rate": operand_supervision_rate,
        "actions_with_masked_operands": sum(
            report["metrics"]["actions_with_masked_operands"] for report in reports
        ),
        "steps_with_masked_operands": sum(
            report["metrics"]["steps_with_masked_operands"] for report in reports
        ),
        "multi_action_steps": sum(
            report["metrics"]["multi_action_step_count"] for report in reports
        ),
        "max_actions_per_step": max(
            report["metrics"]["max_actions_per_step"] for report in reports
        ),
        "repeated_previous_result_occurrences": sum(
            report["metrics"]["repeated_previous_result_occurrences"]
            for report in reports
        ),
        "unresolved_candidate_counts": dict(
            sorted(
                unresolved_candidate_counts.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "max_unresolved_candidates": max(
            (int(count) for count in unresolved_candidate_counts),
            default=0,
        ),
        "target_scale": {
            "count": len(target_scales),
            "min": target_scales[0] if target_scales else None,
            "p50": _nearest_rank_percentile(target_scales, 0.50),
            "p90": _nearest_rank_percentile(target_scales, 0.90),
            "p99": _nearest_rank_percentile(target_scales, 0.99),
            "max": target_scales[-1] if target_scales else None,
            "all_finite": all(math.isfinite(scale) for scale in target_scales),
        },
        "execution_operator_counts": dict(sorted(execution_operator_counts.items())),
        "supervision_operator_counts": dict(
            sorted(supervision_operator_counts.items())
        ),
        "operand_reference_counts": dict(sorted(operand_reference_counts.items())),
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


# Public entry point


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
