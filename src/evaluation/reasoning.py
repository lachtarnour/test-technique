"""Per-response analysis and dataset-level metric aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config import CONFIG
from src.evaluation.answers import extract_final_answer
from src.evaluation.arithmetic import (
    SUPPORTED_OPERATOR_SYMBOLS,
    parse_annotated_formulas,
)
from src.evaluation.numeric import (
    MAX_SYMMETRIC_RELATIVE_ERROR,
    REPEATING_DECIMAL_PLACES,
    normalize_numeric_value,
    numeric_symmetric_relative_error,
    numeric_values_equal,
)

STEP_AGGREGATION = "macro_per_example"


def build_evaluation_protocol(
    *,
    system_prompt: str = CONFIG.system_prompt,
    max_new_tokens: int = CONFIG.max_new_tokens,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the configuration required to reproduce an evaluation."""
    generation = dict(generation_kwargs or {})
    return {
        "prompt": system_prompt,
        "scoring": {
            "require_final_marker": False,
            "strict_metric_requires_final_marker": True,
            "strict_marker_policy": "terminal_numeric_or_fraction",
            "fallback_policy": "last_standalone_numeric_or_fraction",
            "fraction_normalization": {
                "internal": "exact_fraction",
                "repeating_decimal_places": REPEATING_DECIMAL_PLACES,
            },
            "numeric_comparison": {
                "mode": "exact_or_decimal_rounding_or_truncation",
                "minimum_decimal_places": 1,
            },
            "final_answer_error": {
                "mode": "symmetric_relative_error",
                "minimum": 0.0,
                "maximum": float(MAX_SYMMETRIC_RELATIVE_ERROR),
                "missing_prediction": float(MAX_SYMMETRIC_RELATIVE_ERROR),
            },
            "supported_operators": list(SUPPORTED_OPERATOR_SYMBOLS),
            "step_aggregation": STEP_AGGREGATION,
        },
        "generation": {
            "do_sample": generation.get("do_sample", False),
            "num_beams": generation.get("num_beams", 1),
            "max_new_tokens": generation.get(
                "max_new_tokens",
                max_new_tokens,
            ),
        },
    }


def _normalize_reference(reference_answer: str) -> str:
    marked_reference = extract_final_answer(reference_answer)
    if marked_reference is not None:
        return marked_reference

    normalized_reference = normalize_numeric_value(reference_answer)
    if normalized_reference is None:
        raise ValueError(
            "The reference answer must be a number, a fraction, "
            "or contain a valid terminal #### marker."
        )
    return normalized_reference


def analyze_response(generated_text: str, reference_answer: str) -> dict[str, Any]:
    """Evaluate the internal arithmetic consistency of one GSM8K response."""
    marked_prediction = extract_final_answer(generated_text)
    fallback_prediction = (
        extract_final_answer(
            generated_text,
            require_marker=False,
        )
        if marked_prediction is None
        else None
    )
    prediction = marked_prediction or fallback_prediction
    prediction_source = (
        "final_marker"
        if marked_prediction is not None
        else "fallback"
        if fallback_prediction is not None
        else None
    )

    reference = _normalize_reference(reference_answer)
    formulas = parse_annotated_formulas(generated_text)

    formula_count = len(formulas)
    parsed_formula_count = sum(formula.parse_success for formula in formulas)
    correct_formula_count = sum(formula.is_correct for formula in formulas)
    mean_step_accuracy = (
        correct_formula_count / formula_count if formula_count > 0 else 0.0
    )
    formula_parse_rate = (
        parsed_formula_count / formula_count if formula_count > 0 else 0.0
    )

    all_steps_correct = bool(formulas) and all(
        formula.is_correct for formula in formulas
    )
    final_step_matches_answer = (
        all_steps_correct
        and marked_prediction is not None
        and formulas[-1].evaluated_result is not None
        and numeric_values_equal(
            marked_prediction,
            formulas[-1].evaluated_result,
        )
    )
    internal_arithmetic_consistency = final_step_matches_answer
    strict_final_answer_correct = (
        marked_prediction is not None
        and numeric_values_equal(marked_prediction, reference)
    )
    final_answer_correct = prediction is not None and numeric_values_equal(
        prediction, reference
    )
    symmetric_relative_error = (
        numeric_symmetric_relative_error(prediction, reference)
        if prediction is not None
        else None
    )
    final_answer_error = float(
        symmetric_relative_error
        if symmetric_relative_error is not None
        else MAX_SYMMETRIC_RELATIVE_ERROR
    )
    correct_and_internally_consistent = (
        final_answer_correct and internal_arithmetic_consistency
    )

    return {
        "prediction": prediction,
        "marked_prediction": marked_prediction,
        "fallback_prediction": fallback_prediction,
        "prediction_source": prediction_source,
        "reference": reference,
        "generated_text": generated_text,
        "correct": final_answer_correct,
        "strict_correct": strict_final_answer_correct,
        "final_answer_error": final_answer_error,
        "final_answer_format_compliant": marked_prediction is not None,
        "reasoning": {
            "formula_count": formula_count,
            "parsed_formula_count": parsed_formula_count,
            "correct_formula_count": correct_formula_count,
            "formula_parse_rate": formula_parse_rate,
            "mean_step_arithmetic_accuracy": mean_step_accuracy,
            "all_steps_correct": all_steps_correct,
            "final_step_matches_answer": final_step_matches_answer,
            "internal_arithmetic_consistency": internal_arithmetic_consistency,
            "correct_and_internally_consistent": (correct_and_internally_consistent),
            "formulas": [formula.to_dict() for formula in formulas],
        },
    }


def aggregate_metrics(
    base_results: dict[str, Any],
    *,
    evaluation_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add arithmetic-consistency metrics to generated responses."""
    predictions = [
        {
            "question": prediction["question"],
            **analyze_response(
                generated_text=prediction["generated_text"],
                reference_answer=prediction["reference"],
            ),
        }
        for prediction in base_results["predictions"]
    ]

    total = len(predictions)
    if total == 0:
        raise ValueError("Cannot aggregate an empty prediction list.")

    correct_count = sum(prediction["correct"] for prediction in predictions)
    strict_correct_count = sum(
        prediction["strict_correct"] for prediction in predictions
    )
    internally_consistent_count = sum(
        prediction["reasoning"]["internal_arithmetic_consistency"]
        for prediction in predictions
    )
    correct_and_consistent_count = sum(
        prediction["reasoning"]["correct_and_internally_consistent"]
        for prediction in predictions
    )
    format_compliant_count = sum(
        prediction["final_answer_format_compliant"] for prediction in predictions
    )
    formula_count = sum(
        prediction["reasoning"]["formula_count"] for prediction in predictions
    )
    parsed_formula_count = sum(
        prediction["reasoning"]["parsed_formula_count"] for prediction in predictions
    )
    mean_step_accuracy = (
        sum(
            prediction["reasoning"]["mean_step_arithmetic_accuracy"]
            for prediction in predictions
        )
        / total
    )
    final_answer_error = (
        sum(prediction["final_answer_error"] for prediction in predictions) / total
    )

    protocol = (
        dict(evaluation_protocol)
        if evaluation_protocol is not None
        else build_evaluation_protocol()
    )
    return {
        "evaluation_protocol": protocol,
        "strict_final_answer_accuracy": strict_correct_count / total,
        "final_answer_accuracy": correct_count / total,
        "final_answer_error": final_answer_error,
        "mean_step_arithmetic_accuracy": mean_step_accuracy,
        "internal_arithmetic_consistency_rate": (internally_consistent_count / total),
        "correct_and_internally_consistent_rate": (
            correct_and_consistent_count / total
        ),
        "diagnostics": {
            "final_answer_format_compliance_rate": (format_compliant_count / total),
            "formula_parse_rate": (
                parsed_formula_count / formula_count if formula_count > 0 else 0.0
            ),
        },
        "counts": {
            "strict_correct_final_answers": strict_correct_count,
            "correct_final_answers": correct_count,
            "internally_consistent_answers": internally_consistent_count,
            "correct_and_internally_consistent": (correct_and_consistent_count),
            "parsed_formulas": parsed_formula_count,
            "formulas": formula_count,
            "total": total,
        },
        "elapsed_seconds": base_results["elapsed_seconds"],
        "samples_per_second": base_results["samples_per_second"],
        "total": total,
        "predictions": predictions,
    }
