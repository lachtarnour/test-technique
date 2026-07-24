"""Shared diagnostics for GPU evaluation reports and CPU rescoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

METRIC_KEYS = (
    "strict_final_answer_accuracy",
    "final_answer_accuracy",
    "final_answer_error",
    "mean_step_arithmetic_accuracy",
    "internal_arithmetic_consistency_rate",
    "correct_and_internally_consistent_rate",
)
SAMPLE_LIMIT = 50


def select_evaluation_metrics(
    results: Mapping[str, Any],
) -> dict[str, int | float]:
    """Return only the scientific metrics intended for experiment tracking."""
    return {
        key: value
        for key in METRIC_KEYS
        if isinstance((value := results.get(key)), (int, float))
        and not isinstance(value, bool)
    }


def diagnose_evaluation(results: dict[str, Any]) -> dict[str, Any]:
    """Summarize format and arithmetic failures in generated predictions."""
    counts: Counter[str] = Counter(
        {
            "responses": 0,
            "missing_predictions": 0,
            "responses_without_final_marker": 0,
            "fallback_predictions": 0,
            "correct_fallback_predictions": 0,
            "incorrect_fallback_predictions": 0,
            "responses_without_formulas": 0,
            "internally_inconsistent_responses": 0,
            "formulas": 0,
            "unparsed_formulas": 0,
            "execution_errors": 0,
            "incorrect_arithmetic": 0,
        }
    )
    error_types: Counter[str] = Counter()
    samples: dict[str, list[int]] = {
        "missing_prediction": [],
        "without_final_marker": [],
        "correct_fallback": [],
        "incorrect_fallback": [],
        "without_formulas": [],
        "unparsed_formulas": [],
        "execution_errors": [],
        "incorrect_arithmetic": [],
        "internally_inconsistent": [],
    }

    for index, prediction in enumerate(results["predictions"]):
        counts["responses"] += 1
        if prediction["prediction"] is None:
            counts["missing_predictions"] += 1
            samples["missing_prediction"].append(index)
        if not prediction["final_answer_format_compliant"]:
            counts["responses_without_final_marker"] += 1
            samples["without_final_marker"].append(index)
        if prediction["prediction_source"] == "fallback":
            counts["fallback_predictions"] += 1
            if prediction["correct"]:
                counts["correct_fallback_predictions"] += 1
                samples["correct_fallback"].append(index)
            else:
                counts["incorrect_fallback_predictions"] += 1
                samples["incorrect_fallback"].append(index)

        reasoning = prediction["reasoning"]
        formulas = reasoning["formulas"]
        if not formulas:
            counts["responses_without_formulas"] += 1
            samples["without_formulas"].append(index)
        if not reasoning["internal_arithmetic_consistency"]:
            counts["internally_inconsistent_responses"] += 1
            samples["internally_inconsistent"].append(index)

        counts["formulas"] += len(formulas)
        for formula in formulas:
            if not formula["parse_success"]:
                counts["unparsed_formulas"] += 1
                samples["unparsed_formulas"].append(index)
            elif not formula["execution_success"]:
                counts["execution_errors"] += 1
                samples["execution_errors"].append(index)
            elif not formula["arithmetic_correct"]:
                counts["incorrect_arithmetic"] += 1
                samples["incorrect_arithmetic"].append(index)
            if formula["error"] is not None:
                error_types[formula["error"]] += 1

    return {
        **counts,
        "formula_error_types": dict(sorted(error_types.items())),
        "sample_indices": {
            name: indices[:SAMPLE_LIMIT] for name, indices in samples.items()
        },
        "sample_limit": SAMPLE_LIMIT,
    }


def compare_evaluations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two runs performed with the same frozen protocol."""
    reference_predictions = reference["predictions"]
    candidate_predictions = candidate["predictions"]
    same_size = len(reference_predictions) == len(candidate_predictions)
    comparable_count = min(len(reference_predictions), len(candidate_predictions))
    changed_indices = [
        index
        for index in range(comparable_count)
        if (
            reference_predictions[index]["question"]
            != candidate_predictions[index]["question"]
            or reference_predictions[index]["generated_text"]
            != candidate_predictions[index]["generated_text"]
        )
    ]
    return {
        "same_protocol": (
            reference["evaluation_protocol"] == candidate["evaluation_protocol"]
        ),
        "same_size": same_size,
        "exact_generation_match": same_size and not changed_indices,
        "changed_generation_count": len(changed_indices),
        "changed_generation_indices": changed_indices[:SAMPLE_LIMIT],
        "metric_deltas": {key: candidate[key] - reference[key] for key in METRIC_KEYS},
    }
