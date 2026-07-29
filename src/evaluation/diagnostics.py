"""Selection of scientific metrics for experiment tracking."""

from __future__ import annotations

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
