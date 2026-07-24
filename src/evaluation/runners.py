"""Public model evaluation entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.config import CONFIG
from src.evaluation.generation import (
    generate_checkpoint_responses,
    generate_model_responses,
    generate_pretrained_responses,
)
from src.evaluation.reasoning import aggregate_metrics, build_evaluation_protocol


def _evaluate_and_aggregate(
    evaluator: Callable[..., dict[str, Any]],
    *args: Any,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    evaluation_kwargs.setdefault("system_prompt", CONFIG.system_prompt)
    protocol = build_evaluation_protocol(
        system_prompt=evaluation_kwargs["system_prompt"],
        max_new_tokens=evaluation_kwargs.get(
            "max_new_tokens",
            CONFIG.max_new_tokens,
        ),
        generation_kwargs=evaluation_kwargs.get("generation_kwargs"),
    )
    base_results = evaluator(*args, **evaluation_kwargs)
    return aggregate_metrics(
        base_results,
        evaluation_protocol=protocol,
    )


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate an already loaded model."""
    return _evaluate_and_aggregate(
        generate_model_responses,
        model,
        tokenizer,
        dataset,
        **evaluation_kwargs,
    )


def evaluate_pretrained_model(
    model_name: str,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Load and evaluate a pretrained model."""
    return _evaluate_and_aggregate(
        generate_pretrained_responses,
        model_name,
        **evaluation_kwargs,
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Load and evaluate a LoRA checkpoint."""
    return _evaluate_and_aggregate(
        generate_checkpoint_responses,
        checkpoint_path,
        **evaluation_kwargs,
    )
