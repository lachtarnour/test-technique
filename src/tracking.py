"""Small, testable Weights & Biases integration helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

WANDB_MODES = ("online", "offline", "disabled")
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def configure_wandb_api_key(
    env_file: str | Path | None = None,
) -> str | None:
    """Load ``WANDB_KEY`` from .env and expose it under W&B's official name."""
    resolved_env_file = (
        Path(env_file)
        if env_file is not None
        else Path(os.getenv("WANDB_ENV_FILE", DEFAULT_ENV_FILE))
    )
    load_dotenv(dotenv_path=resolved_env_file, override=False)

    official_key = os.getenv("WANDB_API_KEY")
    if official_key:
        return official_key

    wandb_key = os.getenv("WANDB_KEY")
    if not wandb_key or not wandb_key.strip():
        return None

    normalized_key = wandb_key.strip()
    os.environ["WANDB_API_KEY"] = normalized_key
    return normalized_key


def initialize_wandb_run(
    *,
    project: str,
    run_name: str,
    job_type: str,
    mode: str,
    config: Mapping[str, Any],
) -> Any | None:
    """Create a W&B run, or return ``None`` when tracking is disabled."""
    if mode not in WANDB_MODES:
        raise ValueError(f"Unsupported W&B mode: {mode}")
    if mode == "disabled":
        return None

    configure_wandb_api_key()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases is required for tracking. Install the runtime "
            "dependencies or use --wandb-mode disabled."
        ) from exc

    return wandb.init(
        project=project,
        name=run_name,
        job_type=job_type,
        mode=mode,
        config=dict(config),
    )


def flatten_numeric_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str | None = None,
) -> dict[str, int | float]:
    """Flatten finite numeric values for W&B while omitting bulky predictions."""
    flattened: dict[str, int | float] = {}

    def visit(values: Mapping[str, Any], path: tuple[str, ...]) -> None:
        for key, value in values.items():
            if key == "predictions":
                continue
            next_path = (*path, str(key))
            if isinstance(value, Mapping):
                visit(value, next_path)
            elif (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                flattened["/".join(next_path)] = value

    initial_path = (prefix,) if prefix else ()
    visit(metrics, initial_path)
    return flattened


def log_wandb_metrics(
    run: Any | None,
    *,
    model_name: str,
    experiment_name: str,
    metrics: Mapping[str, Any],
    prefix: str | None = None,
) -> dict[str, int | float]:
    """Log metrics and retain model/experiment identity in the run summary."""
    if run is None:
        return {}

    numeric_metrics = flatten_numeric_metrics(metrics, prefix=prefix)
    if numeric_metrics:
        run.log(numeric_metrics)
    run.summary.update(
        {
            "model_name": model_name,
            "experiment_name": experiment_name,
            **numeric_metrics,
        }
    )
    return numeric_metrics


def finish_wandb_run(run: Any | None, *, exit_code: int = 0) -> None:
    """Close an initialized W&B run."""
    if run is not None:
        run.finish(exit_code=exit_code)
