"""Standard Hugging Face callbacks used by every training experiment."""

from __future__ import annotations

import json
import math
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState


class StructuredLoggingCallback(TrainerCallback):
    """Print Trainer metrics as one JSON record per logging event."""

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        *,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del args, control, kwargs
        if not state.is_world_process_zero or not logs:
            return

        if any(key.startswith("eval_") for key in logs):
            phase = "validation"
        elif any(key.startswith("train_") for key in logs):
            phase = "training_summary"
        else:
            phase = "training"
        payload = {
            "phase": phase,
            "event": "metrics",
            "global_step": state.global_step,
            **logs,
        }
        print(
            json.dumps(
                payload,
                default=str,
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


class EpochIntervalCallback(TrainerCallback):
    """Apply logging, validation and checkpoint intervals in epoch units."""

    def __init__(
        self,
        *,
        validation_every_epochs: int,
        log_every_epochs: int,
    ) -> None:
        self.validation_every_epochs = validation_every_epochs
        self.log_every_epochs = log_every_epochs

    def on_epoch_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        del args, kwargs
        if state.epoch is None:
            return control

        completed_epoch = max(1, math.ceil(state.epoch - 1e-9))
        is_final_epoch = state.global_step >= state.max_steps
        should_validate = (
            completed_epoch % self.validation_every_epochs == 0 or is_final_epoch
        )

        control.should_log = (
            completed_epoch % self.log_every_epochs == 0 or is_final_epoch
        )
        control.should_evaluate = should_validate
        control.should_save = should_validate
        return control
