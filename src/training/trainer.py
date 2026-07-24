"""Thin Hugging Face Trainer adapter for an injectable objective."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState

from src.training.objective import ObjectiveContext, TrainingObjective

_MODEL_INPUTS = frozenset({"input_ids", "attention_mask"})


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
        # Keep checkpoint selection aligned with the validation cadence.
        control.should_save = should_validate
        return control


def release_training_memory(trainer: Trainer) -> None:
    """Release training-only state and clear the device cache before generation."""
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    trainer.model.zero_grad(set_to_none=True)

    # Trainer and Accelerator both retain references to the optimizer.
    trainer.optimizer = None
    trainer.lr_scheduler = None
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        # Clears Accelerator's optimizer/scheduler/dataloader references, runs
        # garbage collection and empties the active CUDA or MPS device cache.
        accelerator.free_memory()
    else:
        from accelerate.utils.memory import clear_device_cache

        clear_device_cache(garbage_collection=True)


class MathConsistencyTrainer(Trainer):
    """Common orchestration layer, currently exercised by A1 only."""

    def __init__(
        self,
        *args: Any,
        objective: TrainingObjective,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.model_accepts_loss_kwargs = True
        self.label_names = ["labels"]
        self._evaluation_tokens_per_example: float | None = None

    def _get_num_items_in_batch(
        self,
        batch_samples: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor | int | None:
        """Count causal targets across the complete accumulation window."""
        shifted_samples: list[dict[str, Any]] = []
        for batch in batch_samples:
            labels = batch.get("labels")
            if not isinstance(labels, torch.Tensor) or labels.ndim < 2:
                return None
            shifted_samples.append({**batch, "labels": labels[..., 1:]})
        return super()._get_num_items_in_batch(shifted_samples, device)

    @staticmethod
    def _count_dataset_supervised_tokens(dataset: Any) -> tuple[int, int]:
        """Return examples and valid causal targets for one evaluation dataset."""
        try:
            example_count = len(dataset)
        except TypeError as exc:
            raise TypeError(
                "Exact token-normalized eval_loss requires a sized dataset."
            ) from exc
        if example_count <= 0:
            raise ValueError("The evaluation dataset must not be empty.")

        try:
            label_rows = dataset["labels"]
        except (KeyError, TypeError):
            label_rows = [dataset[index]["labels"] for index in range(example_count)]

        supervised_tokens = sum(
            sum(int(label) != -100 for label in labels[1:]) for labels in label_rows
        )
        if supervised_tokens <= 0:
            raise ValueError(
                "The evaluation dataset must contain supervised completion tokens."
            )
        return example_count, supervised_tokens

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        """Aggregate eval_loss as one mean over all supervised tokens."""
        example_count, supervised_tokens = self._count_dataset_supervised_tokens(
            dataloader.dataset
        )
        self._evaluation_tokens_per_example = supervised_tokens / example_count
        try:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self._evaluation_tokens_per_example = None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        outputs = model(
            **{key: value for key, value in inputs.items() if key in _MODEL_INPUTS}
        )
        is_training = model.training
        normalization_count: int | float | torch.Tensor | None = num_items_in_batch
        evaluation_tokens_per_example = getattr(
            self,
            "_evaluation_tokens_per_example",
            None,
        )
        if not is_training and evaluation_tokens_per_example is not None:
            normalization_count = (
                inputs["labels"].shape[0] * evaluation_tokens_per_example
            )
        loss_output = self.objective.compute_loss(
            model_outputs=outputs,
            batch=inputs,
            context=ObjectiveContext(
                global_step=self.state.global_step,
                is_training=is_training,
                normalization_count=normalization_count,
            ),
        )
        total_loss = loss_output.total_loss
        if (
            is_training
            and num_items_in_batch is not None
            and getattr(
                getattr(self, "args", None),
                "average_tokens_across_devices",
                False,
            )
        ):
            process_count = (
                self.args.n_gpu
                if self.args.n_gpu > 1
                else self.accelerator.num_processes
            )
            total_loss = total_loss * process_count
        if return_outputs:
            return total_loss, outputs
        return total_loss
