"""Thin Hugging Face Trainer adapter for an injectable objective."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from src.model.heads import get_auxiliary_heads
from src.training.objective import (
    ScalarNormalizer,
    compute_objective,
    normalization_counts,
)

_MODEL_INPUTS = frozenset({"input_ids", "attention_mask"})
_GIBIBYTE = 1024**3


def _current_cuda_memory_metrics() -> dict[str, float]:
    """Return the process allocator state without synchronizing the GPU."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_memory_allocated_gib": torch.cuda.memory_allocated() / _GIBIBYTE,
        "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / _GIBIBYTE,
    }


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
    """Common orchestration layer shared by every training ablation."""

    def __init__(
        self,
        *args: Any,
        losses: dict[str, float],
        head_names: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.losses = dict(losses)
        self.head_names = head_names
        self.needs_last_hidden_state = bool(head_names)
        self.model_accepts_loss_kwargs = True
        self.label_names = ["labels"]
        self._training_normalizers: dict[str, ScalarNormalizer] = {}
        self._evaluation_normalizers_per_example: dict[str, float] | None = None
        self._component_statistics: dict[
            str,
            dict[str, tuple[torch.Tensor, torch.Tensor]],
        ] = {"train": {}, "eval": {}}

    def get_eval_dataloader(self, eval_dataset: Any | None = None) -> Any:
        """Keep every validation example even when train drops its remainder."""
        original_drop_last = self.args.dataloader_drop_last
        self.args.dataloader_drop_last = False
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.args.dataloader_drop_last = original_drop_last

    def _record_component_statistics(
        self,
        *,
        phase: str,
        statistics: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Accumulate additive statistics without retaining autograd graphs."""
        all_statistics = getattr(self, "_component_statistics", None)
        if all_statistics is None:
            all_statistics = {"train": {}, "eval": {}}
            self._component_statistics = all_statistics
        phase_statistics = all_statistics[phase]
        for name, (numerator, denominator) in statistics.items():
            numerator = numerator.detach()
            denominator = denominator.detach()
            previous = phase_statistics.get(name)
            if previous is not None:
                numerator = previous[0] + numerator
                denominator = previous[1] + denominator
            phase_statistics[name] = (numerator, denominator)

    @staticmethod
    def _forward_with_last_hidden_state(
        model: torch.nn.Module,
        model_inputs: dict[str, Any],
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Capture the final hidden state without retaining every layer."""
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if not callable(get_output_embeddings):
            raise RuntimeError(
                "Structured losses require model.get_output_embeddings()."
            )
        output_embeddings = get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError(
                "Structured losses require a registered output embedding head."
            )

        captured: list[torch.Tensor] = []

        def capture_input(
            _module: torch.nn.Module,
            arguments: tuple[Any, ...],
        ) -> None:
            if not arguments or not isinstance(arguments[0], torch.Tensor):
                raise RuntimeError(
                    "Could not capture the final hidden state before lm_head."
                )
            captured.append(arguments[0])

        handle = output_embeddings.register_forward_pre_hook(capture_input)
        try:
            raw_outputs = model(**model_inputs)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("The model forward did not invoke its lm_head.")
        return raw_outputs, {
            "logits": raw_outputs.logits,
            "last_hidden_state": captured[-1],
        }

    def _consume_component_metrics(self, phase: str) -> dict[str, float]:
        """Return globally reduced component means and clear the phase."""
        all_statistics = getattr(self, "_component_statistics", None)
        if not all_statistics:
            return {}
        phase_statistics = all_statistics[phase]
        metrics: dict[str, float] = {}
        accelerator = getattr(self, "accelerator", None)
        for name, (numerator, denominator) in phase_statistics.items():
            totals = torch.stack(
                (
                    numerator.to(dtype=torch.float32),
                    denominator.to(
                        device=numerator.device,
                        dtype=torch.float32,
                    ),
                )
            )
            if accelerator is not None:
                totals = accelerator.reduce(totals, reduction="sum")
            if totals[1].item() <= 0:
                continue
            metrics[name] = (totals[0] / totals[1]).item()
        phase_statistics.clear()
        return metrics

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        """Inject exact per-component means into standard Trainer logs."""
        is_evaluation = any(key.startswith("eval_") for key in logs)
        phase = "eval" if is_evaluation else "train"
        prefix = "eval_" if is_evaluation else ""
        for name, value in self._consume_component_metrics(phase).items():
            logs.setdefault(f"{prefix}{name}", value)
        if not is_evaluation:
            logs.update(_current_cuda_memory_metrics())
        super().log(logs, start_time=start_time)

    def _get_num_items_in_batch(
        self,
        batch_samples: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor | int | None:
        """Compute one exact normalizer per loss over the accumulation window."""
        normalizers = normalization_counts(self.losses, batch_samples)
        if not normalizers:
            return super()._get_num_items_in_batch(batch_samples, device)

        self._training_normalizers = {
            name: self._globalize_normalizer(value, device=device)
            for name, value in normalizers.items()
        }
        language_name = next(
            name for name in self.losses if name.startswith("language")
        )
        language_count = self._training_normalizers.get(language_name)
        if isinstance(language_count, (int, torch.Tensor)):
            return language_count
        return None

    def _globalize_normalizer(
        self,
        value: ScalarNormalizer,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Apply the same distributed scaling contract to every loss."""
        count = (
            value.to(device)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=device)
        )
        if self.args.average_tokens_across_devices and self.args.world_size >= 1:
            count = self.accelerator.gather(count).sum()
        elif self.args.n_gpu >= 1:
            count = count / self.args.n_gpu

        if self.args.n_gpu > 1 and count.dim() == 0:
            count = count.unsqueeze(0).expand(self.args.n_gpu, -1)
        parallelism_config = getattr(
            self.accelerator,
            "parallelism_config",
            None,
        )
        if parallelism_config is not None:
            count = count / parallelism_config.non_data_parallel_size
        return count

    def _count_evaluation_normalizers(
        self,
        dataset: Any,
    ) -> tuple[int, dict[str, float]]:
        """Count all valid targets once for exact validation means."""
        try:
            example_count = len(dataset)
        except TypeError as exc:
            raise TypeError(
                "Exact normalized eval_loss requires a sized dataset."
            ) from exc
        if example_count <= 0:
            raise ValueError("The evaluation dataset must not be empty.")

        totals: dict[str, float] = {}
        for start in range(0, example_count, 1024):
            stop = min(start + 1024, example_count)
            rows = [dataset[index] for index in range(start, stop)]
            batch = {key: [row[key] for row in rows] for key in rows[0]}
            counts = normalization_counts(self.losses, [batch])
            for name, value in counts.items():
                scalar = (
                    value.detach().item()
                    if isinstance(value, torch.Tensor)
                    else float(value)
                )
                totals[name] = totals.get(name, 0.0) + scalar
        invalid = sorted(name for name, count in totals.items() if count <= 0)
        if invalid:
            raise ValueError(
                f"Evaluation losses have no supervised elements: {invalid}"
            )
        return example_count, totals

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        """Aggregate each validation loss over all its supervised elements."""
        example_count, normalizers = self._count_evaluation_normalizers(
            dataloader.dataset
        )
        self._evaluation_normalizers_per_example = {
            name: count / example_count for name, count in normalizers.items()
        }
        try:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self._evaluation_normalizers_per_example = None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        model_inputs = {
            key: value for key, value in inputs.items() if key in _MODEL_INPUTS
        }
        if getattr(self, "needs_last_hidden_state", False):
            outputs, objective_outputs = self._forward_with_last_hidden_state(
                model,
                model_inputs,
            )
        else:
            outputs = model(**model_inputs)
            objective_outputs = {"logits": outputs.logits}
        head_names = getattr(self, "head_names", frozenset())
        auxiliary_heads = get_auxiliary_heads(model, head_names)
        is_training = model.training
        normalizers = (
            dict(getattr(self, "_training_normalizers", {})) if is_training else {}
        )
        if num_items_in_batch is not None:
            language_name = next(
                name for name in self.losses if name.startswith("language")
            )
            normalizers.setdefault(language_name, num_items_in_batch)
        evaluation_normalizers_per_example = getattr(
            self,
            "_evaluation_normalizers_per_example",
            None,
        )
        if not is_training and evaluation_normalizers_per_example is not None:
            normalizers = {
                name: inputs["labels"].shape[0] * per_example
                for name, per_example in (evaluation_normalizers_per_example.items())
            }
        total_loss, statistics = compute_objective(
            self.losses,
            model_outputs=objective_outputs,
            batch=inputs,
            normalizers=normalizers,
            auxiliary_heads=auxiliary_heads,
        )
        self._record_component_statistics(
            phase="train" if is_training else "eval",
            statistics=statistics,
        )
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
