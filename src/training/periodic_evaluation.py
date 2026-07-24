"""Fixed-sample generation evaluation during training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from accelerate.utils.memory import clear_device_cache
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState

from src.config import CONFIG
from src.evaluation import evaluate_model
from src.load_data import load_frozen_gsm8k_split
from src.tracking import flatten_numeric_metrics

PERIODIC_EVAL_SAMPLES_PER_SPLIT = 300
PERIODIC_EVAL_BATCH_SIZE = 300
PERIODIC_EVAL_SEED = CONFIG.seed
PERIODIC_EVAL_SPLITS = ("train", "validation")


def dataset_content_sha256(dataset: Dataset) -> str:
    """Hash ordered questions and answers to identify an exact evaluation subset."""
    digest = hashlib.sha256()
    for question, answer in zip(
        dataset["question"],
        dataset["answer"],
        strict=True,
    ):
        encoded = json.dumps(
            {"question": question, "answer": answer},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_fixed_periodic_evaluation_datasets(
    dataset_path: str | Path,
) -> dict[str, Dataset]:
    """Load the same 300 train and validation examples for every experiment run."""
    return {
        split: load_frozen_gsm8k_split(
            split,
            dataset_path=dataset_path,
            subset_size=PERIODIC_EVAL_SAMPLES_PER_SPLIT,
            seed=PERIODIC_EVAL_SEED,
        )
        for split in PERIODIC_EVAL_SPLITS
    }


def periodic_evaluation_metadata(
    datasets: Mapping[str, Dataset],
) -> dict[str, Any]:
    """Describe and fingerprint the fixed periodic-evaluation datasets."""
    return {
        "seed": PERIODIC_EVAL_SEED,
        "samples_per_split": PERIODIC_EVAL_SAMPLES_PER_SPLIT,
        "splits": {
            split: {
                "examples": len(dataset),
                "sha256": dataset_content_sha256(dataset),
            }
            for split, dataset in datasets.items()
        },
    }


class PeriodicGenerationEvaluationCallback(TrainerCallback):
    """Evaluate fixed train/validation samples at epochs 0, N, 2N, ..."""

    def __init__(
        self,
        *,
        datasets: Mapping[str, Dataset],
        tokenizer: Any,
        eval_every: int,
        batch_size: int,
        max_new_tokens: int,
        wandb_run: Any | None = None,
        evaluator: Callable[..., dict[str, Any]] = evaluate_model,
        clear_cache: Callable[..., None] = clear_device_cache,
    ) -> None:
        if isinstance(eval_every, bool) or not isinstance(eval_every, int):
            raise ValueError("eval_every must be a strictly positive integer.")
        if eval_every <= 0:
            raise ValueError("eval_every must be a strictly positive integer.")
        for name, value in (
            ("batch_size", batch_size),
            ("max_new_tokens", max_new_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a strictly positive integer.")
        if set(datasets) != set(PERIODIC_EVAL_SPLITS):
            raise ValueError(
                "Periodic evaluation requires fixed train and validation datasets."
            )

        self.datasets = dict(datasets)
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.wandb_run = wandb_run
        self.evaluator = evaluator
        self.clear_cache = clear_cache
        self.history: list[dict[str, Any]] = []
        self._evaluated_epochs: set[int] = set()

    @staticmethod
    def _summary(results: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in results.items() if key != "predictions"}

    def _evaluate_epoch(
        self,
        *,
        epoch: int,
        state: TrainerState,
        model: Any,
    ) -> None:
        if epoch in self._evaluated_epochs or not state.is_world_process_zero:
            return

        model.zero_grad(set_to_none=True)
        self.clear_cache(garbage_collection=True)
        was_training = model.training
        original_use_cache = getattr(model.config, "use_cache", None)
        split_summaries: dict[str, Any] = {}
        try:
            model.config.use_cache = True
            model.eval()
            for split in PERIODIC_EVAL_SPLITS:
                results = self.evaluator(
                    model,
                    self.tokenizer,
                    self.datasets[split],
                    batch_size=self.batch_size,
                    max_new_tokens=self.max_new_tokens,
                )
                split_summaries[split] = self._summary(results)
        finally:
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache
            if was_training:
                model.train()
            self.clear_cache(garbage_collection=True)

        record = {
            "epoch": epoch,
            "global_step": state.global_step,
            "splits": split_summaries,
        }
        self.history.append(record)
        self._evaluated_epochs.add(epoch)

        if self.wandb_run is not None:
            payload: dict[str, int | float] = {
                "periodic_evaluation/epoch": epoch,
                "train/global_step": state.global_step,
            }
            for split, metrics in split_summaries.items():
                payload.update(
                    flatten_numeric_metrics(
                        metrics,
                        prefix=f"periodic_evaluation/{split}",
                    )
                )
            self.wandb_run.log(payload)

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        *,
        model: Any,
        **kwargs: Any,
    ) -> TrainerControl:
        del args, kwargs
        self._evaluate_epoch(epoch=0, state=state, model=model)
        return control

    def on_epoch_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        *,
        model: Any,
        **kwargs: Any,
    ) -> TrainerControl:
        del args, kwargs
        if state.epoch is None:
            return control

        completed_epoch = math.floor(state.epoch + 1e-9)
        if completed_epoch > 0 and completed_epoch % self.eval_every == 0:
            self._evaluate_epoch(
                epoch=completed_epoch,
                state=state,
                model=model,
            )
        return control
