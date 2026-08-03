"""Build the common Hugging Face infrastructure for one selected ablation."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    PrinterCallback,
    TrainerCallback,
    TrainingArguments,
)

from src.config import CONFIG
from src.data.features import BASE_MODEL_COLUMNS
from src.data.language.collator import collate_completion_only
from src.model.device import cuda_supports_native_bf16
from src.training.callbacks import EpochIntervalCallback, StructuredLoggingCallback
from src.training.trainer import MathConsistencyTrainer


def validate_tokenized_dataset(
    dataset: DatasetDict,
    *,
    required_columns: frozenset[str] = BASE_MODEL_COLUMNS,
) -> None:
    """Fail early if a split cannot enter the common Trainer."""
    missing_splits = {"train", "validation"} - set(dataset)
    if missing_splits:
        raise ValueError(f"Missing dataset splits: {sorted(missing_splits)}")
    for split_name in ("train", "validation"):
        split = dataset[split_name]
        if not len(split):
            raise ValueError(f"The {split_name!r} split must not be empty.")
        missing_columns = required_columns - set(split.column_names)
        if missing_columns:
            raise ValueError(
                f"Missing tokenized columns in {split_name!r}: "
                f"{sorted(missing_columns)}"
            )


def build_training_arguments(
    *,
    output_dir: str | Path,
    num_train_epochs: float = CONFIG.num_train_epochs,
    max_steps: int | None = None,
    train_batch_size: int = CONFIG.train_batch_size,
    eval_batch_size: int = CONFIG.eval_batch_size,
    gradient_accumulation_steps: int = CONFIG.gradient_accumulation_steps,
    learning_rate: float = 1e-4,
    logging_steps: int = 10,
    validation_every_epochs: int = 1,
    log_every_epochs: int = 1,
    run_name: str | None = None,
    report_to_wandb: bool = True,
    seed: int = CONFIG.seed,
) -> TrainingArguments:
    """Create infrastructure arguments that stay identical across experiments."""
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a strictly positive integer.")

    positive_values = {
        "num_train_epochs": num_train_epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "logging_steps": logging_steps,
        "validation_every_epochs": validation_every_epochs,
        "log_every_epochs": log_every_epochs,
    }
    for name, value in positive_values.items():
        if (name.endswith("_every_epochs") or name == "logging_steps") and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"{name} must be a strictly positive integer.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be strictly positive.")

    use_cuda = torch.cuda.is_available()
    use_bf16 = cuda_supports_native_bf16(torch)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_drop_last=CONFIG.drop_incomplete_train_batch,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "threshold": 0.005,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 1e-5,
        },
        warmup_ratio=0.0,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=logging_steps,
        disable_tqdm=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        average_tokens_across_devices=True,
        seed=seed,
        data_seed=seed,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        report_to=["wandb"] if report_to_wandb else "none",
        run_name=run_name,
    )
    arguments.validation_every_epochs = validation_every_epochs
    arguments.log_every_epochs = log_every_epochs
    arguments.early_stopping_patience = 6
    arguments.early_stopping_threshold = 1e-3
    return arguments


def build_training_trainer(
    *,
    model: torch.nn.Module,
    dataset: DatasetDict,
    tokenizer: Any,
    training_arguments: TrainingArguments,
    experiment: dict[str, Any],
    callbacks: list[TrainerCallback] | None = None,
) -> MathConsistencyTrainer:
    """Build the single Trainer used by every implemented loss recipe."""
    validate_tokenized_dataset(
        dataset,
        required_columns=BASE_MODEL_COLUMNS | frozenset(experiment["features"]),
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id.")
    trainer = MathConsistencyTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=partial(
            collate_completion_only,
            pad_token_id=tokenizer.pad_token_id,
        ),
        processing_class=tokenizer,
        losses=experiment["losses"],
        head_names=frozenset(experiment["heads"]),
        callbacks=[
            StructuredLoggingCallback(),
            EpochIntervalCallback(
                validation_every_epochs=(training_arguments.validation_every_epochs),
                log_every_epochs=training_arguments.log_every_epochs,
            ),
            EarlyStoppingCallback(
                early_stopping_patience=(training_arguments.early_stopping_patience),
                early_stopping_threshold=(training_arguments.early_stopping_threshold),
            ),
            *(callbacks or []),
        ],
    )
    trainer.remove_callback(PrinterCallback)
    return trainer
