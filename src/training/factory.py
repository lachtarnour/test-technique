"""Build common Hugging Face infrastructure from a compiled experiment plan."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from transformers import PrinterCallback, TrainerCallback

from src.config import CONFIG
from src.data.language.collator import CompletionOnlyDataCollator
from src.model.device import cuda_supports_native_bf16
from src.training.arguments import ExperimentTrainingArguments
from src.training.callbacks import EpochIntervalCallback, StructuredLoggingCallback
from src.training.plan import (
    BASE_MODEL_COLUMNS,
    ExperimentPlan,
    build_objective,
)
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
    num_train_epochs: float = 1.0,
    max_steps: int | None = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,
    validation_every_epochs: int = 1,
    log_every_epochs: int = 1,
    run_name: str | None = None,
    report_to_wandb: bool = True,
    seed: int = CONFIG.seed,
) -> ExperimentTrainingArguments:
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
    return ExperimentTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
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
        validation_every_epochs=validation_every_epochs,
        log_every_epochs=log_every_epochs,
    )


def build_training_trainer(
    *,
    model: torch.nn.Module,
    dataset: DatasetDict,
    tokenizer: Any,
    training_arguments: ExperimentTrainingArguments,
    experiment_plan: ExperimentPlan,
    callbacks: list[TrainerCallback] | None = None,
) -> MathConsistencyTrainer:
    """Build the single Trainer used by every implemented loss recipe."""
    validate_tokenized_dataset(
        dataset,
        required_columns=experiment_plan.required_columns,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id.")
    trainer = MathConsistencyTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=CompletionOnlyDataCollator(pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
        objective=build_objective(experiment_plan.config),
        experiment_plan=experiment_plan,
        callbacks=[
            StructuredLoggingCallback(),
            EpochIntervalCallback(
                validation_every_epochs=(training_arguments.validation_every_epochs),
                log_every_epochs=training_arguments.log_every_epochs,
            ),
            *(callbacks or []),
        ],
    )
    trainer.remove_callback(PrinterCallback)
    return trainer
