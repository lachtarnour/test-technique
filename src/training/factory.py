"""Build the common Hugging Face training infrastructure."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from transformers import TrainerCallback

from src.config import CONFIG
from src.data.collator import CompletionOnlyDataCollator
from src.device import cuda_supports_native_bf16
from src.training.arguments import (
    CAUSAL_LANGUAGE_MODELING,
    ExperimentConfig,
    ExperimentTrainingArguments,
)
from src.training.objective import CausalLanguageModelingObjective
from src.training.trainer import EpochIntervalCallback, MathConsistencyTrainer

REQUIRED_COLUMNS = frozenset({"input_ids", "attention_mask", "labels"})


def validate_tokenized_dataset(dataset: DatasetDict) -> None:
    """Fail early if a split cannot enter the common Trainer."""
    missing_splits = {"train", "validation"} - set(dataset)
    if missing_splits:
        raise ValueError(f"Missing dataset splits: {sorted(missing_splits)}")
    for split_name in ("train", "validation"):
        split = dataset[split_name]
        if not len(split):
            raise ValueError(f"The {split_name!r} split must not be empty.")
        missing_columns = REQUIRED_COLUMNS - set(split.column_names)
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
    validation_every_epochs: int = 1,
    log_every_epochs: int = 1,
    eval_every: int = 2,
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
        "validation_every_epochs": validation_every_epochs,
        "log_every_epochs": log_every_epochs,
        "eval_every": eval_every,
    }
    for name, value in positive_values.items():
        if (name.endswith("_every_epochs") or name == "eval_every") and (
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
        logging_strategy="epoch",
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
        eval_every=eval_every,
    )


def build_training_trainer(
    *,
    model: torch.nn.Module,
    experiment_config: ExperimentConfig,
    dataset: DatasetDict,
    tokenizer: Any,
    training_arguments: ExperimentTrainingArguments,
    callbacks: list[TrainerCallback] | None = None,
) -> MathConsistencyTrainer:
    """Build A1 through the objective-injection seam used later."""
    validate_tokenized_dataset(dataset)
    if experiment_config.objective != CAUSAL_LANGUAGE_MODELING:
        raise ValueError(f"Unsupported objective: {experiment_config.objective}")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id.")
    return MathConsistencyTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=CompletionOnlyDataCollator(pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
        objective=CausalLanguageModelingObjective(),
        callbacks=[
            EpochIntervalCallback(
                validation_every_epochs=(training_arguments.validation_every_epochs),
                log_every_epochs=training_arguments.log_every_epochs,
            ),
            *(callbacks or []),
        ],
    )
