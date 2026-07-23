"""Create the TRL SFTTrainer used to fine-tune Qwen on GSM8K."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from src.config import MODEL_NAME
from src.tokenizer import load_tokenizer


def build_trainer(
    dataset: DatasetDict,
    *,
    model_name: str = MODEL_NAME,
    output_dir: str | Path = "outputs/qwen2.5-1.5b-gsm8k",
    max_length: int = 1024,
    num_train_epochs: float = 1.0,
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> SFTTrainer:
    """Return an SFTTrainer configured to learn only from completions."""
    tokenizer = load_tokenizer(model_name, padding_side="right")
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_length,
        completion_only_loss=True,
        loss_type="nll",
        model_init_kwargs={"dtype": "auto"},
        num_train_epochs=num_train_epochs,
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
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        seed=seed,
        data_seed=seed,
        dataloader_pin_memory=use_cuda,
        report_to="none",
    )

    return SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
