"""Explicit completion-only tokenization for the controlled A1 baseline."""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_training_example(
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one conversation and mask every prompt token in ``labels``."""
    if max_length <= 0:
        raise ValueError("max_length must be strictly positive.")

    prompt = example["prompt"]
    completion = example["completion"]
    if len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError("completion must contain exactly one assistant message.")

    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [*prompt, *completion],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("The chat template does not expose a stable prompt prefix.")

    full_encoding = dict(
        tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
    )
    prompt_encoding = dict(
        tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
    )
    input_ids = [int(token_id) for token_id in full_encoding["input_ids"]]
    prompt_ids = [int(token_id) for token_id in prompt_encoding["input_ids"]]
    prompt_length = min(len(prompt_ids), len(input_ids))
    if input_ids[:prompt_length] != prompt_ids[:prompt_length]:
        raise ValueError("Prompt tokenization is not a prefix of the full example.")

    attention_mask = [
        int(value)
        for value in full_encoding.get("attention_mask", [1] * len(input_ids))
    ]
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError(
            "max_length truncates the entire completion; increase it or "
            "shorten the prompt."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset_split(
    dataset: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize one normalized split and remove transient conversation text."""
    return dataset.map(
        lambda example: tokenize_training_example(
            tokenizer,
            example,
            max_length=max_length,
        ),
        remove_columns=dataset.column_names,
        keep_in_memory=True,
        desc="Tokenizing GSM8K for A1-control",
    )
