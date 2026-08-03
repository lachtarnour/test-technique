"""Explicit completion-only tokenization for language-model training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.features import build_training_features


def tokenize_training_example(
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, Any],
    *,
    max_length: int,
    feature_columns: Iterable[str] = (),
    math_token_weight: float = 1.0,
) -> dict[str, Any]:
    """Tokenize one conversation and mask every prompt token in ``labels``."""
    if max_length <= 0:
        raise ValueError("max_length must be strictly positive.")

    prompt = example["prompt"]
    completion = example["completion"]
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError("prompt must end with exactly one user question.")
    if len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError("completion must contain exactly one assistant message.")
    question = prompt[-1].get("content")
    answer = completion[0].get("content")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise TypeError("Question and assistant completion content must be strings.")

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

    requested_features = frozenset(feature_columns)
    full_tokenization_options: dict[str, Any] = {
        "add_special_tokens": False,
        "truncation": True,
        "max_length": max_length,
    }
    if requested_features:
        full_tokenization_options["return_offsets_mapping"] = True
    full_encoding = dict(tokenizer(full_text, **full_tokenization_options))
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
    tokenized: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if requested_features:
        offsets = full_encoding.get("offset_mapping")
        if offsets is None:
            raise ValueError(
                "Structured features require a fast tokenizer with "
                "return_offsets_mapping support."
            )
        tokenized.update(
            build_training_features(
                question=question,
                answer=answer,
                input_ids=input_ids,
                labels=labels,
                offset_mapping=offsets,
                answer_offset=len(prompt_text),
                requested_columns=requested_features,
                math_token_weight=math_token_weight,
            )
        )
    return tokenized


def tokenize_dataset_split(
    dataset: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    feature_columns: Iterable[str] = (),
    math_token_weight: float = 1.0,
) -> Dataset:
    """Tokenize one normalized split and remove transient conversation text."""
    requested_features = frozenset(feature_columns)
    return dataset.map(
        lambda example: tokenize_training_example(
            tokenizer,
            example,
            max_length=max_length,
            feature_columns=requested_features,
            math_token_weight=math_token_weight,
        ),
        remove_columns=dataset.column_names,
        keep_in_memory=True,
        desc="Tokenizing GSM8K for training",
    )
