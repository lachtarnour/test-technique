"""Load and validate the tokenizer used by Qwen2.5-1.5B-Instruct."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import MODEL_NAME, SYSTEM_PROMPT


Message = dict[str, str]


def validate_qwen_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Ensure that the loaded tokenizer contains Qwen's chat configuration."""
    if not tokenizer.chat_template:
        raise ValueError("The tokenizer does not provide a chat template.")

    expected_special_tokens = {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }
    for attribute, expected_value in expected_special_tokens.items():
        actual_value = getattr(tokenizer, attribute)
        if actual_value != expected_value:
            raise ValueError(
                f"Unexpected {attribute}: {actual_value!r}; "
                f"expected {expected_value!r} for {MODEL_NAME}."
            )

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    required_markers = ("<|im_start|>user", "<|im_end|>", "<|im_start|>assistant")
    missing_markers = [marker for marker in required_markers if marker not in rendered]
    if missing_markers:
        raise ValueError(
            "The tokenizer chat template is not compatible with Qwen: "
            f"missing {missing_markers}."
        )


def load_tokenizer(
    model_name: str = MODEL_NAME,
    *,
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    """Load the official tokenizer and verify its Qwen chat template."""
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be either 'left' or 'right'.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side=padding_side,
    )
    validate_qwen_tokenizer(tokenizer)
    return tokenizer


def tokenize_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Message],
    *,
    add_generation_prompt: bool = False,
    max_length: int | None = None,
    return_tensors: str | None = None,
) -> dict[str, Any]:
    """Tokenize a conversation with Qwen's official chat template."""
    if not messages:
        raise ValueError("messages must contain at least one message.")

    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        truncation=max_length is not None,
        max_length=max_length,
        return_dict=True,
        return_tensors=return_tensors,
    )
    return dict(encoded)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tokenizer smoke test."""
    parser = argparse.ArgumentParser(
        description="Load and verify the Qwen2.5-1.5B-Instruct tokenizer."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    """Run a small local encoding/decoding verification."""
    args = parse_args()
    tokenizer = load_tokenizer(args.model_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    encoded = tokenize_messages(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"]
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Number of tokens: {len(input_ids)}")
    print(tokenizer.decode(input_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
