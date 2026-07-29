"""Load and validate the shared Qwen tokenizer.

Training-specific rendering, masking, offsets, and truncation live in
``src.data.language.tokenization``. This module remains deliberately small because the
frozen evaluation pipeline also uses the same tokenizer factory.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import CONFIG


def validate_qwen_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_name: str = CONFIG.model_name,
) -> None:
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
                f"expected {expected_value!r} for {model_name}."
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
    model_name: str = CONFIG.model_name,
    *,
    padding_side: str = "right",
    revision: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load the official tokenizer and verify its Qwen chat template."""
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be either 'left' or 'right'.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side=padding_side,
        revision=revision,
    )
    validate_qwen_tokenizer(tokenizer, model_name=model_name)
    return tokenizer
