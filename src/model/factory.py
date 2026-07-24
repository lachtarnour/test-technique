"""Build the LoRA causal language model used by A1."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from src.config import CONFIG
from src.device import training_model_dtype


def build_language_model(
    *,
    model_name: str = CONFIG.model_name,
    model_loading_kwargs: dict[str, Any] | None = None,
) -> PeftModel:
    """Load the backbone and attach the LoRA configuration."""
    loading_kwargs = dict(model_loading_kwargs or {})
    loading_kwargs.setdefault("dtype", training_model_dtype(torch))
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **loading_kwargs,
    )
    language_model.config.use_cache = False
    return get_peft_model(
        language_model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
