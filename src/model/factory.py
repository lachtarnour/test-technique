"""Build one LoRA language model with experiment-specific auxiliary heads."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from src.config import CONFIG
from src.model.device import training_model_dtype
from src.model.heads import auxiliary_module_name, build_auxiliary_heads


def _model_hidden_size(config: Any) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("The model config does not expose a valid hidden size.")


def build_language_model(
    *,
    model_name: str = CONFIG.model_name,
    model_loading_kwargs: dict[str, Any] | None = None,
    head_names: frozenset[str] = frozenset(),
) -> PeftModel:
    """Load the backbone, register heads, then attach LoRA safely."""
    loading_kwargs = dict(model_loading_kwargs or {})
    loading_kwargs.setdefault("dtype", training_model_dtype(torch))
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **loading_kwargs,
    )
    language_model.config.use_cache = False
    auxiliary_heads = (
        build_auxiliary_heads(
            head_names,
            hidden_size=_model_hidden_size(language_model.config),
        )
        if head_names
        else {}
    )
    auxiliary_module_names: list[str] = []
    for head_name, head in auxiliary_heads.items():
        module_name = auxiliary_module_name(head_name)
        language_model.add_module(module_name, head)
        auxiliary_module_names.append(module_name)

    return get_peft_model(
        language_model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            exclude_modules=auxiliary_module_names or None,
            modules_to_save=auxiliary_module_names or None,
            task_type="CAUSAL_LM",
        ),
    )
