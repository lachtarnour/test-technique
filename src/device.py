"""CUDA capability helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Any


def cuda_supports_native_bf16(torch: Any) -> bool:
    """Return whether CUDA can execute BF16 natively, without emulation."""
    if not torch.cuda.is_available():
        return False

    try:
        return bool(
            torch.cuda.is_bf16_supported(
                including_emulation=False,
            )
        )
    except TypeError:
        properties = torch.cuda.get_device_properties(
            torch.cuda.current_device(),
        )
        return bool(torch.cuda.is_bf16_supported() and properties.major >= 8)


def training_model_dtype(torch: Any) -> Any:
    """Select a model dtype that the active training device supports natively."""
    if not torch.cuda.is_available():
        return torch.float32
    if cuda_supports_native_bf16(torch):
        return torch.bfloat16
    return torch.float16
