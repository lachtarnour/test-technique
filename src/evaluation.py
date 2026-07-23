"""Evaluate pretrained and fine-tuned causal language models on GSM8K."""

from __future__ import annotations

import re
from collections.abc import Mapping
from math import sqrt
from typing import TYPE_CHECKING, Any

from src.config import DEFAULT_EVALUATION_SPLIT

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any

ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")
DEFAULT_PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning, then write the "
    "final answer as: #### <number>.\n\nQuestion: {question}\n\nAnswer:"
)


def extract_final_answer(
    text: str, *, require_marker: bool = True
) -> str | None:
    """Extract the last numerical answer following ``####``.

    When ``require_marker`` is true, a bare number is not accepted as a valid
    final answer. This is the format used by GSM8K references and evaluation.
    """
    matches = ANSWER_PATTERN.findall(text)
    if not matches and not require_marker:
        matches = NUMBER_PATTERN.findall(text)

    if not matches:
        return None

    answer = matches[-1].replace(",", "")
    try:
        number = float(answer)
    except ValueError:
        return answer
    return str(int(number)) if number.is_integer() else str(number)


def _absolute_difference(
    prediction: str | None, reference: str | None
) -> float | None:
    """Return the absolute numerical error, if both answers are valid."""
    if prediction is None or reference is None:
        return None
    try:
        return abs(float(prediction) - float(reference))
    except ValueError:
        return None


def _model_device(model: Any) -> Any:
    """Return the input device for a regular or device-mapped model."""
    if hasattr(model, "hf_device_map"):
        devices = [
            device
            for device in model.hf_device_map.values()
            if device not in {"cpu", "disk"}
        ]
        if devices:
            device = devices[0]
            return f"cuda:{device}" if isinstance(device, int) else device
    return getattr(model, "device", "cpu")


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an already loaded model.

    Keeping model loading outside this function makes it reusable for both the
    original pretrained model and every fine-tuning checkpoint.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be strictly positive.")
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    missing_columns = {"question", "answer"} - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a pad token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    device = _model_device(model)
    options = {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        **(generation_kwargs or {}),
    }
    predictions: list[dict[str, Any]] = []
    correct = 0
    squared_errors: list[float] = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        prompts = [
            prompt_template.format(question=question)
            for question in batch["question"]
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        # Import lazily so answer parsing can be used without installing torch.
        import torch

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                **options,
            )

        prompt_length = encoded["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )

        for question, reference_text, generated_text in zip(
            batch["question"], batch["answer"], generated_texts, strict=True
        ):
            prediction = extract_final_answer(generated_text)
            reference = extract_final_answer(reference_text)
            is_correct = prediction == reference and reference is not None
            absolute_difference = _absolute_difference(prediction, reference)
            squared_error = (
                absolute_difference**2
                if absolute_difference is not None
                else None
            )
            if squared_error is not None:
                squared_errors.append(squared_error)
            correct += int(is_correct)
            predictions.append(
                {
                    "question": question,
                    "prediction": prediction,
                    "reference": reference,
                    "generated_text": generated_text,
                    "correct": is_correct,
                    "absolute_difference": absolute_difference,
                    "squared_error": squared_error,
                }
            )

    total = len(predictions)
    valid_predictions = len(squared_errors)
    return {
        "rmse": (
            sqrt(sum(squared_errors) / valid_predictions)
            if valid_predictions
            else None
        ),
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "valid_predictions": valid_predictions,
        "valid_prediction_rate": valid_predictions / total,
        "predictions": predictions,
    }


def evaluate_pretrained_model(
    model_name: str,
    *,
    dataset: Dataset | None = None,
    split: str = DEFAULT_EVALUATION_SPLIT,
    subset_size: int | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Load and evaluate a pretrained Hugging Face causal language model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data import load_gsm8k_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **(model_kwargs or {})
    )
    evaluation_dataset = (
        dataset
        if dataset is not None
        else load_gsm8k_dataset(split=split, subset_size=subset_size)
    )
    return evaluate_model(
        model, tokenizer, evaluation_dataset, **evaluation_kwargs
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    dataset: Dataset | None = None,
    split: str = DEFAULT_EVALUATION_SPLIT,
    subset_size: int | None = None,
    tokenizer_path: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Load and evaluate a full Hugging Face fine-tuning checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data import load_gsm8k_dataset

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or checkpoint_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, **(model_kwargs or {})
    )
    evaluation_dataset = (
        dataset
        if dataset is not None
        else load_gsm8k_dataset(split=split, subset_size=subset_size)
    )
    return evaluate_model(
        model, tokenizer, evaluation_dataset, **evaluation_kwargs
    )
