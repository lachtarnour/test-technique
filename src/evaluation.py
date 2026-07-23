"""Evaluate pretrained and fine-tuned causal language models on GSM8K."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.config import DEFAULT_EVALUATION_SPLIT, DEFAULT_SEED, SYSTEM_PROMPT

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any

NUMERIC_VALUE_PATTERN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
ANSWER_PATTERN = re.compile(rf"####\s*({NUMERIC_VALUE_PATTERN})")
NUMBER_PATTERN = re.compile(NUMERIC_VALUE_PATTERN)


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
        return None
    return str(int(number)) if number.is_integer() else str(number)


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
    max_new_tokens: int = 512,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure exact match on final numerical GSM8K answers."""
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
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        **(generation_kwargs or {}),
    }
    predictions: list[dict[str, Any]] = []
    correct = 0
    valid_predictions = 0
    format_compliant_predictions = 0

    from tqdm.auto import tqdm

    started_at = time.perf_counter()
    starts = range(0, len(dataset), batch_size)
    for start in tqdm(starts, desc="Evaluating GSM8K", unit="batch"):
        batch = dataset[start : start + batch_size]
        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            for question in batch["question"]
        ]
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
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
            marked_prediction = extract_final_answer(generated_text)
            prediction = marked_prediction or extract_final_answer(
                generated_text,
                require_marker=False,
            )
            reference = extract_final_answer(reference_text)
            is_correct = prediction is not None and prediction == reference
            valid_predictions += int(prediction is not None)
            format_compliant_predictions += int(marked_prediction is not None)
            correct += int(is_correct)
            predictions.append(
                {
                    "question": question,
                    "prediction": prediction,
                    "reference": reference,
                    "generated_text": generated_text,
                    "correct": is_correct,
                    "format_compliant": marked_prediction is not None,
                }
            )

    total = len(predictions)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "exact_match": correct / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "valid_predictions": valid_predictions,
        "valid_prediction_rate": valid_predictions / total,
        "format_compliant_predictions": format_compliant_predictions,
        "format_compliance_rate": format_compliant_predictions / total,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": total / elapsed_seconds,
        "predictions": predictions,
    }


def evaluate_pretrained_model(
    model_name: str,
    *,
    dataset: Dataset | None = None,
    split: str = DEFAULT_EVALUATION_SPLIT,
    subset_size: int | None = None,
    seed: int = DEFAULT_SEED,
    model_kwargs: Mapping[str, Any] | None = None,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Load and evaluate a pretrained Hugging Face causal language model."""
    import torch
    from transformers import AutoModelForCausalLM
    from src.load_data import load_gsm8k_dataset
    from src.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(model_name, padding_side="left")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else "auto"
    )
    loading_options = {"dtype": dtype, "device_map": "auto"}
    loading_options.update(model_kwargs or {})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **loading_options,
    )
    evaluation_dataset = (
        dataset
        if dataset is not None
        else load_gsm8k_dataset(
            split=split,
            subset_size=subset_size,
            seed=seed,
        )
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
    seed: int = DEFAULT_SEED,
    tokenizer_path: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Load and evaluate a saved LoRA adapter checkpoint."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from src.load_data import load_gsm8k_dataset

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or checkpoint_path
    )
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else "auto"
    )
    loading_options = {"dtype": dtype, "device_map": "auto"}
    loading_options.update(model_kwargs or {})
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        **loading_options,
    )
    evaluation_dataset = (
        dataset
        if dataset is not None
        else load_gsm8k_dataset(
            split=split,
            subset_size=subset_size,
            seed=seed,
        )
    )
    return evaluate_model(
        model, tokenizer, evaluation_dataset, **evaluation_kwargs
    )
