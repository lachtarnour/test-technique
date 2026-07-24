"""Answer extraction, model generation and model-loading helpers."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.config import CONFIG
from src.device import cuda_supports_native_bf16
from src.evaluation.numeric import (
    NUMERIC_OR_FRACTION_PATTERN,
    normalize_numeric_value,
)

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any

VALUE_BOUNDARY = r"(?![\w/-]|\.\d)"
TERMINAL_ANSWER_PATTERN = re.compile(rf"####\s*({NUMERIC_OR_FRACTION_PATTERN})\s*\Z")
STANDALONE_VALUE_PATTERN = re.compile(
    rf"(?<![\w.-])({NUMERIC_OR_FRACTION_PATTERN}){VALUE_BOUNDARY}"
)
PROGRESS_HEARTBEAT_SECONDS = 30.0


def _print_evaluation_progress(message: str) -> None:
    """Write one immediately visible line to local and remote job logs."""
    print(f"[evaluation] {message}", flush=True)


@contextmanager
def _generation_heartbeat(
    *,
    batch_index: int,
    total_batches: int,
    completed_examples: int,
    total_examples: int,
):
    """Report progress while the blocking model.generate call is running."""
    stop_event = threading.Event()
    batch_started_at = time.perf_counter()

    def report_until_stopped() -> None:
        while not stop_event.wait(PROGRESS_HEARTBEAT_SECONDS):
            elapsed = time.perf_counter() - batch_started_at
            percentage = 100 * completed_examples / total_examples
            _print_evaluation_progress(
                f"batch {batch_index}/{total_batches} running"
                f" | completed={completed_examples}/{total_examples}"
                f" ({percentage:.1f}%)"
                f" | batch_elapsed={elapsed:.1f}s"
            )

    reporter = threading.Thread(
        target=report_until_stopped,
        name="evaluation-progress",
        daemon=True,
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join()


def extract_final_answer(text: str, *, require_marker: bool = True) -> str | None:
    """Extract a terminal marked answer or the last standalone numeric value."""
    terminal_match = TERMINAL_ANSWER_PATTERN.search(text)
    if terminal_match is not None:
        return normalize_numeric_value(terminal_match.group(1))
    if require_marker:
        return None

    matches = STANDALONE_VALUE_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_numeric_value(matches[-1])


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


def _model_loading_options(
    torch: Any,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build consistent loading options for base models and adapters."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if cuda_supports_native_bf16(torch) else torch.float16
    else:
        dtype = "auto"

    options = {"dtype": dtype, "device_map": "auto"}
    options.update(overrides or {})
    return options


def generate_model_responses(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    batch_size: int = 4,
    max_new_tokens: int = CONFIG.max_new_tokens,
    generation_kwargs: Mapping[str, Any] | None = None,
    system_prompt: str = CONFIG.system_prompt,
) -> dict[str, Any]:
    """Generate responses and normalize GSM8K references."""
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be strictly positive.")
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    missing_columns = CONFIG.required_columns - set(dataset.column_names)
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
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
        **(generation_kwargs or {}),
    }
    predictions: list[dict[str, Any]] = []

    started_at = time.perf_counter()
    starts = range(0, len(dataset), batch_size)
    total_examples = len(dataset)
    total_batches = len(starts)
    _print_evaluation_progress(
        f"started | completed=0/{total_examples} (0.0%)"
        f" | batches={total_batches} | batch_size={batch_size}"
    )

    for batch_index, start in enumerate(starts, start=1):
        stop = min(start + batch_size, total_examples)
        completed_percentage = 100 * start / total_examples
        _print_evaluation_progress(
            f"batch {batch_index}/{total_batches} started"
            f" | examples={start + 1}-{stop}/{total_examples}"
            f" | completed={start}/{total_examples} ({completed_percentage:.1f}%)"
        )
        batch_started_at = time.perf_counter()
        batch = dataset[start : start + batch_size]
        conversations = [
            [
                {"role": "system", "content": system_prompt},
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

        import torch

        with (
            _generation_heartbeat(
                batch_index=batch_index,
                total_batches=total_batches,
                completed_examples=start,
                total_examples=total_examples,
            ),
            torch.inference_mode(),
        ):
            generated = model.generate(**encoded, **options)

        prompt_length = encoded["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )

        for question, reference_text, generated_text in zip(
            batch["question"],
            batch["answer"],
            generated_texts,
            strict=True,
        ):
            reference = extract_final_answer(reference_text)
            if reference is None:
                raise ValueError(
                    "A reference answer does not contain a valid #### marker: "
                    f"{reference_text!r}"
                )
            predictions.append(
                {
                    "question": question,
                    "reference": reference,
                    "generated_text": generated_text,
                }
            )

        completed_examples = len(predictions)
        elapsed_seconds = time.perf_counter() - started_at
        batch_elapsed_seconds = time.perf_counter() - batch_started_at
        samples_per_second = completed_examples / elapsed_seconds
        remaining_examples = total_examples - completed_examples
        eta_seconds = (
            remaining_examples / samples_per_second if remaining_examples else 0.0
        )
        completed_percentage = 100 * completed_examples / total_examples
        _print_evaluation_progress(
            f"batch {batch_index}/{total_batches} completed"
            f" | completed={completed_examples}/{total_examples}"
            f" ({completed_percentage:.1f}%)"
            f" | batch_time={batch_elapsed_seconds:.1f}s"
            f" | elapsed={elapsed_seconds:.1f}s"
            f" | speed={samples_per_second:.2f} examples/s"
            f" | eta={eta_seconds:.1f}s"
        )

    total = len(predictions)
    elapsed_seconds = time.perf_counter() - started_at
    samples_per_second = total / elapsed_seconds
    _print_evaluation_progress(
        f"completed | completed={total}/{total_examples} (100.0%)"
        f" | elapsed={elapsed_seconds:.1f}s"
        f" | speed={samples_per_second:.2f} examples/s"
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": samples_per_second,
        "predictions": predictions,
    }


def generate_pretrained_responses(
    model_name: str,
    *,
    dataset: Dataset | None = None,
    split: str = CONFIG.evaluation_split,
    subset_size: int | None = None,
    seed: int = CONFIG.seed,
    dataset_path: str = CONFIG.dataset_path,
    model_kwargs: Mapping[str, Any] | None = None,
    **generation_options: Any,
) -> dict[str, Any]:
    """Load a pretrained model and generate evaluation responses."""
    import torch
    from transformers import AutoModelForCausalLM

    from src.load_data import load_frozen_gsm8k_split, load_gsm8k_dataset
    from src.tokenizer import load_tokenizer

    revision = (model_kwargs or {}).get("revision")
    tokenizer = load_tokenizer(
        model_name,
        padding_side="left",
        revision=revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_model_loading_options(torch, model_kwargs),
    )
    if dataset is not None:
        evaluation_dataset = dataset
    elif split in {"train", "validation", "test"}:
        evaluation_dataset = load_frozen_gsm8k_split(
            split,
            dataset_path=dataset_path,
            subset_size=subset_size,
            seed=seed,
        )
    else:
        evaluation_dataset = load_gsm8k_dataset(
            split=split,
            subset_size=subset_size,
            seed=seed,
        )
    return generate_model_responses(
        model,
        tokenizer,
        evaluation_dataset,
        **generation_options,
    )


def generate_checkpoint_responses(
    checkpoint_path: str,
    *,
    dataset: Dataset | None = None,
    split: str = CONFIG.evaluation_split,
    subset_size: int | None = None,
    seed: int = CONFIG.seed,
    dataset_path: str = CONFIG.dataset_path,
    tokenizer_path: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    **generation_options: Any,
) -> dict[str, Any]:
    """Load a LoRA checkpoint and generate evaluation responses."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    from src.load_data import load_frozen_gsm8k_split, load_gsm8k_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or checkpoint_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        **_model_loading_options(torch, model_kwargs),
    )
    if dataset is not None:
        evaluation_dataset = dataset
    elif split in {"train", "validation", "test"}:
        evaluation_dataset = load_frozen_gsm8k_split(
            split,
            dataset_path=dataset_path,
            subset_size=subset_size,
            seed=seed,
        )
    else:
        evaluation_dataset = load_gsm8k_dataset(
            split=split,
            subset_size=subset_size,
            seed=seed,
        )
    return generate_model_responses(
        model,
        tokenizer,
        evaluation_dataset,
        **generation_options,
    )
