"""Fine-tune the A1 LoRA baseline and evaluate generated answers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import disable_progress_bars

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import positive_float, positive_int
from src.config import CONFIG
from src.data.dataset import prepare_tokenized_dataset
from src.evaluation import evaluate_model
from src.evaluation.diagnostics import select_evaluation_metrics
from src.load_data import load_frozen_gsm8k_split
from src.model.factory import build_language_model
from src.tokenizer import load_tokenizer
from src.tracking import (
    WANDB_MODES,
    finish_wandb_run,
    initialize_wandb_run,
    log_wandb_metrics,
)
from src.training.arguments import load_experiment_config
from src.training.factory import (
    build_training_arguments,
    build_training_trainer,
)
from src.training.periodic_evaluation import (
    PERIODIC_EVAL_BATCH_SIZE,
    PERIODIC_EVAL_SAMPLES_PER_SPLIT,
    PERIODIC_EVAL_SEED,
    PeriodicGenerationEvaluationCallback,
    load_fixed_periodic_evaluation_datasets,
    periodic_evaluation_metadata,
)
from src.training.trainer import release_training_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the completion-only A1 LoRA baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/a1_control.json"),
        help="A1 scientific experiment configuration.",
    )
    parser.add_argument("--model-name", default=CONFIG.model_name)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen2.5-1.5b-gsm8k-a1-control"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(CONFIG.dataset_path),
    )
    parser.add_argument("--train-subset-size", type=positive_int, default=None)
    parser.add_argument(
        "--validation-subset-size",
        type=positive_int,
        default=None,
        help="Optional validation subset, useful for smoke training.",
    )
    parser.add_argument(
        "--evaluation-subset-size",
        type=positive_int,
        default=None,
        help="Optional subset for post-training generation on validation.",
    )
    parser.add_argument("--max-length", type=positive_int, default=1024)
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=CONFIG.max_new_tokens,
    )
    parser.add_argument("--num-train-epochs", type=positive_float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional optimizer-step limit; overrides num-train-epochs.",
    )
    parser.add_argument("--train-batch-size", type=positive_int, default=24)
    parser.add_argument("--eval-batch-size", type=positive_int, default=16)
    parser.add_argument("--generation-batch-size", type=positive_int, default=300)
    parser.add_argument(
        "--periodic-eval-batch-size",
        type=positive_int,
        default=PERIODIC_EVAL_BATCH_SIZE,
        help="Batch size for fixed periodic generation evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=2,
    )
    parser.add_argument("--learning-rate", type=positive_float, default=2e-4)
    parser.add_argument(
        "--logging-steps",
        type=positive_int,
        default=10,
        help="Write structured training metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=positive_int,
        default=1,
        help="Run validation and save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--log-every-epochs",
        type=positive_int,
        default=1,
        help="Log training metrics to W&B every N epochs.",
    )
    parser.add_argument(
        "--eval-every",
        type=positive_int,
        default=2,
        help=(
            "Run generation metrics on fixed train/validation samples at "
            "epochs 0, N, 2N, ..."
        ),
    )
    parser.add_argument("--wandb-project", default="qwen-gsm8k")
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name; defaults to the experiment identifier.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=WANDB_MODES,
        default="online",
        help="Use offline to defer upload, or disabled to turn W&B off.",
    )
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail immediately unless a CUDA GPU is available.",
    )
    return parser.parse_args()


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep scalar Trainer metrics suitable for a JSON report."""
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            cleaned[key] = None
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
    return cleaned


def _cuda_memory_metrics() -> dict[str, int | float] | None:
    """Return exact process-level CUDA peaks for the complete training run."""
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    gibibyte = 1024**3
    return {
        "device_index": device,
        "total_bytes": total_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "total_gib": total_bytes / gibibyte,
        "peak_allocated_gib": peak_allocated_bytes / gibibyte,
        "peak_reserved_gib": peak_reserved_bytes / gibibyte,
        "peak_reserved_fraction": peak_reserved_bytes / total_bytes,
        "headroom_gib": (total_bytes - peak_reserved_bytes) / gibibyte,
    }


def main() -> None:
    args = parse_args()
    disable_progress_bars()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA GPU is available.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = load_experiment_config(args.config)
    wandb_run_name = args.wandb_run_name or experiment_config.experiment_id
    wandb_run = initialize_wandb_run(
        project=args.wandb_project,
        run_name=wandb_run_name,
        job_type="training",
        mode=args.wandb_mode,
        config={
            "model_name": args.model_name,
            "wandb_run_name": wandb_run_name,
            "experiment": experiment_config.to_dict(),
            "experiment_config_file": str(args.config),
            "data_dir": str(args.data_dir),
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "periodic_eval_batch_size": args.periodic_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "logging_steps": args.logging_steps,
            "validation_every_epochs": args.validation_every_epochs,
            "log_every_epochs": args.log_every_epochs,
            "eval_every": args.eval_every,
            "periodic_eval_samples_per_split": PERIODIC_EVAL_SAMPLES_PER_SPLIT,
            "periodic_eval_seed": PERIODIC_EVAL_SEED,
            "seed": args.seed,
            "require_cuda": args.require_cuda,
        },
    )
    tokenizer = load_tokenizer(
        args.model_name,
        padding_side="right",
    )
    dataset = prepare_tokenized_dataset(
        tokenizer=tokenizer,
        dataset_path=args.data_dir,
        max_length=args.max_length,
        train_subset_size=args.train_subset_size,
        validation_subset_size=args.validation_subset_size,
        seed=args.seed,
    )
    periodic_evaluation_datasets = load_fixed_periodic_evaluation_datasets(
        args.data_dir
    )
    periodic_metadata = periodic_evaluation_metadata(periodic_evaluation_datasets)
    periodic_configuration = {
        **periodic_metadata,
        "eval_every": args.eval_every,
        "batch_size": args.periodic_eval_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if wandb_run is not None:
        wandb_run.config.update(
            {"periodic_evaluation": periodic_configuration},
            allow_val_change=True,
        )
    model = build_language_model(model_name=args.model_name)
    model_dtype = str(next(model.parameters()).dtype)
    training_arguments = build_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        validation_every_epochs=args.validation_every_epochs,
        log_every_epochs=args.log_every_epochs,
        eval_every=args.eval_every,
        run_name=wandb_run_name,
        report_to_wandb=wandb_run is not None,
        seed=args.seed,
    )
    periodic_evaluation_callback = PeriodicGenerationEvaluationCallback(
        datasets=periodic_evaluation_datasets,
        tokenizer=tokenizer,
        eval_every=training_arguments.eval_every,
        batch_size=args.periodic_eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        wandb_run=wandb_run,
    )
    trainer = build_training_trainer(
        model=model,
        experiment_config=experiment_config,
        dataset=dataset,
        tokenizer=tokenizer,
        training_arguments=training_arguments,
        callbacks=[periodic_evaluation_callback],
    )
    model.print_trainable_parameters()

    train_result = trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    trainer.processing_class.save_pretrained(str(args.output_dir))

    release_training_memory(trainer)
    trainer.model.config.use_cache = True
    evaluation_dataset = load_frozen_gsm8k_split(
        "validation",
        dataset_path=args.data_dir,
        subset_size=args.evaluation_subset_size,
        seed=args.seed,
    )
    evaluation_results = evaluate_model(
        trainer.model,
        trainer.processing_class,
        evaluation_dataset,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        progress_context={
            "evaluation_kind": "post_training",
            "split": "validation",
        },
    )
    cuda_memory = _cuda_memory_metrics()

    report = {
        "model": args.model_name,
        "method": "A1-control LoRA SFT",
        "experiment": experiment_config.to_dict(),
        "experiment_config_file": str(args.config),
        "prompt": {
            "version": CONFIG.prompt_version,
            "sha256": CONFIG.system_prompt_sha256,
        },
        "seed": args.seed,
        "model_dtype": model_dtype,
        "data_dir": str(args.data_dir),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "generation_evaluation_split": "validation",
        "evaluation_examples": len(evaluation_dataset),
        "hyperparameters": {
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "periodic_eval_batch_size": args.periodic_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "logging_steps": args.logging_steps,
            "validation_every_epochs": args.validation_every_epochs,
            "log_every_epochs": args.log_every_epochs,
            "eval_every": args.eval_every,
            "validation_subset_size": args.validation_subset_size,
            "evaluation_subset_size": args.evaluation_subset_size,
        },
        "periodic_evaluation": {
            "configuration": periodic_configuration,
            "history": periodic_evaluation_callback.history,
        },
        "train_metrics": _json_safe(train_result.metrics),
        "validation_metrics": _json_safe(validation_metrics),
        "generation_evaluation": {
            key: value
            for key, value in evaluation_results.items()
            if key != "predictions"
        },
        "cuda_memory": cuda_memory,
        "predictions": evaluation_results["predictions"],
    }
    report_path = args.output_dir / "fine_tuned_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "predictions"}
    log_wandb_metrics(
        wandb_run,
        model_name=args.model_name,
        experiment_name=experiment_config.experiment_id,
        metrics={
            "train_final": report["train_metrics"],
            "validation_final": report["validation_metrics"],
            "generation_evaluation": select_evaluation_metrics(
                report["generation_evaluation"]
            ),
            "cuda_memory": report["cuda_memory"] or {},
        },
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Adapter and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
