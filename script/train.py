"""Fine-tune one configured LoRA experiment and evaluate generated answers."""

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
from src.data.language.dataset import prepare_tokenized_dataset
from src.data.loading import load_frozen_gsm8k_split
from src.evaluation.diagnostics import select_evaluation_metrics
from src.evaluation.runners import evaluate_model
from src.model.factory import build_language_model
from src.model.tokenizer import load_tokenizer
from src.tracking import (
    WANDB_MODES,
    finish_wandb_run,
    initialize_wandb_run,
    log_wandb_metrics,
)
from src.training.factory import (
    build_training_arguments,
    build_training_trainer,
)
from src.training.objective import ABLATIONS, compile_experiment
from src.training.periodic_evaluation import (
    PERIODIC_EVAL_SAMPLES_PER_SPLIT,
    PERIODIC_EVAL_SEED,
    PeriodicGenerationEvaluationCallback,
    load_fixed_periodic_evaluation_datasets,
    periodic_evaluation_metadata,
)
from src.training.trainer import release_training_memory

# Command-line contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a declarative LoRA reasoning experiment."
    )
    parser.add_argument(
        "--ablation",
        choices=tuple(ABLATIONS),
        default="A1",
        help="Training recipe; A0 and A8 belong to evaluation.",
    )
    parser.add_argument("--math-token-weight", type=positive_float, default=2.0)
    parser.add_argument("--model-name", default=CONFIG.model_name)
    parser.add_argument(
        "--lora-r",
        type=positive_int,
        default=8,
        help="LoRA rank; lora_alpha is set automatically to 2 * rank.",
    )
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
    parser.add_argument(
        "--num-train-epochs",
        type=positive_float,
        default=CONFIG.num_train_epochs,
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional optimizer-step limit; overrides num-train-epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=positive_int,
        default=CONFIG.train_batch_size,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=positive_int,
        default=CONFIG.eval_batch_size,
    )
    parser.add_argument(
        "--generation-batch-size",
        type=positive_int,
        default=CONFIG.generation_batch_size,
    )
    parser.add_argument(
        "--periodic-eval-batch-size",
        type=positive_int,
        default=CONFIG.periodic_eval_batch_size,
        help="Batch size for fixed periodic generation evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=CONFIG.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=positive_float, default=1e-4)
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


# Report helpers


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep scalar Trainer metrics suitable for a JSON report."""
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            cleaned[key] = None
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
    return cleaned


# Training workflow


def main() -> None:
    args = parse_args()
    disable_progress_bars()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA GPU is available.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment = compile_experiment(
        args.ablation,
        math_token_weight=args.math_token_weight,
        require_implemented=True,
    )
    wandb_run_name = args.wandb_run_name or experiment["id"]
    wandb_run = initialize_wandb_run(
        project=args.wandb_project,
        run_name=wandb_run_name,
        job_type="training",
        mode=args.wandb_mode,
        config={
            "model_name": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": 2 * args.lora_r,
            "wandb_run_name": wandb_run_name,
            "experiment": experiment,
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
            "drop_incomplete_train_batch": CONFIG.drop_incomplete_train_batch,
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
        feature_columns=experiment["features"],
        math_token_weight=experiment["math_token_weight"],
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
    model = build_language_model(
        model_name=args.model_name,
        head_names=frozenset(experiment["heads"]),
        lora_r=args.lora_r,
    )
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
        run_name=wandb_run_name,
        report_to_wandb=wandb_run is not None,
        seed=args.seed,
    )
    optimization_configuration = {
        "lr_scheduler_type": training_arguments.lr_scheduler_type.value,
        "lr_scheduler_kwargs": training_arguments.lr_scheduler_kwargs,
        "warmup_ratio": training_arguments.warmup_ratio,
        "early_stopping_patience": (training_arguments.early_stopping_patience),
        "early_stopping_threshold": (training_arguments.early_stopping_threshold),
    }
    if wandb_run is not None:
        wandb_run.config.update(
            {"optimization": optimization_configuration},
            allow_val_change=True,
        )
    periodic_evaluation_callback = PeriodicGenerationEvaluationCallback(
        datasets=periodic_evaluation_datasets,
        tokenizer=tokenizer,
        eval_every=args.eval_every,
        batch_size=args.periodic_eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        wandb_run=wandb_run,
    )
    trainer = build_training_trainer(
        model=model,
        experiment=experiment,
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
    report = {
        "model": args.model_name,
        "method": f"{experiment['id']} LoRA SFT",
        "experiment": experiment,
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
            "lora_r": args.lora_r,
            "lora_alpha": 2 * args.lora_r,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "periodic_eval_batch_size": args.periodic_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "drop_incomplete_train_batch": CONFIG.drop_incomplete_train_batch,
            "learning_rate": args.learning_rate,
            **optimization_configuration,
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
        experiment_name=experiment["id"],
        metrics={
            "train_final": report["train_metrics"],
            "validation_final": report["validation_metrics"],
            "generation_evaluation": select_evaluation_metrics(
                report["generation_evaluation"]
            ),
        },
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Adapter and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
