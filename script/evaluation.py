"""Run the reasoning-aware GSM8K evaluation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import positive_int
from src.config import CONFIG
from src.evaluation.diagnostics import select_evaluation_metrics
from src.evaluation.runners import (
    evaluate_checkpoint,
    evaluate_pretrained_model,
)
from src.tracking import (
    WANDB_MODES,
    finish_wandb_run,
    initialize_wandb_run,
    log_wandb_metrics,
)

# Command-line contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final answers and annotated reasoning on GSM8K."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-name")
    source.add_argument("--checkpoint-path")
    parser.add_argument(
        "--experiment-name",
        help="W&B run name; defaults to the evaluated model or checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(CONFIG.dataset_path),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=CONFIG.evaluation_split,
    )
    parser.add_argument(
        "--subset-size",
        type=positive_int,
        default=None,
        help="Optional deterministic subset size; defaults to the full split.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=CONFIG.max_new_tokens,
    )
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail immediately when no CUDA GPU is visible.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/a0_pretrained_test.json"),
    )
    parser.add_argument("--wandb-project", default="qwen-gsm8k")
    parser.add_argument(
        "--wandb-mode",
        choices=WANDB_MODES,
        default="online",
        help="Use offline to defer upload, or disabled to turn W&B off.",
    )
    return parser.parse_args()


# Evaluation workflow


def main() -> None:
    args = parse_args()

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    common_options = {
        "split": args.split,
        "dataset_path": args.data_dir,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "generation_kwargs": {
            "do_sample": False,
            "num_beams": 1,
        },
    }

    model_source = args.checkpoint_path or args.model_name or CONFIG.model_name
    model_type = "lora_checkpoint" if args.checkpoint_path else "A0_pretrained"
    experiment_name = args.experiment_name or str(model_source)
    wandb_run = initialize_wandb_run(
        project=args.wandb_project,
        run_name=experiment_name,
        job_type="evaluation",
        mode=args.wandb_mode,
        config={
            "model_source": str(model_source),
            "model_type": model_type,
            "experiment_name": experiment_name,
            "split": args.split,
            "data_dir": str(args.data_dir),
            "subset_size": args.subset_size,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    if args.checkpoint_path:
        results = evaluate_checkpoint(
            args.checkpoint_path,
            **common_options,
        )
    else:
        results = evaluate_pretrained_model(
            model_source,
            **common_options,
        )

    report = {
        "model_source": model_source,
        "model_type": model_type,
        "experiment_name": experiment_name,
        "split": args.split,
        "data_dir": str(args.data_dir),
        "subset_size": args.subset_size,
        "evaluated_examples": results["total"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
        "gpu_name": gpu_name,
        **results,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {key: value for key, value in report.items() if key != "predictions"}
    log_wandb_metrics(
        wandb_run,
        model_name=str(model_source),
        experiment_name=experiment_name,
        metrics=select_evaluation_metrics(results),
        prefix="evaluation",
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Full results saved to {args.output_file}")


if __name__ == "__main__":
    main()
