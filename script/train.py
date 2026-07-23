"""Fine-tune Qwen with LoRA, validate it, and evaluate exact match."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SEED, MODEL_NAME
from src.evaluation import evaluate_model
from src.load_data import load_gsm8k_dataset
from src.preprocess_data import prepare_gsm8k_dataset
from src.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-1.5B-Instruct on GSM8K with LoRA."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen2.5-1.5b-gsm8k"),
    )
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-subset-size", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_gsm8k_dataset(
        validation_size=args.validation_size,
        train_subset_size=args.train_subset_size,
        test_subset_size=1,
        seed=args.seed,
    )
    trainer = build_trainer(
        dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    trainer.model.print_trainable_parameters()

    train_result = trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    trainer.processing_class.save_pretrained(str(args.output_dir))

    trainer.model.config.use_cache = True
    test_dataset = load_gsm8k_dataset(
        split="test",
        subset_size=args.test_subset_size,
        seed=args.seed,
    )
    test_results = evaluate_model(
        trainer.model,
        trainer.processing_class,
        test_dataset,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    report = {
        "model": args.model_name,
        "method": "LoRA SFT",
        "seed": args.seed,
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "test_examples": len(test_dataset),
        "hyperparameters": {
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "validation_size": args.validation_size,
        },
        "train_metrics": _json_safe(train_result.metrics),
        "validation_metrics": _json_safe(validation_metrics),
        "test_exact_match": test_results["exact_match"],
        "test_correct": test_results["correct"],
        "test_total": test_results["total"],
        "valid_prediction_rate": test_results["valid_prediction_rate"],
        "format_compliance_rate": test_results["format_compliance_rate"],
        "predictions": test_results["predictions"],
    }
    report_path = args.output_dir / "fine_tuned_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "predictions"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Adapter and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
