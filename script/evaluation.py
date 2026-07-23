"""Run GSM8K evaluation for a pretrained model from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SEED, MODEL_NAME
from src.evaluation import evaluate_pretrained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained causal language model on GSM8K."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail immediately when no CUDA GPU is visible.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/baseline_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable. Check the NVIDIA driver, "
            "NVIDIA Container Toolkit, and Docker --gpus all."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"Evaluation device: {device}" + (f" ({gpu_name})" if gpu_name else ""))

    results = evaluate_pretrained_model(
        args.model_name,
        split=args.split,
        subset_size=args.subset_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    report = {
        "model": args.model_name,
        "split": args.split,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
        "gpu_name": gpu_name,
        "exact_match": results["exact_match"],
        "correct": results["correct"],
        "valid_predictions": results["valid_predictions"],
        "valid_prediction_rate": results["valid_prediction_rate"],
        "format_compliance_rate": results["format_compliance_rate"],
        "elapsed_seconds": results["elapsed_seconds"],
        "samples_per_second": results["samples_per_second"],
        "total": results["total"],
        "predictions": results["predictions"],
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "predictions"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Full results saved to {args.output_file}")


if __name__ == "__main__":
    main()
