"""Run GSM8K evaluation for a pretrained model from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_pretrained_model

DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained causal language model on GSM8K."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        help="Number of examples to evaluate (default: 10).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_pretrained_model(
        args.model_name,
        split=args.split,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    summary = {
        "model": args.model_name,
        "split": args.split,
        "rmse": results["rmse"],
        "accuracy": results["accuracy"],
        "valid_predictions": results["valid_predictions"],
        "valid_prediction_rate": results["valid_prediction_rate"],
        "total": results["total"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
