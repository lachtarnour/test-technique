"""Audit V1 graph construction and exact execution on frozen GSM8K data."""

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
from src.data.graph_audit import audit_graph_rows
from src.load_data import load_frozen_gsm8k_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and execute conservative V1 calculation graphs in parallel."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(CONFIG.dataset_path),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--sample-size", type=positive_int, default=1000)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_frozen_gsm8k_split(
        args.split,
        dataset_path=args.data_dir,
        subset_size=args.sample_size,
        seed=args.seed,
    )
    rows = [
        {
            "question": str(row["question"]),
            "answer": str(row["answer"]),
        }
        for row in dataset
    ]
    report = {
        "configuration": {
            "data_dir": str(args.data_dir),
            "split": args.split,
            "sample_size": len(rows),
            "workers": args.workers,
            "seed": args.seed,
        },
        **audit_graph_rows(rows, workers=args.workers),
    }
    output_file = args.output_file or Path(
        f"outputs/math_graph_audit_{args.split}_{len(rows)}"
        f"_seed{args.seed}_workers{args.workers}.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Full graph audit saved to {output_file}")


if __name__ == "__main__":
    main()
