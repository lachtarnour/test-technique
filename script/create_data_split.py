"""Create the frozen GSM8K train/validation/test dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.data.splits import create_frozen_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the official GSM8K train into train/validation and preserve test."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(CONFIG.dataset_path),
    )
    parser.add_argument("--validation-size", type=float, default=CONFIG.validation_size)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_frozen_split(
        args.output_dir,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Frozen split saved to {args.output_dir}")


if __name__ == "__main__":
    main()
