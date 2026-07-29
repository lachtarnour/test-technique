"""Verify that sequential and parallel graph audits are byte-stable per example."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import positive_int
from src.config import CONFIG
from src.data.graph.audit import audit_graph_rows
from src.data.loading import load_frozen_gsm8k_split


def _stable_example_hash(example: dict[str, Any]) -> str:
    stable_payload = {
        key: value for key, value in example.items() if key != "worker_pid"
    }
    serialized = json.dumps(
        stable_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _audit_hashes(
    rows: list[dict[str, str]],
    *,
    workers: int,
) -> list[str]:
    report = audit_graph_rows(rows, workers=workers)
    hashes = [_stable_example_hash(example) for example in report["examples"]]
    del report
    gc.collect()
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compare complete sequential and parallel graph-audit outputs.")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(CONFIG.dataset_path),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument("--workers", type=positive_int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, dict[str, int | list[int]]] = {}
    for split in args.splits:
        dataset = load_frozen_gsm8k_split(
            split,
            dataset_path=args.data_dir,
        )
        rows = [
            {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in dataset
        ]
        sequential_hashes = _audit_hashes(rows, workers=1)
        parallel_hashes = _audit_hashes(rows, workers=args.workers)
        mismatches = [
            index
            for index, (sequential_hash, parallel_hash) in enumerate(
                zip(
                    sequential_hashes,
                    parallel_hashes,
                    strict=True,
                )
            )
            if sequential_hash != parallel_hash
        ]
        results[split] = {
            "examples": len(rows),
            "matching_examples": len(rows) - len(mismatches),
            "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:10],
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    mismatch_count = sum(int(result["mismatch_count"]) for result in results.values())
    if mismatch_count:
        raise RuntimeError(f"Audit determinism failed for {mismatch_count} examples.")


if __name__ == "__main__":
    main()
