"""Reusable argument validators for the command-line entry points."""

from __future__ import annotations

import argparse
import math


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for ``argparse``."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a strictly positive integer")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point value for ``argparse``."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a strictly positive number")
    return parsed
