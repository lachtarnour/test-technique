"""Exact parsing and normalization of final numeric answers."""

from __future__ import annotations

import re
from decimal import Decimal, DecimalException
from fractions import Fraction

NUMERIC_VALUE_PATTERN = r"-?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
FRACTION_VALUE_PATTERN = rf"{NUMERIC_VALUE_PATTERN}\s*/\s*{NUMERIC_VALUE_PATTERN}"
NUMERIC_OR_FRACTION_PATTERN = rf"(?:{FRACTION_VALUE_PATTERN}|{NUMERIC_VALUE_PATTERN})"

NUMBER_PATTERN = re.compile(NUMERIC_VALUE_PATTERN)
NUMERIC_OR_FRACTION_FULLMATCH = re.compile(NUMERIC_OR_FRACTION_PATTERN)

REPEATING_DECIMAL_PLACES = 10
MAX_SYMMETRIC_RELATIVE_ERROR = Fraction(2)


def _rounded_decimal(value: Fraction, *, places: int) -> str:
    """Render a fraction at a fixed precision using exact half-up rounding."""
    scale = 10**places
    scaled_numerator = abs(value.numerator) * scale
    quotient, remainder = divmod(scaled_numerator, value.denominator)
    if remainder * 2 >= value.denominator:
        quotient += 1

    sign = "-" if value < 0 else ""
    digits = str(quotient).rjust(places + 1, "0")
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def normalize_fraction(value: Fraction) -> str:
    """Return an exact finite decimal or a stable repeating approximation."""
    if value.denominator == 1:
        return str(value.numerator)

    denominator = value.denominator
    powers_of_two = 0
    powers_of_five = 0
    while denominator % 2 == 0:
        denominator //= 2
        powers_of_two += 1
    while denominator % 5 == 0:
        denominator //= 5
        powers_of_five += 1

    if denominator != 1:
        return _rounded_decimal(value, places=REPEATING_DECIMAL_PLACES)

    scale = max(powers_of_two, powers_of_five)
    multiplier = (2 ** (scale - powers_of_two)) * (5 ** (scale - powers_of_five))
    scaled_numerator = value.numerator * multiplier
    sign = "-" if scaled_numerator < 0 else ""
    digits = str(abs(scaled_numerator)).rjust(scale + 1, "0")
    integer_part = digits[:-scale]
    fractional_part = digits[-scale:].rstrip("0")
    if not fractional_part:
        return f"{sign}{integer_part}"
    return f"{sign}{integer_part}.{fractional_part}"


def parse_numeric_fraction(value: str) -> Fraction | None:
    """Parse one decimal or ``numerator/denominator`` value exactly."""
    candidate = value.strip()
    if NUMERIC_OR_FRACTION_FULLMATCH.fullmatch(candidate) is None:
        return None

    parts = candidate.split("/")
    try:
        if len(parts) == 1:
            number = Decimal(parts[0].replace(",", ""))
            return Fraction(number) if number.is_finite() else None
        if len(parts) != 2:
            return None

        numerator = Decimal(parts[0].strip().replace(",", ""))
        denominator = Decimal(parts[1].strip().replace(",", ""))
        if not numerator.is_finite() or not denominator.is_finite() or denominator == 0:
            return None
        return Fraction(numerator) / Fraction(denominator)
    except (DecimalException, ZeroDivisionError):
        return None


def normalize_numeric_value(value: str) -> str | None:
    """Normalize a decimal or fraction without floating-point rounding."""
    parsed = parse_numeric_fraction(value)
    return normalize_fraction(parsed) if parsed is not None else None


def _decimal_places(value: str) -> int | None:
    """Return the declared decimal precision, excluding explicit fractions."""
    candidate = value.strip()
    if "/" in candidate or NUMBER_PATTERN.fullmatch(candidate) is None:
        return None
    candidate = candidate.replace(",", "")
    return len(candidate.rsplit(".", 1)[1]) if "." in candidate else 0


def _decimal_approximations(value: Fraction, *, places: int) -> set[Fraction]:
    """Return exact truncation and half-up rounding at ``places`` decimals."""
    scale = 10**places
    scaled_numerator = abs(value.numerator) * scale
    quotient, remainder = divmod(scaled_numerator, value.denominator)

    truncated_numerator = -quotient if value < 0 else quotient
    rounded_quotient = quotient + int(remainder * 2 >= value.denominator)
    rounded_numerator = -rounded_quotient if value < 0 else rounded_quotient
    return {
        Fraction(truncated_numerator, scale),
        Fraction(rounded_numerator, scale),
    }


def numeric_prediction_matches_fraction(
    prediction: str,
    target: Fraction,
) -> bool:
    """Accept an exact value or its stated decimal rounding/truncation."""
    predicted_value = parse_numeric_fraction(prediction)
    if predicted_value is None:
        return False
    if predicted_value == target:
        return True

    places = _decimal_places(prediction)
    if places is None or places < 1:
        return False
    return predicted_value in _decimal_approximations(target, places=places)


def numeric_values_equal(prediction: str, target: str) -> bool:
    """Compare a prediction to an exact target at the prediction's precision."""
    target_value = parse_numeric_fraction(target)
    if target_value is None:
        return False
    return numeric_prediction_matches_fraction(prediction, target_value)


def numeric_symmetric_relative_error(
    prediction: str,
    target: str,
) -> Fraction | None:
    """Return ``2*|prediction-target|/(|prediction|+|target|)`` exactly."""
    predicted_value = parse_numeric_fraction(prediction)
    target_value = parse_numeric_fraction(target)
    if predicted_value is None or target_value is None:
        return None

    denominator = abs(predicted_value) + abs(target_value)
    if denominator == 0:
        return Fraction(0)
    return 2 * abs(predicted_value - target_value) / denominator
