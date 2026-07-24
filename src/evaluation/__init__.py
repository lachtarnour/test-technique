"""Public API for GSM8K arithmetic-consistency evaluation."""

from src.evaluation.arithmetic import (
    FormulaAnalysis,
    FormulaParseError,
    parse_annotated_formulas,
)
from src.evaluation.generation import extract_final_answer
from src.evaluation.numeric import numeric_symmetric_relative_error
from src.evaluation.reasoning import (
    aggregate_metrics,
    analyze_response,
)
from src.evaluation.runners import (
    evaluate_checkpoint,
    evaluate_model,
    evaluate_pretrained_model,
)

__all__ = [
    "FormulaAnalysis",
    "FormulaParseError",
    "aggregate_metrics",
    "analyze_response",
    "evaluate_checkpoint",
    "evaluate_model",
    "evaluate_pretrained_model",
    "extract_final_answer",
    "numeric_symmetric_relative_error",
    "parse_annotated_formulas",
]
