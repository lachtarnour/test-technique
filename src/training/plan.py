"""Single catalog that compiles experiment configuration into runtime parts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.model.heads import (
    ACTION_OPERATOR_HEAD,
    COMPOSITION_HEAD,
    NUMERIC_RESULT_HEAD,
    OPERAND_REFERENCE_HEAD,
)
from src.training.arguments import (
    COMPOSITION_LOSS,
    DEPENDENCY_LOSS,
    EXECUTION_LOSS,
    FINAL_ANSWER_LOSS,
    LANGUAGE_LOSS,
    MATH_WEIGHTED_LANGUAGE_LOSS,
    OPERATOR_LOSS,
    RESULT_LOSS,
    STANDARD_LANGUAGE_LOSS,
    STRUCTURED_ACTION_LOSS,
    ExperimentConfig,
    LossTermConfig,
)
from src.training.objective import (
    CompositeObjective,
    LossTerm,
    StandardLanguageLossTerm,
)

TOKEN_LOSS_WEIGHTS = "token_loss_weights"
STEP_POSITIONS = "step_positions"
STEP_TARGETS = "step_targets"
STEP_TARGET_SCALES = "step_target_scales"
STEP_MASK = "step_mask"
POSTFIX_PROGRAM = "postfix_program"
FINAL_ANSWER_POSITION = "final_answer_position"
FINAL_ANSWER_TARGET = "final_answer_target"
FINAL_ANSWER_SCALE = "final_answer_scale"
FINAL_ANSWER_MASK = "final_answer_mask"

BASE_MODEL_COLUMNS = frozenset({"input_ids", "attention_mask", "labels"})
LossFactory = Callable[[LossTermConfig], LossTerm]
LossKey = tuple[str, str | None]


@dataclass(frozen=True)
class LossDefinition:
    """Everything the infrastructure must know about one loss variant."""

    columns: frozenset[str] = frozenset()
    heads: frozenset[str] = frozenset()
    needs_last_hidden_state: bool = False
    factory: LossFactory | None = None


LOSS_CATALOG: dict[LossKey, LossDefinition] = {
    (LANGUAGE_LOSS, STANDARD_LANGUAGE_LOSS): LossDefinition(
        factory=StandardLanguageLossTerm,
    ),
    (LANGUAGE_LOSS, MATH_WEIGHTED_LANGUAGE_LOSS): LossDefinition(
        columns=frozenset({TOKEN_LOSS_WEIGHTS}),
    ),
    (RESULT_LOSS, None): LossDefinition(
        columns=frozenset(
            {STEP_POSITIONS, STEP_TARGETS, STEP_TARGET_SCALES, STEP_MASK}
        ),
        heads=frozenset({NUMERIC_RESULT_HEAD}),
        needs_last_hidden_state=True,
    ),
    (EXECUTION_LOSS, None): LossDefinition(
        columns=frozenset(
            {STEP_POSITIONS, STEP_TARGET_SCALES, STEP_MASK, POSTFIX_PROGRAM}
        ),
        heads=frozenset({NUMERIC_RESULT_HEAD}),
        needs_last_hidden_state=True,
    ),
    (OPERATOR_LOSS, None): LossDefinition(
        columns=frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        heads=frozenset({ACTION_OPERATOR_HEAD}),
        needs_last_hidden_state=True,
    ),
    (DEPENDENCY_LOSS, None): LossDefinition(
        columns=frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        heads=frozenset({OPERAND_REFERENCE_HEAD}),
        needs_last_hidden_state=True,
    ),
    (STRUCTURED_ACTION_LOSS, None): LossDefinition(
        columns=frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        heads=frozenset({ACTION_OPERATOR_HEAD, OPERAND_REFERENCE_HEAD}),
        needs_last_hidden_state=True,
    ),
    (FINAL_ANSWER_LOSS, None): LossDefinition(
        columns=frozenset(
            {
                FINAL_ANSWER_POSITION,
                FINAL_ANSWER_TARGET,
                FINAL_ANSWER_SCALE,
                FINAL_ANSWER_MASK,
            }
        ),
        heads=frozenset({NUMERIC_RESULT_HEAD}),
        needs_last_hidden_state=True,
    ),
    (COMPOSITION_LOSS, None): LossDefinition(
        columns=frozenset({STEP_POSITIONS, STEP_MASK, POSTFIX_PROGRAM}),
        heads=frozenset({COMPOSITION_HEAD}),
        needs_last_hidden_state=True,
    ),
}


@dataclass(frozen=True)
class ExperimentPlan:
    """Flat union of requirements for one experiment."""

    config: ExperimentConfig
    required_columns: frozenset[str]
    head_names: frozenset[str]
    needs_last_hidden_state: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "features": sorted(self.required_columns - BASE_MODEL_COLUMNS),
            "heads": sorted(self.head_names),
            "last_hidden_state": self.needs_last_hidden_state,
        }


def _loss_key(config: LossTermConfig) -> LossKey:
    return (
        config.name,
        config.mode if config.name == LANGUAGE_LOSS else None,
    )


def _definition(config: LossTermConfig) -> LossDefinition:
    try:
        return LOSS_CATALOG[_loss_key(config)]
    except KeyError as exc:
        raise ValueError(
            f"No loss definition registered for {config.name!r} (mode={config.mode!r})."
        ) from exc


def compile_experiment(config: ExperimentConfig) -> ExperimentPlan:
    """Merge requirements declared in the single loss catalog."""
    definitions = tuple(_definition(loss) for loss in config.losses)
    return ExperimentPlan(
        config=config,
        required_columns=BASE_MODEL_COLUMNS.union(
            *(definition.columns for definition in definitions)
        ),
        head_names=frozenset().union(*(definition.heads for definition in definitions)),
        needs_last_hidden_state=any(
            definition.needs_last_hidden_state for definition in definitions
        ),
    )


def build_objective(config: ExperimentConfig) -> CompositeObjective:
    """Instantiate configured loss terms from the same catalog."""
    terms: list[LossTerm] = []
    for loss in config.losses:
        definition = _definition(loss)
        if definition.factory is None:
            raise NotImplementedError(
                "Loss term is configured but not implemented yet: "
                f"{loss.name!r} (mode={loss.mode!r})."
            )
        terms.append(definition.factory(loss))
    return CompositeObjective(tuple(terms))
