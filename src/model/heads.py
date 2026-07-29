"""Reusable auxiliary heads registered as part of the trained model."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from src.data.graph.postfix_schemas import PostfixOperator

AUXILIARY_MODULE_PREFIX = "auxiliary_"
NUMERIC_RESULT_HEAD = "numeric_result"
ACTION_OPERATOR_HEAD = "action_operator"
OPERAND_REFERENCE_HEAD = "operand_reference"
COMPOSITION_HEAD = "composition"
OPERATOR_LABELS = tuple(
    dict.fromkeys(operator.supervision_operator.value for operator in PostfixOperator)
)
# Unresolved operand positions are masked and therefore have no training class.
OPERAND_KIND_LABELS = (
    "problem_number",
    "previous_result",
    "local_result",
    "literal",
)


class NumericResultHead(nn.Module):
    """Predict one scalar from a causal hidden state."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(self.normalization(hidden_states)).squeeze(-1)


class ActionOperatorHead(nn.Module):
    """Classify one ordered postfix operator."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, len(OPERATOR_LABELS))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(self.normalization(hidden_states))


class OperandReferenceHead(nn.Module):
    """Predict operand kind, candidate pointer and literal value."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size)
        self.kind_projection = nn.Linear(hidden_size, len(OPERAND_KIND_LABELS))
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.literal_projection = nn.Linear(hidden_size, 1)
        self.pointer_scale = hidden_size**-0.5

    def forward(
        self,
        query_states: torch.Tensor,
        candidate_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score ordered operand occurrences against causal candidates."""
        normalized = self.normalization(query_states)
        queries = self.query_projection(normalized)
        candidates = self.candidate_projection(candidate_states)
        pointer_logits = (
            torch.einsum("...h,...ch->...c", queries, candidates) * self.pointer_scale
        )
        return {
            "kind_logits": self.kind_projection(normalized),
            "pointer_logits": pointer_logits,
            "literal_values": self.literal_projection(normalized).squeeze(-1),
        }


class CompositionHead(nn.Module):
    """Compose an ordered unary or binary action into one hidden state."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.operator_embedding = nn.Embedding(len(OPERATOR_LABELS), hidden_size)
        self.missing_second_operand = nn.Parameter(torch.zeros(hidden_size))
        self.composer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        operator_ids: torch.Tensor,
        first_operand_states: torch.Tensor,
        second_operand_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if second_operand_states is None:
            second_operand_states = self.missing_second_operand.expand_as(
                first_operand_states
            )
        operator_states = self.operator_embedding(operator_ids)
        return self.composer(
            torch.cat(
                (
                    operator_states,
                    first_operand_states,
                    second_operand_states,
                ),
                dim=-1,
            )
        )


def auxiliary_module_name(head_name: str) -> str:
    """Return the stable checkpoint module name for one logical head."""
    return f"{AUXILIARY_MODULE_PREFIX}{head_name}"


def build_auxiliary_heads(
    head_names: frozenset[str],
    *,
    hidden_size: int,
) -> dict[str, nn.Module]:
    """Instantiate only the modules required by the experiment plan."""
    if hidden_size <= 0:
        raise ValueError("hidden_size must be strictly positive.")
    factories = {
        NUMERIC_RESULT_HEAD: NumericResultHead,
        ACTION_OPERATOR_HEAD: ActionOperatorHead,
        OPERAND_REFERENCE_HEAD: OperandReferenceHead,
        COMPOSITION_HEAD: CompositionHead,
    }
    unknown = head_names - set(factories)
    if unknown:
        raise ValueError(f"Unknown auxiliary heads: {sorted(unknown)}")
    return {name: factories[name](hidden_size) for name in sorted(head_names)}


def get_auxiliary_heads(
    model: nn.Module,
    head_names: frozenset[str],
) -> Mapping[str, nn.Module]:
    """Resolve registered heads through a plain or PEFT-wrapped model."""
    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    resolved: dict[str, nn.Module] = {}
    for name in sorted(head_names):
        module_name = auxiliary_module_name(name)
        try:
            resolved[name] = getattr(base_model, module_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Model is missing required auxiliary head {name!r}."
            ) from exc
    return resolved
