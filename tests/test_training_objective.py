"""Tests for the functional objective and its thin Trainer adapter."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from datasets import Dataset
from transformers import TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

import src.training.trainer as trainer_module
from src.data.features import TOKEN_LOSS_WEIGHTS
from src.data.language.collator import collate_completion_only
from src.model.heads import (
    NUMERIC_RESULT_HEAD,
    NumericResultHead,
    auxiliary_module_name,
)
from src.training.objective import (
    LOSSES,
    compile_experiment,
    compute_objective,
    count_language_tokens,
    normalization_counts,
)
from src.training.trainer import MathConsistencyTrainer


def test_a1_is_standard_completion_only_cross_entropy() -> None:
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3]])

    loss, statistics = compute_objective(
        {"language": 1.0},
        model_outputs={"logits": logits},
        batch={"labels": labels},
    )
    expected = functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    assert torch.allclose(loss, expected)
    numerator, denominator = statistics["language_loss"]
    assert torch.allclose(numerator / denominator, expected)


def test_a1_normalizes_an_accumulation_window_by_all_tokens() -> None:
    first_logits = torch.randn(1, 4, 7, requires_grad=True)
    second_logits = torch.randn(1, 6, 7, requires_grad=True)
    first_labels = torch.tensor([[-100, -100, -100, 3]])
    second_labels = torch.tensor([[-100, -100, 2, 3, 4, 5]])
    losses = []
    expected_sums = []

    for logits, labels in (
        (first_logits, first_labels),
        (second_logits, second_labels),
    ):
        losses.append(
            compute_objective(
                {"language": 1.0},
                model_outputs={"logits": logits},
                batch={"labels": labels},
                normalizers={"language": 5},
            )[0]
        )
        expected_sums.append(
            functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )

    assert torch.allclose(sum(losses), sum(expected_sums) / 5)


def test_a2_is_cross_entropy_weighted_by_the_exact_token_weight_sum() -> None:
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3]])
    weights = torch.tensor([[0.0, 0.0, 3.0, 1.0]])

    loss, statistics = compute_objective(
        {"language:math": 1.0},
        model_outputs={"logits": logits},
        batch={"labels": labels, TOKEN_LOSS_WEIGHTS: weights},
    )
    token_losses = functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(1, -1)
    expected = (token_losses * weights[:, 1:]).sum() / weights[:, 1:].sum()

    numerator, denominator = statistics["language_loss"]
    assert torch.allclose(loss, expected)
    assert torch.allclose(numerator / denominator, expected)
    assert denominator.item() == pytest.approx(4.0)
    assert count_language_tokens(
        {"labels": labels, TOKEN_LOSS_WEIGHTS: weights}
    ).item() == pytest.approx(4.0)


def test_a2_uses_one_weighted_normalizer_for_the_accumulation_window() -> None:
    batches = [
        {
            "labels": torch.tensor([[-100, -100, 3]]),
            TOKEN_LOSS_WEIGHTS: torch.tensor([[0.0, 0.0, 4.0]]),
            "logits": torch.randn(1, 3, 7, requires_grad=True),
        },
        {
            "labels": torch.tensor([[-100, 2, 3]]),
            TOKEN_LOSS_WEIGHTS: torch.tensor([[0.0, 1.0, 2.0]]),
            "logits": torch.randn(1, 3, 7, requires_grad=True),
        },
    ]
    normalizer = normalization_counts(
        {"language:math": 1.0},
        [
            {
                "labels": batch["labels"],
                TOKEN_LOSS_WEIGHTS: batch[TOKEN_LOSS_WEIGHTS],
            }
            for batch in batches
        ],
    )["language:math"]

    losses = []
    numerators = []
    for batch in batches:
        loss, statistics = compute_objective(
            {"language:math": 1.0},
            model_outputs={"logits": batch["logits"]},
            batch=batch,
            normalizers={"language:math": normalizer},
        )
        losses.append(loss)
        numerators.append(statistics["language_loss"][0])

    assert normalizer.item() == pytest.approx(7.0)
    assert torch.allclose(sum(losses), sum(numerators) / normalizer)


def test_objective_applies_coefficients_without_weighting_statistics(
    monkeypatch,
) -> None:
    def constant(value: float):
        def compute(*args: object):
            del args
            tensor = torch.tensor(value)
            return tensor, (tensor * 3, torch.tensor(3.0))

        return compute

    monkeypatch.setitem(
        LOSSES,
        "first",
        (1.0, frozenset(), frozenset(), constant(2.0), lambda batch: 1),
    )
    monkeypatch.setitem(
        LOSSES,
        "second",
        (1.0, frozenset(), frozenset(), constant(4.0), lambda batch: 1),
    )

    total, statistics = compute_objective(
        {"first": 1.0, "second": 0.25},
        model_outputs={},
        batch={},
    )

    assert total.item() == pytest.approx(3.0)
    numerator, denominator = statistics["second_loss"]
    assert numerator.item() == pytest.approx(12.0)
    assert denominator.item() == pytest.approx(3.0)


def test_each_loss_has_its_own_window_normalizer(monkeypatch) -> None:
    def count_results(batch: dict[str, object]) -> torch.Tensor:
        mask = batch["result_mask"]
        assert isinstance(mask, torch.Tensor)
        return mask.sum()

    monkeypatch.setitem(
        LOSSES,
        "result_test",
        (1.0, frozenset(), frozenset(), None, count_results),
    )
    batches = [
        {"labels": [[-100, 1]], "result_mask": torch.tensor([1.0, 0.0])},
        {"labels": [[-100, 2, 3, 4]], "result_mask": torch.tensor([1.0] * 3)},
    ]

    counts = normalization_counts(
        {"language": 1.0, "result_test": 1.0},
        batches,
    )

    assert counts["language"] == 4
    assert int(counts["result_test"]) == 4


def test_common_trainer_uses_the_functional_a1_objective() -> None:
    class TinyLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 5)
            self.projection = torch.nn.Linear(5, 8)

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> CausalLMOutput:
            del attention_mask
            return CausalLMOutput(logits=self.projection(self.embedding(input_ids)))

    trainer = object.__new__(MathConsistencyTrainer)
    trainer.losses = {"language": 1.0}
    trainer.state = SimpleNamespace(global_step=0)
    model = TinyLanguageModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3]]),
    }

    loss = trainer.compute_loss(model, inputs)

    assert loss.ndim == 0
    loss.backward()
    assert model.projection.weight.grad is not None


def test_trainer_exposes_hidden_states_and_dynamic_heads(monkeypatch) -> None:
    experiment = compile_experiment("A3")
    captured: dict[str, object] = {}

    class TinyStructuredModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 7)
            self.add_module(
                auxiliary_module_name(NUMERIC_RESULT_HEAD),
                NumericResultHead(hidden_size=4),
            )

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.projection

        def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            del attention_mask
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            return SimpleNamespace(logits=self.projection(hidden))

    def capture_objective(
        losses: object,
        *,
        model_outputs: dict[str, torch.Tensor],
        batch: object,
        normalizers: object,
        auxiliary_heads: dict[str, torch.nn.Module],
    ):
        del losses, batch, normalizers
        captured["hidden_shape"] = tuple(model_outputs["last_hidden_state"].shape)
        captured["head"] = auxiliary_heads[NUMERIC_RESULT_HEAD]
        loss = next(captured["head"].parameters()).sum() * 0
        return loss, {}

    monkeypatch.setattr(trainer_module, "compute_objective", capture_objective)
    trainer = object.__new__(MathConsistencyTrainer)
    trainer.losses = experiment["losses"]
    trainer.head_names = frozenset(experiment["heads"])
    trainer.needs_last_hidden_state = True
    trainer.state = SimpleNamespace(global_step=0)
    model = TinyStructuredModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3]]),
    }

    trainer.compute_loss(model, inputs)

    assert captured["hidden_shape"] == (1, 3, 4)
    assert captured["head"] is getattr(
        model,
        auxiliary_module_name(NUMERIC_RESULT_HEAD),
    )


def test_eval_loss_is_one_global_mean_over_completion_tokens(tmp_path) -> None:
    class DeterministicLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 7)
            self.projection = torch.nn.Linear(7, 16)

        def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            del attention_mask
            return CausalLMOutput(logits=self.projection(self.embedding(input_ids)))

    rows = {
        "input_ids": [[1, 2, 3, 4], [1, 5, 6, 7, 8, 9], [1, 10, 11]],
        "attention_mask": [[1] * 4, [1] * 6, [1] * 3],
        "labels": [
            [-100, -100, -100, 4],
            [-100, -100, 6, 7, 8, 9],
            [-100, 10, 11],
        ],
    }
    dataset = Dataset.from_dict(rows)
    torch.manual_seed(7)
    model = DeterministicLanguageModel()
    trainer = MathConsistencyTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            prediction_loss_only=True,
            report_to="none",
        ),
        eval_dataset=dataset,
        data_collator=partial(
            collate_completion_only,
            pad_token_id=0,
            pad_to_multiple_of=None,
        ),
        losses={"language": 1.0},
    )

    loss_sums = []
    token_count = 0
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for row in dataset:
            input_ids = torch.tensor([row["input_ids"]], device=model_device)
            labels = torch.tensor([row["labels"]], device=model_device)
            logits = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            ).logits
            loss_sums.append(
                functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            )
            token_count += labels[:, 1:].ne(-100).sum().item()

    metrics = trainer.evaluate()

    expected = sum(loss_sums).item() / token_count
    assert metrics["eval_loss"] == pytest.approx(expected, rel=1e-6)
    assert metrics["eval_language_loss"] == pytest.approx(expected, rel=1e-6)
