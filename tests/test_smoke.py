"""Minimal public smoke tests for the training and evaluation pipeline."""

from types import SimpleNamespace
from typing import Any

import torch

import src.tokenizer as tokenizer_module
from src.config import CONFIG
from src.data.formatting import format_training_example
from src.evaluation import extract_final_answer, parse_annotated_formulas
from src.model import factory


def test_model_forward(monkeypatch: Any) -> None:
    class TinyCausalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            batch_size, sequence_length = input_ids.shape
            return SimpleNamespace(logits=torch.zeros(batch_size, sequence_length, 8))

    backbone = TinyCausalModel()
    monkeypatch.setattr(
        factory.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: backbone,
    )
    monkeypatch.setattr(factory, "get_peft_model", lambda model, _config: model)

    model = factory.build_language_model(
        model_name="tiny-test-model",
        model_loading_kwargs={"dtype": torch.float32},
    )
    outputs = model(input_ids=torch.tensor([[1, 2, 3]]))

    assert outputs.logits.shape == (1, 3, 8)
    assert model.config.use_cache is False


def test_tokenizer_loader(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    fake_tokenizer = object()

    def fake_from_pretrained(model_name: str, **options: Any) -> object:
        captured.update(model_name=model_name, **options)
        return fake_tokenizer

    monkeypatch.setattr(
        tokenizer_module.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        tokenizer_module,
        "validate_qwen_tokenizer",
        lambda *_args, **_kwargs: None,
    )

    loaded = tokenizer_module.load_tokenizer(
        "tiny-test-tokenizer",
        padding_side="left",
        revision="frozen-revision",
    )

    assert loaded is fake_tokenizer
    assert captured == {
        "model_name": "tiny-test-tokenizer",
        "use_fast": True,
        "padding_side": "left",
        "revision": "frozen-revision",
    }


def test_final_answer_parsing() -> None:
    assert extract_final_answer("Reasoning: 6 × 7 = 42.\n#### 42") == "42"

    accepted_approximations = (
        "<<1/3=.3>>",
        "<<1/3=0.33>>",
        "<<1/3=0.333>>",
        "<<6/7=0.857142857>>",
        "<<6/7=0.86>>",
    )
    assert all(
        parse_annotated_formulas(annotation)[0].is_correct
        for annotation in accepted_approximations
    )
    assert not parse_annotated_formulas("<<6/7=0.87>>")[0].is_correct


def test_mock_question_uses_the_frozen_prompt() -> None:
    formatted = format_training_example(
        {
            "question": "  What is 2 + 3? ",
            "answer": "Add <<2+3=5>>5.\n#### 5",
        }
    )

    assert formatted["prompt"] == [
        {"role": "system", "content": CONFIG.system_prompt},
        {"role": "user", "content": "What is 2 + 3?"},
    ]
    assert formatted["completion"] == [
        {"role": "assistant", "content": "Add <<2+3=5>>5.\n#### 5"}
    ]
