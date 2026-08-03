"""Tests for the shared A1/A2 data pipeline."""

import pytest
from datasets import Dataset

from src.config import CONFIG
from src.data.language.dataset import _prepare_split, prepare_tokenized_dataset
from src.data.language.formatting import (
    canonicalize_calculation_annotations,
    format_training_example,
    remove_trivial_identity_annotations,
)


def test_training_prompt_contract_is_frozen() -> None:
    assert CONFIG.prompt_version == "2.4.0"
    assert (
        CONFIG.system_prompt_sha256
        == "1aae71f8cd37f1d32a0b74d5ef12fe209126aa68f78e698a636084c0f54f66f1"
    )
    assert (
        "Write every arithmetic calculation exactly once, inside one complete "
        "<<expression=result>> annotation." in CONFIG.system_prompt
    )
    assert (
        "Do not repeat the expression before its annotation or its numeric "
        "result after the annotation." in CONFIG.system_prompt
    )
    assert (
        "The last arithmetic annotation must calculate the same value "
        "as the final answer after ####." in CONFIG.system_prompt
    )


def test_formatting_builds_the_prompt_completion_contract() -> None:
    formatted = format_training_example(
        {
            "question": "  What is 2 + 3? ",
            "answer": " Add <<2+3=5>>5.\n#### 5 ",
        }
    )

    assert formatted == {
        "prompt": [
            {"role": "system", "content": CONFIG.system_prompt},
            {"role": "user", "content": "What is 2 + 3?"},
        ],
        "completion": [{"role": "assistant", "content": "Add <<2+3=5>>.\n#### 5"}],
    }


def test_remove_trivial_identity_preserves_the_visible_result() -> None:
    answer = "First <<14=14>>14 oz, then <<14/2=7>>7 oz.\n#### 7"

    assert remove_trivial_identity_annotations(answer) == (
        "First 14 oz, then <<14/2=7>>7 oz.\n#### 7"
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "Tamtam has 13 + 8 + 18 + 12 = <<13+8+18+12=51>>51 shells.\n#### 51",
            "Tamtam has <<13+8+18+12=51>> shells.\n#### 51",
        ),
        (
            "Five boxes cost 5 x $10 = $<<5*10=50>>50.\n#### 50",
            "Five boxes cost <<5*10=50>>.\n#### 50",
        ),
        (
            "Jimmy has 120 matches because 5 times 24 equals "
            "<<5*24=120>>120.\n#### 120",
            "<<5*24=120>>.\n#### 120",
        ),
        (
            "One part is <<1/3=1/3>>1/3 meter.\n#### 1/3",
            "One part is <<1/3=1/3>> meter.\n#### 1/3",
        ),
        (
            "She spent S + 30 + 46 = S + <<+30+46=76>>76.\n#### 76",
            "<<+30+46=76>>.\n#### 76",
        ),
        (
            "He has <<100/10=10>>10 $10 bills.\n#### 10",
            "He has <<100/10=10>> $10 bills.\n#### 10",
        ),
    ],
)
def test_canonicalization_keeps_each_calculation_once(
    answer: str,
    expected: str,
) -> None:
    assert canonicalize_calculation_annotations(answer) == expected
    assert canonicalize_calculation_annotations(expected) == expected


@pytest.mark.parametrize(
    "example",
    [
        {"question": " ", "answer": "Reasoning\n#### 1"},
        {"question": "Question", "answer": " "},
    ],
)
def test_formatting_rejects_empty_rows(example: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        format_training_example(example)


def test_training_data_never_loads_the_official_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_splits: list[str] = []
    raw = Dataset.from_dict(
        {
            "question": ["Question"],
            "answer": ["Reasoning\n#### 1"],
        }
    )

    def fake_load(split: str, **_: object) -> Dataset:
        loaded_splits.append(split)
        return raw

    monkeypatch.setattr(
        "src.data.language.dataset.load_frozen_gsm8k_split",
        fake_load,
    )
    monkeypatch.setattr(
        "src.data.language.dataset._prepare_split",
        lambda dataset, **_: dataset,
    )

    prepared = prepare_tokenized_dataset(tokenizer=object())  # type: ignore[arg-type]

    assert loaded_splits == ["train", "validation"]
    assert set(prepared) == {"train", "validation"}


def test_training_data_forwards_optional_subset_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_options: list[tuple[str, int | None, int | None]] = []
    raw = Dataset.from_dict(
        {
            "question": ["Question"],
            "answer": ["Reasoning\n#### 1"],
        }
    )

    def fake_load(
        split: str,
        *,
        subset_size: int | None = None,
        seed: int | None = None,
        **_: object,
    ) -> Dataset:
        loaded_options.append((split, subset_size, seed))
        return raw

    monkeypatch.setattr(
        "src.data.language.dataset.load_frozen_gsm8k_split",
        fake_load,
    )
    monkeypatch.setattr(
        "src.data.language.dataset._prepare_split",
        lambda dataset, **_: dataset,
    )

    prepare_tokenized_dataset(
        tokenizer=object(),  # type: ignore[arg-type]
        train_subset_size=24,
        validation_subset_size=8,
        seed=7,
    )

    assert loaded_options == [
        ("train", 24, 7),
        ("validation", 8, 7),
    ]


def test_training_formatting_stays_in_memory_on_read_only_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_options: dict[str, object] = {}

    class ReadOnlyDatasetDouble:
        column_names = ["question", "answer"]

        def map(self, _function, **kwargs):
            map_options.update(kwargs)
            return self

        def __len__(self) -> int:
            return 1

    dataset = ReadOnlyDatasetDouble()
    monkeypatch.setattr(
        "src.data.language.dataset.tokenize_dataset_split",
        lambda formatted, **_: formatted,
    )

    _prepare_split(
        dataset,  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        max_length=1024,
    )

    assert map_options["keep_in_memory"] is True
