"""Tests for causal structured features and their dynamic collation."""

from __future__ import annotations

import json
from typing import Any

from src.data.features import (
    POSTFIX_PROGRAM,
    STEP_FEATURE_COLUMNS,
    STEP_MASK,
    STEP_POSITIONS,
    STEP_TARGET_SCALES,
    STEP_TARGETS,
    TOKEN_LOSS_WEIGHTS,
)
from src.data.language.collator import collate_completion_only
from src.data.language.tokenization import tokenize_training_example


class CharacterTokenizer:
    """Tokenizer double with exact one-character offsets."""

    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        **_: object,
    ) -> str:
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text: str, **options: Any) -> dict[str, Any]:
        length = min(len(text), int(options.get("max_length", len(text))))
        encoding: dict[str, Any] = {
            "input_ids": list(range(length)),
            "attention_mask": [1] * length,
        }
        if options.get("return_offsets_mapping"):
            encoding["offset_mapping"] = [(index, index + 1) for index in range(length)]
        return encoding


def _example(question: str, answer: str) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def _prompt_length(tokenizer: CharacterTokenizer, example: dict[str, Any]) -> int:
    return len(
        tokenizer.apply_chat_template(
            example["prompt"],
            add_generation_prompt=True,
        )
    )


def test_step_positions_are_strictly_causal() -> None:
    tokenizer = CharacterTokenizer()
    answer = "First, <<6-2=4>>.\nThen, <<4+1=5>>.\n#### 5"
    example = _example("There are 6 objects, then 2 leave. Add one.", answer)

    row = tokenize_training_example(
        tokenizer,  # type: ignore[arg-type]
        example,
        max_length=4096,
        feature_columns=STEP_FEATURE_COLUMNS,
    )
    answer_offset = _prompt_length(tokenizer, example)
    second_line_start = answer.index("Then")
    assert row[STEP_POSITIONS] == [
        answer_offset - 1,
        answer_offset + second_line_start - 1,
    ]
    assert row[STEP_TARGETS] == [4.0, 5.0]
    assert row[STEP_TARGET_SCALES] == [4.0, 5.0]
    assert row[STEP_MASK] == [True, True]

    program = json.loads(row[POSTFIX_PROGRAM])
    assert len(program["steps"]) == 2
    assert program["steps"][1]["is_final"] is True


def test_fraction_targets_keep_the_normalized_regression_scale() -> None:
    tokenizer = CharacterTokenizer()
    example = _example(
        "One item is divided into 3 equal parts.",
        "One part is <<1/3=1/3>>.\n#### 1/3",
    )

    row = tokenize_training_example(
        tokenizer,  # type: ignore[arg-type]
        example,
        max_length=4096,
        feature_columns=STEP_FEATURE_COLUMNS,
    )

    assert row[STEP_TARGETS] == [1 / 3]
    assert row[STEP_TARGET_SCALES] == [1.0]
    assert row[STEP_MASK] == [True]


def test_truncation_masks_only_targets_that_are_not_fully_visible() -> None:
    tokenizer = CharacterTokenizer()
    answer = "Compute <<2+3=5>>.\n#### 5"
    example = _example("What is 2 + 3?", answer)
    answer_offset = _prompt_length(tokenizer, example)
    cut_inside_annotation = answer_offset + answer.index(">>") + 1

    row = tokenize_training_example(
        tokenizer,  # type: ignore[arg-type]
        example,
        max_length=cut_inside_annotation,
        feature_columns=STEP_FEATURE_COLUMNS,
    )

    assert row[STEP_MASK] == [False]


def test_math_weights_cover_annotations_but_never_prompt_tokens() -> None:
    tokenizer = CharacterTokenizer()
    answer = "Compute <<2+3=5>>.\n#### 5"
    example = _example("What is 2 + 3?", answer)
    answer_offset = _prompt_length(tokenizer, example)

    row = tokenize_training_example(
        tokenizer,  # type: ignore[arg-type]
        example,
        max_length=4096,
        feature_columns={TOKEN_LOSS_WEIGHTS},
        math_token_weight=3.5,
    )
    annotation_start = answer_offset + answer.index("<<")
    annotation_end = answer_offset + answer.index(">>") + 2

    assert row[TOKEN_LOSS_WEIGHTS][:answer_offset] == [0.0] * answer_offset
    assert set(row[TOKEN_LOSS_WEIGHTS][annotation_start:annotation_end]) == {3.5}
    assert row[TOKEN_LOSS_WEIGHTS][annotation_end] == 1.0


def test_collator_pads_tokens_and_steps_and_decodes_programs() -> None:
    tokenizer = CharacterTokenizer()
    columns = STEP_FEATURE_COLUMNS | {TOKEN_LOSS_WEIGHTS}
    rows = [
        tokenize_training_example(
            tokenizer,  # type: ignore[arg-type]
            _example(
                "There are 6 objects, then 2 leave. Add one.",
                ("First <<6-2=4>>.\nThen <<4+1=5>>.\n#### 5"),
            ),
            max_length=4096,
            feature_columns=columns,
            math_token_weight=2.0,
        ),
        tokenize_training_example(
            tokenizer,  # type: ignore[arg-type]
            _example("What is 2 + 3?", "Compute <<2+3=5>>.\n#### 5"),
            max_length=4096,
            feature_columns=columns,
            math_token_weight=2.0,
        ),
    ]

    batch = collate_completion_only(
        rows,
        pad_token_id=99,
        pad_to_multiple_of=8,
    )

    assert tuple(batch[STEP_POSITIONS].shape) == (2, 2)
    assert batch[STEP_MASK].tolist() == [[True, True], [True, False]]
    assert batch[STEP_TARGET_SCALES][1, 1].item() == 1.0
    assert batch[TOKEN_LOSS_WEIGHTS][1, -1].item() == 0.0
    assert len(batch[POSTFIX_PROGRAM]) == 2
    assert isinstance(batch[POSTFIX_PROGRAM][0], dict)
