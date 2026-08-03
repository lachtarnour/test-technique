"""Tests for epoch-based training logging and validation intervals."""

from types import SimpleNamespace

import pytest
from transformers import EarlyStoppingCallback, TrainerControl, TrainerState

import src.training.factory as training_factory
from src.config import CONFIG
from src.training.callbacks import EpochIntervalCallback
from src.training.factory import build_training_arguments
from src.training.trainer import (
    MathConsistencyTrainer,
    _current_cuda_memory_metrics,
    release_training_memory,
)


def test_default_batches_match_l40s_reference_run(
    tmp_path,
) -> None:
    arguments = build_training_arguments(
        output_dir=tmp_path,
        report_to_wandb=False,
    )

    assert arguments.per_device_train_batch_size == CONFIG.train_batch_size
    assert arguments.per_device_eval_batch_size == CONFIG.eval_batch_size
    assert arguments.gradient_accumulation_steps == CONFIG.gradient_accumulation_steps
    assert (
        arguments.per_device_train_batch_size * arguments.gradient_accumulation_steps
        == CONFIG.train_batch_size * CONFIG.gradient_accumulation_steps
    )
    assert arguments.dataloader_drop_last is True
    assert arguments.per_device_train_batch_size == 24
    assert arguments.gradient_accumulation_steps == 3
    assert arguments.per_device_eval_batch_size == 32
    assert arguments.per_device_train_batch_size * 3 == 72


def test_default_training_uses_plateau_reduction_and_early_stopping(
    tmp_path,
) -> None:
    arguments = build_training_arguments(
        output_dir=tmp_path,
        report_to_wandb=False,
    )

    assert arguments.num_train_epochs == 30.0
    assert arguments.learning_rate == pytest.approx(1e-4)
    assert arguments.lr_scheduler_type.value == "reduce_lr_on_plateau"
    assert arguments.lr_scheduler_kwargs == {
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "threshold": 0.005,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 1e-5,
    }
    assert arguments.warmup_ratio == 0.0
    assert arguments.metric_for_best_model == "eval_loss"
    assert arguments.greater_is_better is False
    assert arguments.early_stopping_patience == 6
    assert arguments.early_stopping_threshold == pytest.approx(1e-3)


def test_training_trainer_registers_the_early_stopping_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class Split:
        column_names = ["input_ids", "attention_mask", "labels"]

        def __len__(self) -> int:
            return 1

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.callbacks = kwargs["callbacks"]

        def remove_callback(self, callback: object) -> None:
            del callback

    monkeypatch.setattr(training_factory, "MathConsistencyTrainer", FakeTrainer)
    arguments = build_training_arguments(
        output_dir=tmp_path,
        report_to_wandb=False,
    )

    trainer = training_factory.build_training_trainer(
        model=object(),
        dataset={"train": Split(), "validation": Split()},
        tokenizer=SimpleNamespace(pad_token_id=0),
        training_arguments=arguments,
        experiment={"features": [], "losses": {"language": 1.0}, "heads": []},
    )

    early_stopping = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, EarlyStoppingCallback)
    ]
    assert len(early_stopping) == 1
    assert early_stopping[0].early_stopping_patience == 6
    assert early_stopping[0].early_stopping_threshold == pytest.approx(1e-3)


def test_validation_keeps_its_incomplete_final_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_drop_last: list[bool] = []
    trainer = object.__new__(MathConsistencyTrainer)
    trainer.args = SimpleNamespace(dataloader_drop_last=True)
    monkeypatch.setattr(
        "src.training.trainer.Trainer.get_eval_dataloader",
        lambda self, eval_dataset=None: observed_drop_last.append(
            self.args.dataloader_drop_last
        ),
    )

    trainer.get_eval_dataloader()

    assert observed_drop_last == [False]
    assert trainer.args.dataloader_drop_last is True


def test_training_arguments_expose_epoch_frequencies(tmp_path) -> None:
    arguments = build_training_arguments(
        output_dir=tmp_path,
        validation_every_epochs=3,
        log_every_epochs=2,
        max_steps=3,
        run_name="A1-control",
    )

    assert arguments.validation_every_epochs == 3
    assert arguments.log_every_epochs == 2
    assert arguments.max_steps == 3
    assert arguments.logging_strategy.value == "steps"
    assert arguments.logging_steps == 10
    assert arguments.disable_tqdm is True
    assert arguments.eval_strategy.value == "epoch"
    assert arguments.save_strategy.value == "epoch"
    assert arguments.report_to == ["wandb"]
    assert arguments.run_name == "A1-control"


def test_current_cuda_memory_metrics_are_reported_in_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: 3 * 1024**3,
        memory_reserved=lambda: 5 * 1024**3,
    )
    monkeypatch.setattr("src.training.trainer.torch.cuda", cuda)

    assert _current_cuda_memory_metrics() == {
        "gpu_memory_allocated_gib": 3.0,
        "gpu_memory_reserved_gib": 5.0,
    }


def test_trainer_injects_cuda_memory_only_into_training_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, float]] = []
    trainer = object.__new__(MathConsistencyTrainer)
    trainer._consume_component_metrics = lambda phase: {}  # type: ignore[method-assign]
    monkeypatch.setattr(
        "src.training.trainer._current_cuda_memory_metrics",
        lambda: {
            "gpu_memory_allocated_gib": 3.0,
            "gpu_memory_reserved_gib": 5.0,
        },
    )
    monkeypatch.setattr(
        "src.training.trainer.Trainer.log",
        lambda self, logs, start_time=None: captured.append(dict(logs)),
    )

    MathConsistencyTrainer.log(trainer, {"loss": 0.5})
    MathConsistencyTrainer.log(trainer, {"eval_loss": 0.4})

    assert captured[0]["gpu_memory_allocated_gib"] == 3.0
    assert captured[0]["gpu_memory_reserved_gib"] == 5.0
    assert "gpu_memory_allocated_gib" not in captured[1]
    assert "gpu_memory_reserved_gib" not in captured[1]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("validation_every_epochs", 0),
        ("validation_every_epochs", 1.5),
        ("log_every_epochs", 0),
        ("log_every_epochs", True),
        ("logging_steps", 0),
        ("logging_steps", 1.5),
        ("max_steps", 0),
        ("max_steps", 1.5),
    ],
)
def test_training_arguments_reject_invalid_epoch_frequencies(
    tmp_path,
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        build_training_arguments(output_dir=tmp_path, **{name: value})


def test_epoch_callback_applies_intervals_and_always_runs_at_end() -> None:
    callback = EpochIntervalCallback(
        validation_every_epochs=3,
        log_every_epochs=2,
    )

    first_epoch = callback.on_epoch_end(
        None,
        TrainerState(epoch=1.0, global_step=10, max_steps=40),
        TrainerControl(should_log=True, should_evaluate=True, should_save=True),
    )
    assert first_epoch.should_log is False
    assert first_epoch.should_evaluate is False
    assert first_epoch.should_save is False

    second_epoch = callback.on_epoch_end(
        None,
        TrainerState(epoch=2.0, global_step=20, max_steps=40),
        TrainerControl(should_log=True, should_evaluate=True, should_save=True),
    )
    assert second_epoch.should_log is True
    assert second_epoch.should_evaluate is False
    assert second_epoch.should_save is False

    final_epoch = callback.on_epoch_end(
        None,
        TrainerState(epoch=4.0, global_step=40, max_steps=40),
        TrainerControl(),
    )
    assert final_epoch.should_log is True
    assert final_epoch.should_evaluate is True
    assert final_epoch.should_save is True


def test_release_training_memory_drops_optimizer_and_clears_cache() -> None:
    class TrackedObject:
        def __init__(self) -> None:
            self.zero_grad_calls: list[bool] = []

        def zero_grad(self, *, set_to_none: bool) -> None:
            self.zero_grad_calls.append(set_to_none)

    class FakeAccelerator:
        def __init__(self) -> None:
            self.free_memory_calls = 0

        def free_memory(self) -> None:
            self.free_memory_calls += 1

    model = TrackedObject()
    optimizer = TrackedObject()
    accelerator = FakeAccelerator()
    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=object(),
        accelerator=accelerator,
    )

    release_training_memory(trainer)  # type: ignore[arg-type]

    assert optimizer.zero_grad_calls == [True]
    assert model.zero_grad_calls == [True]
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert accelerator.free_memory_calls == 1
