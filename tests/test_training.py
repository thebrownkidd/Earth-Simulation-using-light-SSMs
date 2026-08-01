"""Tests for the training loop, optimiser construction and tracking.

The trainer lives in a library module precisely so it can be tested in-process
like this, rather than only through a subprocess that runs a whole experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from tinyearth.config.schema import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from tinyearth.datasets.loaders import LoaderSettings, build_dataloader
from tinyearth.datasets.synthetic import SyntheticEarthNet2021, SyntheticSpec
from tinyearth.models.decoders import CNNDecoder
from tinyearth.models.encoders import CNNEncoder
from tinyearth.models.forecaster import Forecaster
from tinyearth.models.losses import L1Loss
from tinyearth.models.temporal import ConvLSTMBackbone
from tinyearth.training.optim import build_optimizer, build_scheduler, current_lr, parameter_groups
from tinyearth.training.tracking import JsonTracker, MultiTracker, NullTracker
from tinyearth.training.trainer import Trainer

SPEC = SyntheticSpec(n_cubes=6, n_frames=10, size=16, seed=0)
CHANNELS, LATENT, HORIZON = 4, 16, 2


def make_model() -> Forecaster:
    """Build a very small forecaster."""
    return Forecaster(
        encoder=CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=2),
        backbone=ConvLSTMBackbone(latent_dim=LATENT, hidden_dim=16, n_layers=1),
        decoder=CNNDecoder(CHANNELS, LATENT, base_channels=8, depth=2),
        horizon=HORIZON,
    )


def make_training_config(**kwargs: object) -> TrainingConfig:
    """Build a fast training config."""
    cfg = TrainingConfig(
        epochs=2,
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(name="cosine", warmup_epochs=0),
        checkpoint=CheckpointConfig(enabled=True),
        evaluation=EvaluationConfig(efficiency=False),
        max_steps_per_epoch=2,
        log_every_n_steps=1,
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def loaders(tmp_path: Path):
    """Return small train and validation loaders over synthetic cubes."""
    common = {
        "spec": SPEC,
        "history_length": 3,
        "horizon": HORIZON,
        "val_fraction": 0.25,
        "context_frames": 5,
    }
    train = SyntheticEarthNet2021(tmp_path / "syn", "train", **common)  # type: ignore[arg-type]
    val = SyntheticEarthNet2021(tmp_path / "syn", "val", **common)  # type: ignore[arg-type]
    settings = LoaderSettings(batch_size=2, shuffle=False)
    return build_dataloader(train, settings), build_dataloader(val, settings)


@pytest.fixture
def trainer(loaders, tmp_path: Path) -> Trainer:
    train_loader, val_loader = loaders
    model = make_model()
    cfg = make_training_config()
    return Trainer(
        model=model,
        loss=L1Loss(),
        optimizer=build_optimizer(model, cfg.optimizer),
        cfg=cfg,
        device=torch.device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=tmp_path / "run",
        tracker=NullTracker(),
    )


class TestParameterGroups:
    def test_splits_into_two_groups(self):
        groups = parameter_groups(make_model(), weight_decay=0.01)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0

    def test_norms_and_biases_are_excluded_from_decay(self):
        """Decaying a norm scale fights the layer's purpose, unevenly per backbone."""
        model = make_model()
        groups = parameter_groups(model, weight_decay=0.01)
        excluded = {id(p) for p in cast("list[torch.nn.Parameter]", groups[1]["params"])}

        for name, param in model.named_parameters():
            if name.endswith(".bias") or param.ndim <= 1:
                assert id(param) in excluded, f"{name} should be excluded from decay"

    def test_every_trainable_parameter_lands_in_exactly_one_group(self):
        model = make_model()
        groups = parameter_groups(model, weight_decay=0.01)
        assigned = [
            id(p) for group in groups for p in cast("list[torch.nn.Parameter]", group["params"])
        ]

        assert len(assigned) == len(set(assigned))
        assert len(assigned) == sum(1 for p in model.parameters() if p.requires_grad)


class TestOptimizer:
    @pytest.mark.parametrize("name", ["adamw", "adam", "sgd"])
    def test_builds_each_supported_optimizer(self, name):
        cfg = OptimizerConfig(name=name, lr=1e-3)
        assert build_optimizer(make_model(), cfg) is not None

    def test_unknown_optimizer(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(make_model(), OptimizerConfig(name="lbfgs"))

    def test_current_lr_reads_the_first_group(self):
        cfg = OptimizerConfig(lr=3e-4)
        assert current_lr(build_optimizer(make_model(), cfg)) == pytest.approx(3e-4)


class TestScheduler:
    def test_none_returns_nothing(self):
        optimizer = build_optimizer(make_model(), OptimizerConfig())
        cfg = SchedulerConfig(name="none")
        assert build_scheduler(optimizer, cfg, steps_per_epoch=5, epochs=2) is None

    def test_plateau_is_driven_by_the_trainer_not_here(self):
        optimizer = build_optimizer(make_model(), OptimizerConfig())
        cfg = SchedulerConfig(name="plateau")
        assert build_scheduler(optimizer, cfg, steps_per_epoch=5, epochs=2) is None

    def test_cosine_decays_the_learning_rate(self):
        optimizer = build_optimizer(make_model(), OptimizerConfig(lr=1e-2))
        scheduler = build_scheduler(
            optimizer, SchedulerConfig(name="cosine", warmup_epochs=0), steps_per_epoch=10, epochs=1
        )
        assert scheduler is not None

        start = current_lr(optimizer)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        assert current_lr(optimizer) < start

    def test_cosine_warmup_starts_low_and_rises(self):
        """Warmup matters for the transformer baseline, which is unstable without it."""
        optimizer = build_optimizer(make_model(), OptimizerConfig(lr=1e-2))
        scheduler = build_scheduler(
            optimizer, SchedulerConfig(name="cosine", warmup_epochs=1), steps_per_epoch=10, epochs=5
        )
        assert scheduler is not None

        first = current_lr(optimizer)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert current_lr(optimizer) > first

    def test_cosine_respects_min_lr(self):
        optimizer = build_optimizer(make_model(), OptimizerConfig(lr=1e-2))
        cfg = SchedulerConfig(name="cosine", warmup_epochs=0, min_lr=1e-4)
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=10, epochs=1)
        assert scheduler is not None

        for _ in range(10):
            optimizer.step()
            scheduler.step()
        assert current_lr(optimizer) == pytest.approx(1e-4, rel=1e-3)

    def test_unknown_scheduler(self):
        optimizer = build_optimizer(make_model(), OptimizerConfig())
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(
                optimizer, SchedulerConfig(name="triangular"), steps_per_epoch=5, epochs=1
            )


class TestTrainer:
    def test_train_epoch_returns_prefixed_metrics(self, trainer):
        metrics = trainer.train_epoch()
        assert "train/loss" in metrics
        assert {"train/mae", "train/rmse", "train/psnr", "train/ssim", "train/sam"} <= set(metrics)

    def test_validate_returns_prefixed_metrics(self, trainer):
        metrics = trainer.validate()
        assert "val/loss" in metrics
        assert "val/mae" in metrics

    def test_training_updates_the_weights(self, trainer):
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.train_epoch()
        after = list(trainer.model.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_validation_does_not_update_weights(self, trainer):
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.validate()
        after = list(trainer.model.parameters())
        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_max_steps_per_epoch_is_respected(self, trainer):
        trainer.cfg.max_steps_per_epoch = 2
        trainer.global_step = 0
        trainer.train_epoch()
        assert trainer.global_step == 2

    def test_fit_runs_every_epoch(self, trainer):
        result = trainer.fit()
        assert len(result.epochs) == trainer.cfg.epochs

    def test_fit_records_the_parameter_breakdown(self, trainer):
        result = trainer.fit()
        assert result.parameters["params/total"] > 0
        assert 0 < result.parameters["params/backbone_fraction"] <= 1

    def test_fit_tracks_the_best_epoch(self, trainer):
        result = trainer.fit()
        assert result.best_epoch >= 0
        assert result.best_metric < float("inf")

    def test_checkpoints_are_written(self, trainer):
        trainer.fit()
        assert (trainer.run_dir / "last.ckpt").is_file()
        assert (trainer.run_dir / "best.ckpt").is_file()

    def test_checkpoint_carries_enough_to_reproduce(self, trainer):
        trainer.fit()
        payload = torch.load(trainer.run_dir / "last.ckpt", weights_only=False)
        assert {"model", "optimizer", "epoch", "metrics", "config", "rng"} <= set(payload)

    def test_checkpointing_can_be_disabled(self, trainer):
        trainer.cfg.checkpoint.enabled = False
        trainer.fit()
        assert not (trainer.run_dir / "last.ckpt").exists()

    def test_no_validation_loader_skips_validation(self, trainer):
        trainer.val_loader = None
        result = trainer.fit()
        assert all(not epoch.validation for epoch in result.epochs)

    def test_early_stopping_triggers(self, trainer):
        trainer.cfg.epochs = 10
        trainer.cfg.early_stopping_patience = 1
        # A zero learning rate leaves the weights untouched, so validation loss
        # is identical every epoch and never counts as an improvement. Freezing
        # the parameters instead would break backward entirely.
        for group in trainer.optimizer.param_groups:
            group["lr"] = 0.0
        trainer.scheduler = None

        result = trainer.fit()

        assert result.stopped_early
        assert len(result.epochs) < 10

    def test_amp_is_disabled_on_cpu(self, trainer):
        trainer.cfg.amp = True
        assert (
            Trainer(
                model=trainer.model,
                loss=trainer.loss,
                optimizer=trainer.optimizer,
                cfg=trainer.cfg,
                device=torch.device("cpu"),
                train_loader=trainer.train_loader,
            ).amp_enabled
            is False
        )

    def test_profile_reports_efficiency(self, trainer):
        trainer.cfg.evaluation.efficiency_warmup = 1
        trainer.cfg.evaluation.efficiency_iterations = 2
        report = trainer.profile()
        assert report is not None
        assert report.parameters > 0
        assert report.latency_ms > 0

    def test_efficiency_runs_automatically_when_enabled(self, trainer):
        trainer.cfg.evaluation.efficiency = True
        trainer.cfg.evaluation.efficiency_warmup = 1
        trainer.cfg.evaluation.efficiency_iterations = 2
        assert trainer.fit().efficiency is not None

    def test_summary_flattens_headline_numbers(self, trainer):
        summary = trainer.fit().summary()
        assert "best/metric" in summary
        assert "params/total" in summary
        assert all(isinstance(value, float) for value in summary.values())

    def test_masks_reach_the_loss(self, trainer):
        """The dataset supplies masks; the trainer must forward them."""
        batch = next(iter(trainer.train_loader))
        assert "target_mask" in batch
        _, _, mask = trainer._to_device(batch)
        assert mask is not None


def _checkpoint_path(trainer: Trainer, name: str = "last") -> Path:
    """Return a trainer's checkpoint path, for tests that know it must exist.

    ``Trainer.run_dir`` is typed ``Path | None`` since a Trainer built without
    one (as in a notebook) never checkpoints; every trainer these resume tests
    build has one, so this asserts that rather than repeating the same
    ``assert ... is not None`` at each call site.
    """
    assert trainer.run_dir is not None
    return trainer.run_dir / f"{name}.ckpt"


class TestResume:
    """Round-trips through save_checkpoint / load_checkpoint.

    The mechanism under test is exercised end to end rather than piece by
    piece: if optimizer momentum, the epoch index, or RNG state failed to
    round-trip correctly, the resumed run would silently diverge from an
    uninterrupted one and the final-metrics comparison below would catch it,
    without needing to separately assert on each internal piece.
    """

    def _make_trainer(
        self,
        loaders,
        tmp_path: Path,
        run_name: str,
        *,
        epochs: int,
        seed: int,
        architecture_version: str = "v1_baseline",
    ) -> Trainer:
        torch.manual_seed(seed)
        model = make_model()
        cfg = make_training_config(epochs=epochs)
        train_loader, val_loader = loaders
        return Trainer(
            model=model,
            loss=L1Loss(),
            optimizer=build_optimizer(model, cfg.optimizer),
            cfg=cfg,
            device=torch.device("cpu"),
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=tmp_path / run_name,
            tracker=NullTracker(),
            architecture_version=architecture_version,
        )

    def test_resumed_run_matches_an_uninterrupted_run(self, loaders, tmp_path):
        """Train 2+2 with a checkpoint in between; must match a straight 4."""
        seed = 123

        straight = self._make_trainer(loaders, tmp_path, "straight", epochs=4, seed=seed)
        straight_result = straight.fit()

        first_half = self._make_trainer(loaders, tmp_path, "resumed", epochs=2, seed=seed)
        first_half.fit()
        checkpoint = _checkpoint_path(first_half)
        assert checkpoint.is_file()

        second_half = self._make_trainer(loaders, tmp_path, "resumed", epochs=4, seed=seed)
        start_epoch = second_half.load_checkpoint(checkpoint)
        assert start_epoch == 2
        resumed_result = second_half.fit(start_epoch=start_epoch)

        # Only epochs 3-4 ran in this call.
        assert [epoch.epoch for epoch in resumed_result.epochs] == [2, 3]

        straight_final = straight_result.epochs[-1].validation
        resumed_final = resumed_result.epochs[-1].validation
        for key in ("val/mae", "val/loss"):
            assert resumed_final[key] == pytest.approx(straight_final[key], rel=1e-4)

        for (name, expected), (_, actual) in zip(
            straight.model.named_parameters(), second_half.model.named_parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, msg=f"parameter {name!r} does not match")

    def test_global_step_resumes_rather_than_restarting(self, loaders, tmp_path):
        first_half = self._make_trainer(loaders, tmp_path, "steps", epochs=2, seed=1)
        first_half.fit()
        steps_after_two_epochs = first_half.global_step

        second_half = self._make_trainer(loaders, tmp_path, "steps", epochs=4, seed=1)
        second_half.load_checkpoint(_checkpoint_path(first_half))
        assert second_half.global_step == steps_after_two_epochs

    def test_resuming_past_the_configured_epochs_runs_nothing_further(self, loaders, tmp_path):
        first = self._make_trainer(loaders, tmp_path, "done", epochs=2, seed=1)
        first.fit()

        second = self._make_trainer(loaders, tmp_path, "done", epochs=2, seed=1)
        start_epoch = second.load_checkpoint(_checkpoint_path(first))
        result = second.fit(start_epoch=start_epoch)

        assert result.epochs == []

    def test_matching_architecture_version_is_accepted(self, loaders, tmp_path):
        first = self._make_trainer(
            loaders, tmp_path, "match", epochs=1, seed=1, architecture_version="v2_skip_gdl"
        )
        first.fit()

        second = self._make_trainer(
            loaders, tmp_path, "match", epochs=2, seed=1, architecture_version="v2_skip_gdl"
        )
        assert (
            second.load_checkpoint(_checkpoint_path(first), architecture_version="v2_skip_gdl") == 1
        )

    def test_mismatched_architecture_version_is_refused_loudly(self, loaders, tmp_path):
        """Resuming a v1 checkpoint into v2 code (or the reverse) must fail, not warp weights."""
        first = self._make_trainer(
            loaders, tmp_path, "v1", epochs=1, seed=1, architecture_version="v1_baseline"
        )
        first.fit()

        second = self._make_trainer(
            loaders, tmp_path, "v2", epochs=2, seed=1, architecture_version="v2_skip_gdl"
        )
        with pytest.raises(ValueError, match="Refusing to resume"):
            second.load_checkpoint(_checkpoint_path(first), architecture_version="v2_skip_gdl")

    def test_no_expected_version_skips_the_check(self, loaders, tmp_path):
        """architecture_version=None is an explicit opt-out, not a default worth hiding."""
        first = self._make_trainer(
            loaders, tmp_path, "any", epochs=1, seed=1, architecture_version="v1_baseline"
        )
        first.fit()

        second = self._make_trainer(
            loaders, tmp_path, "any", epochs=2, seed=1, architecture_version="v2_skip_gdl"
        )
        assert second.load_checkpoint(_checkpoint_path(first)) == 1


class TestTracking:
    def test_json_tracker_writes_one_record_per_call(self, tmp_path: Path):
        tracker = JsonTracker(tmp_path / "metrics.jsonl")
        tracker.log_scalars({"loss": 1.0}, 0)
        tracker.log_scalars({"loss": 0.5}, 1)
        tracker.close()

        lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["loss"] == 0.5

    def test_json_tracker_writes_a_summary(self, tmp_path: Path):
        tracker = JsonTracker(tmp_path / "metrics.jsonl")
        tracker.log_scalars({"loss": 0.25}, 0)
        tracker.close()

        summary = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert summary["loss"] == 0.25

    def test_json_tracker_records_hyperparameters(self, tmp_path: Path):
        tracker = JsonTracker(tmp_path / "metrics.jsonl")
        tracker.log_hyperparameters({"lr": 0.001})
        tracker.close()
        assert "hyperparameters" in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")

    def test_close_is_idempotent(self, tmp_path: Path):
        tracker = JsonTracker(tmp_path / "metrics.jsonl")
        tracker.close()
        tracker.close()

    def test_multi_tracker_forwards_to_all(self, tmp_path: Path):
        first = JsonTracker(tmp_path / "a.jsonl")
        second = JsonTracker(tmp_path / "b.jsonl")
        multi = MultiTracker([first, second])
        multi.log_scalars({"x": 1.0}, 0)
        multi.close()

        assert (tmp_path / "a.jsonl").read_text(encoding="utf-8").strip()
        assert (tmp_path / "b.jsonl").read_text(encoding="utf-8").strip()

    def test_multi_tracker_survives_a_failing_backend(self, tmp_path: Path):
        """One tracker failing to close must not lose the others' results."""

        class Exploding(NullTracker):
            def close(self) -> None:
                raise RuntimeError("boom")

        good = JsonTracker(tmp_path / "good.jsonl")
        multi = MultiTracker([Exploding(), good])
        multi.log_scalars({"x": 1.0}, 0)
        multi.close()

        assert (tmp_path / "metrics.json").is_file()

    def test_build_tracker_always_includes_json(self, tmp_path: Path):
        from tinyearth.training.tracking import build_tracker

        cfg = LoggingConfig()
        cfg.tensorboard.enabled = False
        tracker = build_tracker(cfg, tmp_path, "test")
        tracker.close()

        assert (tmp_path / "metrics.jsonl").is_file()
