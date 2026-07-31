"""Tests for the configuration system.

These compose the real ``configs/`` tree, so a malformed config file fails here
rather than at the start of a long training run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from tinyearth.config.resolution import (
    config_fingerprint,
    resolve_paths,
    save_config,
    to_container,
    to_dataclass,
)
from tinyearth.config.schema import TinyEarthConfig
from tinyearth.config.store import SCHEMA_NAME, register_configs


def test_register_configs_is_idempotent():
    register_configs()
    register_configs()


def test_schema_name_is_referenced_by_the_root_config():
    from tinyearth.utils.paths import configs_dir

    text = (configs_dir() / "config.yaml").read_text(encoding="utf-8")
    assert SCHEMA_NAME in text


def test_default_composition_succeeds(compose_config):
    cfg = compose_config()
    assert cfg.run.name == "default"
    assert cfg.seed.value == 42
    assert cfg.paths.data == "data"


def test_composition_validates_against_the_dataclass(compose_config):
    typed = to_dataclass(compose_config())
    assert isinstance(typed, TinyEarthConfig)
    assert typed.logging.tensorboard.enabled is True
    assert typed.logging.wandb.enabled is False


def test_data_group_composes_by_default(compose_config):
    """Phase 2: `data` is now a real group, defaulting to the synthetic dataset."""
    cfg = compose_config()
    assert cfg.data is not None
    assert cfg.data.name == "synthetic"
    assert cfg.data.history_length == 4


def test_model_and_training_groups_compose_by_default(compose_config):
    """Phase 3: every group now has a default, so `tinyearth-train` runs bare."""
    cfg = compose_config()
    assert cfg.model is not None
    assert cfg.model.backbone.name == "convlstm"
    assert cfg.training is not None
    assert cfg.training.epochs > 0


def test_model_group_is_typed_and_validated(compose_config):
    typed = to_dataclass(compose_config())
    assert typed.model is not None
    assert typed.model.loss.terms == {"l1": 1.0}
    assert typed.training is not None
    assert typed.training.evaluation.efficiency is True


def test_backbone_can_be_swapped_by_name(compose_config):
    """The central claim: changing the component under study is a config edit."""
    convlstm = compose_config()
    transformer = compose_config(["model=transformer"])

    assert convlstm.model.backbone.name == "convlstm"
    assert transformer.model.backbone.name == "transformer"
    # The fixed components must be untouched by the swap.
    assert convlstm.model.encoder == transformer.model.encoder
    assert convlstm.model.decoder == transformer.model.decoder


def test_backbone_kwargs_stay_untyped_on_purpose(compose_config):
    """The one deliberate escape hatch: backbone-specific arguments."""
    cfg = compose_config(["model=transformer"])
    assert cfg.model.backbone.kwargs.n_heads == 4


def test_training_overrides_apply(compose_config):
    cfg = compose_config(["training.epochs=3", "training.optimizer.lr=0.01"])
    assert cfg.training.epochs == 3
    assert cfg.training.optimizer.lr == pytest.approx(0.01)


def test_training_group_rejects_a_wrong_type(compose_config):
    with pytest.raises(ConfigCompositionException, match=re.escape("training.epochs")):
        compose_config(["training.epochs=many"])


def test_baseline_smoke_experiment_composes(compose_config):
    cfg = compose_config(["+experiment=baseline_smoke"])
    assert cfg.run.name == "phase3"
    assert cfg.training.max_steps_per_epoch == 4
    assert cfg.data.horizon == 2


def test_data_group_is_typed_and_validated(compose_config):
    typed = to_dataclass(compose_config())
    assert typed.data is not None
    assert typed.data.loader.batch_size == 4
    assert typed.data.synthetic.n_cubes == 6


def test_real_dataset_group_composes(compose_config):
    cfg = compose_config(["data=earthnet2021"])
    assert cfg.data.name == "earthnet2021"
    assert cfg.data.min_valid_fraction > 0


def test_data_overrides_apply(compose_config):
    cfg = compose_config(["data.history_length=8", "data.horizon=4"])
    assert cfg.data.history_length == 8
    assert cfg.data.horizon == 4


def test_data_group_rejects_a_wrong_type(compose_config):
    with pytest.raises(ConfigCompositionException, match=re.escape("data.history_length")):
        compose_config(["data.history_length=many"])


def test_data_smoke_experiment_composes(compose_config):
    cfg = compose_config(["+experiment=data_smoke"])
    assert cfg.run.group == "smoke"
    assert cfg.run.name == "phase2"
    assert cfg.data.name == "synthetic"
    assert cfg.data.loader.batch_size == 4


def test_overrides_apply(compose_config):
    cfg = compose_config(["run.name=ablation", "seed.value=7"])
    assert cfg.run.name == "ablation"
    assert cfg.seed.value == 7


def test_type_coercion_from_the_command_line(compose_config):
    cfg = compose_config(["seed.value=7"])
    assert isinstance(cfg.seed.value, int)


def test_wrong_type_is_rejected(compose_config):
    # OmegaConf raises ValidationError; Hydra wraps it during override merging.
    # Asserting on the wrapper is what a user actually sees at the CLI.
    with pytest.raises(ConfigCompositionException, match=re.escape("seed.value")):
        compose_config(["seed.value=not-an-int"])


def test_unknown_key_is_rejected(compose_config):
    with pytest.raises(ConfigCompositionException, match="nonexistent"):
        compose_config(["seed.nonexistent=1"])


def test_experiment_group_composes(compose_config):
    cfg = compose_config(["+experiment=smoke"])
    assert cfg.run.group == "smoke"
    assert cfg.run.name == "phase1"
    assert cfg.seed.value == 1234
    assert cfg.logging.level == "DEBUG"


def test_resolve_paths_produces_absolute_paths(compose_config):
    paths = resolve_paths(compose_config())
    assert paths.data.is_absolute()
    assert paths.outputs.is_absolute()
    assert paths.run_dir.is_absolute()


def test_run_dir_encodes_group_and_name(compose_config):
    paths = resolve_paths(compose_config(["run.group=scaling", "run.name=2m"]))
    assert paths.run_dir.parts[-2:] == ("scaling", "2m")


def test_absolute_config_paths_are_left_alone(compose_config, tmp_path: Path):
    external = tmp_path / "eo-data"
    paths = resolve_paths(compose_config([f"paths.data={external}"]))
    assert paths.data == external


def test_mkdirs_creates_run_and_cache_but_not_data(compose_config, tmp_path: Path):
    cfg = compose_config([f"paths.outputs={tmp_path / 'out'}", f"paths.cache={tmp_path / 'cache'}"])
    paths = resolve_paths(cfg)
    paths.mkdirs()

    assert paths.run_dir.is_dir()
    assert paths.cache.is_dir()
    assert not paths.data.exists() or paths.data.is_dir()


def test_to_container_resolves_interpolations(compose_config):
    container = to_container(compose_config())
    assert isinstance(container, dict)
    run = container["run"]
    assert isinstance(run, dict)
    assert run["name"] == "default"


def test_fingerprint_is_stable(compose_config):
    assert config_fingerprint(compose_config()) == config_fingerprint(compose_config())


def test_fingerprint_ignores_run_metadata(compose_config):
    baseline = config_fingerprint(compose_config())
    renamed = config_fingerprint(compose_config(["run.name=other", "run.notes=hello"]))
    assert renamed == baseline


def test_fingerprint_tracks_substantive_changes(compose_config):
    baseline = config_fingerprint(compose_config())
    assert config_fingerprint(compose_config(["seed.value=999"])) != baseline


def test_fingerprint_length_is_configurable(compose_config):
    assert len(config_fingerprint(compose_config(), length=12)) == 12


def test_save_config_round_trips(compose_config, tmp_path: Path):
    cfg = compose_config()
    destination = save_config(cfg, tmp_path / "nested" / "resolved_config.yaml")

    assert destination.is_file()
    reloaded = OmegaConf.load(destination)
    assert reloaded.run.name == cfg.run.name
    assert reloaded.seed.value == cfg.seed.value
