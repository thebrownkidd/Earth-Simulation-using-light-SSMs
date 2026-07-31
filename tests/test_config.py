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


def test_phase_placeholders_are_present_but_empty(compose_config):
    cfg = compose_config()
    assert cfg.data is None
    assert cfg.model is None
    assert cfg.training is None


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
