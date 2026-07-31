"""Tests for shared run initialisation."""

from __future__ import annotations

from pathlib import Path

import torch

from tinyearth.bootstrap import initialise_run


def test_initialise_run_populates_the_context(compose_config, tmp_path: Path):
    cfg = compose_config([f"paths.outputs={tmp_path}", f"paths.cache={tmp_path / 'cache'}"])
    context = initialise_run(cfg)

    assert isinstance(context.device, torch.device)
    assert context.seed == cfg.seed.value
    assert len(context.fingerprint) == 8
    assert context.logger.name == "tinyearth.run"


def test_initialise_run_creates_the_run_directory(compose_config, tmp_path: Path):
    cfg = compose_config(
        [
            f"paths.outputs={tmp_path}",
            f"paths.cache={tmp_path / 'cache'}",
            "run.group=g",
            "run.name=n",
        ]
    )
    context = initialise_run(cfg)

    assert context.paths.run_dir == tmp_path / "g" / "n"
    assert context.paths.run_dir.is_dir()


def test_initialise_run_persists_the_resolved_config(compose_config, tmp_path: Path):
    cfg = compose_config([f"paths.outputs={tmp_path}", f"paths.cache={tmp_path / 'cache'}"])
    context = initialise_run(cfg)

    saved = context.paths.run_dir / "resolved_config.yaml"
    assert saved.is_file()
    assert "fingerprint" not in saved.read_text(encoding="utf-8")  # config only, no derived state


def test_initialise_run_writes_a_log_file(compose_config, tmp_path: Path):
    cfg = compose_config([f"paths.outputs={tmp_path}", f"paths.cache={tmp_path / 'cache'}"])
    context = initialise_run(cfg)

    assert (context.paths.run_dir / "run.log").is_file()


def test_dry_run_touches_nothing(compose_config, tmp_path: Path):
    target = tmp_path / "unwritten"
    cfg = compose_config([f"paths.outputs={target}", f"paths.cache={tmp_path / 'cache'}"])
    context = initialise_run(cfg, create_dirs=False)

    assert not target.exists()
    assert context.fingerprint


def test_seeding_actually_happened(compose_config, tmp_path: Path):
    cfg = compose_config(
        [f"paths.outputs={tmp_path}", f"paths.cache={tmp_path / 'cache'}", "seed.value=321"]
    )

    initialise_run(cfg)
    first = torch.rand(3)
    initialise_run(cfg)
    torch.testing.assert_close(torch.rand(3), first)


def test_repeated_initialisation_does_not_duplicate_log_handlers(compose_config, tmp_path: Path):
    cfg = compose_config([f"paths.outputs={tmp_path}", f"paths.cache={tmp_path / 'cache'}"])

    initialise_run(cfg)
    context = initialise_run(cfg)

    import logging

    assert len(logging.getLogger("tinyearth").handlers) == 2  # console + file
    assert context.logger.name == "tinyearth.run"
