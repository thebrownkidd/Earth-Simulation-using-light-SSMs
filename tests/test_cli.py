"""End-to-end tests for the console entry points.

These run the commands in a subprocess. That is slower than calling ``main()``
directly, but it is the only way to exercise what a user actually invokes:
argument parsing, Hydra's global state, and the console-script wiring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tinyearth.utils.paths import project_root

pytestmark = pytest.mark.slow


def _run(module: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke a CLI module in a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=cwd or project_root(),
        check=False,
    )


def test_info_command_succeeds():
    result = _run("tinyearth.cli.info")
    assert result.returncode == 0, result.stderr
    assert "TinyEarth" in result.stdout
    assert "torch version" in result.stdout


def test_config_command_succeeds(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_config",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr


def test_config_command_writes_the_run_artefacts(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_config",
        "logging.rich=false",
        "run.group=g",
        "run.name=n",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "g" / "n" / "resolved_config.yaml").is_file()
    assert (tmp_path / "g" / "n" / "run.log").is_file()


def test_smoke_experiment_composes(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_config",
        "+experiment=smoke",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "smoke" / "phase1" / "resolved_config.yaml").is_file()


def test_dry_run_writes_absolutely_nothing(tmp_path: Path):
    outputs = tmp_path / "outputs"
    result = _run(
        "tinyearth.cli.inspect_config",
        "--dry-run",
        "logging.rich=false",
        f"paths.outputs={outputs.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )

    assert result.returncode == 0, result.stderr
    assert "nothing written to disk" in result.stdout + result.stderr
    # Neither TinyEarth's run directory nor Hydra's job metadata directory.
    assert not outputs.exists()


def test_data_command_succeeds(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_data",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr
    assert "data pipeline OK" in result.stdout + result.stderr


def test_data_smoke_experiment_reports_shapes(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_data",
        "+experiment=data_smoke",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "images=(4, 4, 4, 16, 16)" in output
    assert "target=(4, 1, 4, 16, 16)" in output
    assert "same seed -> same batch True" in output
    assert "train/val overlap  0" in output


def test_data_command_respects_horizon_override(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_data",
        "logging.rich=false",
        "data.history_length=6",
        "data.horizon=4",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "6 -> 4" in output


def test_missing_real_dataset_gives_download_instructions(tmp_path: Path):
    """The most common setup failure must be actionable, not a bare traceback."""
    result = _run(
        "tinyearth.cli.inspect_data",
        "data=earthnet2021",
        "logging.rich=false",
        f"paths.data={(tmp_path / 'absent').as_posix()}",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "download_earthnet2021" in output
    assert "does not redistribute" in output


def test_train_command_runs_end_to_end(tmp_path: Path):
    result = _run(
        "tinyearth.cli.train",
        "+experiment=baseline_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "training complete" in output

    run_dir = tmp_path / "smoke" / "phase3"
    for artefact in ("summary.json", "metrics.jsonl", "last.ckpt", "best.ckpt"):
        assert (run_dir / artefact).is_file(), f"missing {artefact}"


def test_train_reports_quality_and_efficiency(tmp_path: Path):
    """Both metric families must appear automatically, without a flag."""
    result = _run(
        "tinyearth.cli.train",
        "+experiment=baseline_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((tmp_path / "smoke" / "phase3" / "summary.json").read_text())
    for key in ("val/mae", "val/rmse", "val/psnr", "val/ssim", "val/sam"):
        assert key in summary, f"missing forecast-quality metric {key}"
    for key in (
        "efficiency/parameters",
        "efficiency/latency_ms",
        "efficiency/throughput_samples_per_s",
        "params/backbone_fraction",
    ):
        assert key in summary, f"missing efficiency metric {key}"


def test_swapping_the_backbone_changes_only_the_backbone(tmp_path: Path):
    """The project's central claim, verified through the actual CLI."""
    summaries = {}
    for backbone in ("convlstm", "transformer"):
        out = tmp_path / backbone
        result = _run(
            "tinyearth.cli.train",
            "+experiment=baseline_smoke",
            f"model={backbone}",
            "logging.rich=false",
            "logging.tensorboard.enabled=false",
            f"paths.outputs={out.as_posix()}",
            f"paths.cache={(tmp_path / 'cache').as_posix()}",
        )
        assert result.returncode == 0, result.stderr
        summaries[backbone] = json.loads((out / "smoke" / "phase3" / "summary.json").read_text())

    convlstm, transformer = summaries["convlstm"], summaries["transformer"]
    assert convlstm["params/encoder"] == transformer["params/encoder"]
    assert convlstm["params/decoder"] == transformer["params/decoder"]
    assert convlstm["params/backbone"] != transformer["params/backbone"]


def test_ssm_smoke_experiment_trains(tmp_path: Path):
    result = _run(
        "tinyearth.cli.train",
        "+experiment=ssm_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "smoke" / "phase4" / "summary.json").is_file()


@pytest.mark.parametrize("backbone", ["s4d", "mamba", "convlstm", "transformer"])
def test_every_backbone_trains_from_one_sweep_config(tmp_path: Path, backbone):
    """A cross-architecture sweep must run unchanged on all four backbones.

    The sweep config sets SSM-specific kwargs; the ConvLSTM and transformer must
    drop them rather than crash, or no such sweep is possible.
    """
    result = _run(
        "tinyearth.cli.train",
        "+experiment=ssm_smoke",
        f"model={backbone}",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr


def test_typo_in_a_backbone_argument_still_fails(tmp_path: Path):
    """Dropping inapplicable kwargs must not swallow genuine typos."""
    result = _run(
        "tinyearth.cli.train",
        "+experiment=ssm_smoke",
        "+model.backbone.kwargs.hiden_dim=64",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "hiden_dim" in output


def test_size_tier_resolves_to_a_calibrated_width(tmp_path: Path):
    """A size tier must reach the built model, not merely be computed.

    Model configs ship a concrete `kwargs.hidden_dim`, which wins over `size`.
    Standing it down with `null` is what lets the tier apply. Asserting on the
    *built* width rather than on a log line matters: the message warning that
    the tier was ignored also contains the tier's width, so a substring check
    passed happily while every size-tier run used the default width instead.
    """
    result = _run(
        "tinyearth.cli.inspect_model",
        "model=s4d",
        "model.backbone.size=tiny",
        "~model.backbone.kwargs.hidden_dim",
        "logging.rich=false",
        "data.synthetic.size=16",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "size tier 'tiny' -> hidden_dim=272" in output
    assert "is set and wins" not in output


def test_an_explicit_width_overrides_the_tier_and_says_so(tmp_path: Path):
    """An explicit width must override the tier, and say so loudly.

    The override itself is legitimate -- the hidden_dim sweep relies on it --
    but left quiet it disables a matched-budget comparison without a trace.
    """
    result = _run(
        "tinyearth.cli.inspect_model",
        "model=s4d",
        "model.backbone.size=tiny",
        "model.backbone.kwargs.hidden_dim=64",
        "logging.rich=false",
        "data.synthetic.size=16",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "size tier is NOT in effect" in output


def test_a_backbone_with_no_width_at_all_fails_loudly(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_model",
        "model=s4d",
        "~model.backbone.kwargs.hidden_dim",
        "logging.rich=false",
        "data.synthetic.size=16",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "width is undefined" in output


def test_model_command_reports_size_and_cost(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_model",
        "logging.rich=false",
        "data.synthetic.size=32",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, result.stderr
    assert "parameters" in output
    assert "latency (ms)" in output


def test_resume_continues_training_past_the_checkpointed_epoch(tmp_path: Path):
    """The end-to-end path a user actually types, not just the in-process API.

    ``--resume`` is a non-Hydra flag (Hydra's override grammar has no notion
    of a bare flag taking a value), so it must be stripped from argv before
    Hydra parses the rest -- exercised here through the real console-script
    argument parsing, the way ``--dry-run``'s equivalent tests are.
    """
    outputs = tmp_path.as_posix()
    cache = (tmp_path / "cache").as_posix()

    first = _run(
        "tinyearth.cli.train",
        "+experiment=baseline_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        "training.epochs=2",
        f"paths.outputs={outputs}",
        f"paths.cache={cache}",
    )
    assert first.returncode == 0, first.stderr
    checkpoint = tmp_path / "smoke" / "phase3" / "last.ckpt"
    assert checkpoint.is_file()

    second = _run(
        "tinyearth.cli.train",
        "+experiment=baseline_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        "training.epochs=4",
        f"paths.outputs={outputs}",
        f"paths.cache={cache}",
        "--resume",
        checkpoint.as_posix(),
    )
    output = second.stdout + second.stderr
    assert second.returncode == 0, second.stderr
    assert "resuming at epoch 3/4" in output

    summary = json.loads((tmp_path / "smoke" / "phase3" / "summary.json").read_text())
    assert summary["architecture_version"] == "v1_baseline"


def test_resume_without_a_path_fails_loudly():
    result = _run("tinyearth.cli.train", "--resume")
    assert result.returncode != 0
    assert "requires a checkpoint path" in result.stdout + result.stderr


def test_resume_refuses_a_mismatched_architecture_version(tmp_path: Path):
    """Resuming a v1 checkpoint into v2 code must fail loudly, not warp weights."""
    outputs = tmp_path.as_posix()
    cache = (tmp_path / "cache").as_posix()

    first = _run(
        "tinyearth.cli.train",
        "+experiment=baseline_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        "training.epochs=1",
        "model.architecture_version=v1_baseline",
        f"paths.outputs={outputs}",
        f"paths.cache={cache}",
    )
    assert first.returncode == 0, first.stderr
    checkpoint = tmp_path / "smoke" / "phase3" / "last.ckpt"

    second = _run(
        "tinyearth.cli.train",
        "+experiment=baseline_smoke",
        "logging.rich=false",
        "logging.tensorboard.enabled=false",
        "training.epochs=2",
        "model.architecture_version=v2_skip_gdl",
        f"paths.outputs={(tmp_path / 'v2').as_posix()}",
        f"paths.cache={cache}",
        "--resume",
        checkpoint.as_posix(),
    )
    output = second.stdout + second.stderr
    assert second.returncode != 0
    assert "Refusing to resume" in output


def test_invalid_override_fails_loudly():
    result = _run("tinyearth.cli.inspect_config", "--dry-run", "seed.value=not-an-int")
    assert result.returncode != 0
    assert "seed.value" in result.stdout + result.stderr


def test_unknown_key_fails_loudly():
    result = _run("tinyearth.cli.inspect_config", "--dry-run", "seed.nonexistent=1")
    assert result.returncode != 0
