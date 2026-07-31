"""``tinyearth-config``: compose, validate and inspect an experiment config.

This is the Phase 1 entry point. It exercises the full initialisation path a
training run will take -- schema validation, seeding, device resolution, run
directory creation, config persistence -- without requiring a dataset or a
model. That makes it both a useful debugging tool and the acceptance check for
the scaffold.

Example:
    ```bash
    tinyearth-config                                  # defaults
    tinyearth-config +experiment=smoke                # named experiment
    tinyearth-config run.name=ablation seed.value=7   # ad-hoc overrides
    tinyearth-config --dry-run                        # print only, touch nothing
    ```
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from tinyearth.bootstrap import RunContext, initialise_run
from tinyearth.cli._hydra import run_with_hydra
from tinyearth.utils.logging import log_section

__all__ = ["main"]

_DRY_RUN_FLAG = "--dry-run"


def report(context: RunContext) -> None:
    """Log the resolved config, environment and paths for a run.

    Args:
        context: An initialised run context.
    """
    logger = context.logger

    log_section(logger, "Resolved configuration")
    for line in OmegaConf.to_yaml(context.cfg, resolve=True).rstrip().splitlines():
        logger.info("  %s", line)

    log_section(logger, "Environment")
    for key, value in context.device_info.as_dict().items():
        logger.info("  %-16s %s", key, value)

    log_section(logger, "Paths")
    for label, path in (
        ("root", context.paths.root),
        ("data", context.paths.data),
        ("outputs", context.paths.outputs),
        ("cache", context.paths.cache),
        ("run_dir", context.paths.run_dir),
    ):
        logger.info("  %-16s %s", label, path)


def main() -> None:
    """Console-script entry point for ``tinyearth-config``.

    The non-Hydra ``--dry-run`` flag is stripped from :data:`sys.argv` before
    handing over, since Hydra's parser rejects unknown options.

    Making the flag honest takes two further steps. Suppressing TinyEarth's own
    run directory is not enough: Hydra independently writes job metadata and
    creates its run directory. So ``hydra.output_subdir=null`` stops the files,
    and ``hydra.run.dir`` is redirected to a temporary directory that is removed
    afterwards. The result is that ``--dry-run`` leaves nothing behind.
    """
    dry_run = _DRY_RUN_FLAG in sys.argv
    scratch: str | None = None

    if dry_run:
        sys.argv = [arg for arg in sys.argv if arg != _DRY_RUN_FLAG]
        scratch = tempfile.mkdtemp(prefix="tinyearth-dryrun-")
        sys.argv.append("hydra.output_subdir=null")
        # Forward slashes because Hydra's override grammar treats backslashes as
        # escapes, and quoted because Windows short paths contain `~`, which the
        # grammar otherwise fails to lex.
        sys.argv.append(f"hydra.run.dir='{Path(scratch).as_posix()}'")

    def entrypoint(cfg: DictConfig) -> None:
        """Initialise the run and print a summary."""
        context = initialise_run(cfg, create_dirs=not dry_run)
        report(context)
        if dry_run:
            context.logger.info("dry run: nothing written to disk")
        else:
            context.logger.info("wrote %s", context.paths.run_dir / "resolved_config.yaml")

    try:
        run_with_hydra(entrypoint)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
