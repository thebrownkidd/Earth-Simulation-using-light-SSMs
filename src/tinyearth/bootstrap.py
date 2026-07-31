"""Shared run initialisation for every TinyEarth entry point.

Training, evaluation and benchmarking all need the same opening moves: seed the
RNGs, configure logging, resolve the device, create the run directory and
persist the resolved config. Centralising that here is what keeps entry-point
scripts thin -- the project standard is that a script parses nothing and
orchestrates only.

Example:
    >>> import hydra
    >>> @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    ... def main(cfg):
    ...     context = initialise_run(cfg)
    ...     context.logger.info("training on %s", context.device)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from tinyearth import __version__
from tinyearth.config.resolution import (
    ResolvedPaths,
    config_fingerprint,
    resolve_paths,
    save_config,
)
from tinyearth.utils.device import DeviceInfo, describe_device, resolve_device
from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.seed import seed_everything

__all__ = ["RunContext", "initialise_run"]


@dataclass(frozen=True)
class RunContext:
    """Everything an entry point needs after initialisation.

    Attributes:
        cfg: The composed config, unchanged.
        paths: Absolute, resolved filesystem locations.
        device: Resolved compute device.
        device_info: Hardware provenance for the experimental record.
        seed: The seed that was applied.
        fingerprint: Short hash of the resolved config, excluding ``run.*``.
        logger: Configured logger for the entry point to use.
    """

    cfg: DictConfig
    paths: ResolvedPaths
    device: torch.device
    device_info: DeviceInfo
    seed: int
    fingerprint: str
    logger: logging.Logger


def initialise_run(cfg: DictConfig, *, create_dirs: bool = True) -> RunContext:
    """Seed, configure logging, resolve the device and record provenance.

    Args:
        cfg: Config composed by Hydra against
            :class:`~tinyearth.config.schema.TinyEarthConfig`.
        create_dirs: Create the run and cache directories and write
            ``resolved_config.yaml`` into the run directory. Set ``False`` for
            read-only inspection commands and tests.

    Returns:
        A populated :class:`RunContext`.
    """
    paths = resolve_paths(cfg)
    if create_dirs:
        paths.mkdirs()

    log_file = None
    if create_dirs and cfg.logging.log_file:
        log_file = paths.run_dir / str(cfg.logging.log_file)

    setup_logging(
        level=str(cfg.logging.level),
        rich=bool(cfg.logging.rich),
        log_file=log_file,
        force=True,
    )
    logger = get_logger("tinyearth.run")

    seed = seed_everything(
        int(cfg.seed.value),
        deterministic=bool(cfg.seed.deterministic),
        cudnn_benchmark=bool(cfg.seed.cudnn_benchmark),
    )

    device = resolve_device(str(cfg.run.device))
    device_info = describe_device(device)
    fingerprint = config_fingerprint(cfg)

    if create_dirs:
        save_config(cfg, paths.run_dir / "resolved_config.yaml")

    logger.info("TinyEarth v%s | run=%s/%s", __version__, cfg.run.group, cfg.run.name)
    logger.info("config fingerprint: %s", fingerprint)
    logger.info(
        "device: %s (%s) | seed: %d | deterministic: %s",
        device,
        device_info.device_name,
        seed,
        bool(cfg.seed.deterministic),
    )
    if create_dirs:
        logger.info("run directory: %s", paths.run_dir)

    return RunContext(
        cfg=cfg,
        paths=paths,
        device=device,
        device_info=device_info,
        seed=seed,
        fingerprint=fingerprint,
        logger=logger,
    )
