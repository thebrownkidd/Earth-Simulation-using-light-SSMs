"""Cross-cutting utilities: determinism, logging, devices, paths and registries.

Nothing in this subpackage imports from :mod:`tinyearth.models`,
:mod:`tinyearth.datasets` or :mod:`tinyearth.training`. The dependency arrow
points one way, which keeps these helpers importable from anywhere without
circularity.
"""

from __future__ import annotations

from tinyearth.utils.device import DeviceInfo, describe_device, resolve_device, supports_amp
from tinyearth.utils.logging import get_logger, log_section, setup_logging
from tinyearth.utils.paths import (
    cache_dir,
    configs_dir,
    data_dir,
    outputs_dir,
    project_root,
)
from tinyearth.utils.registry import Registry
from tinyearth.utils.seed import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seeded_generator,
    temporary_seed,
    worker_init_fn,
)

__all__ = [
    "DeviceInfo",
    "Registry",
    "cache_dir",
    "capture_rng_state",
    "configs_dir",
    "data_dir",
    "describe_device",
    "get_logger",
    "log_section",
    "outputs_dir",
    "project_root",
    "resolve_device",
    "restore_rng_state",
    "seed_everything",
    "seeded_generator",
    "setup_logging",
    "supports_amp",
    "temporary_seed",
    "worker_init_fn",
]
