"""Configuration system: structured schemas, registration and resolution.

Every experiment in TinyEarth is reproducible from a single Hydra config. This
subpackage defines the schema for those configs, registers it so Hydra can
validate against it, and resolves a composed config into absolute paths and
typed objects.
"""

from __future__ import annotations

from tinyearth.config.resolution import (
    ResolvedPaths,
    config_fingerprint,
    from_container,
    resolve_paths,
    save_config,
    to_container,
    to_dataclass,
)
from tinyearth.config.schema import (
    LoggingConfig,
    PathsConfig,
    RunConfig,
    SeedConfig,
    TensorBoardConfig,
    TinyEarthConfig,
    WandbConfig,
)
from tinyearth.config.store import SCHEMA_NAME, register_configs

__all__ = [
    "SCHEMA_NAME",
    "LoggingConfig",
    "PathsConfig",
    "ResolvedPaths",
    "RunConfig",
    "SeedConfig",
    "TensorBoardConfig",
    "TinyEarthConfig",
    "WandbConfig",
    "config_fingerprint",
    "from_container",
    "register_configs",
    "resolve_paths",
    "save_config",
    "to_container",
    "to_dataclass",
]
