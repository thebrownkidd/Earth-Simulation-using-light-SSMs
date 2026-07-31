"""Registration of structured configs with Hydra's ConfigStore.

Hydra only validates a YAML tree against a schema if that schema has been
registered *before* composition. :func:`register_configs` performs that
registration and is idempotent, so entry points and tests can both call it
without ordering constraints.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from tinyearth.config.schema import TinyEarthConfig

__all__ = ["SCHEMA_NAME", "register_configs"]

SCHEMA_NAME = "tinyearth_schema"
"""Name that ``configs/config.yaml`` references in its ``defaults`` list."""

_registered = False


def register_configs() -> None:
    """Register TinyEarth structured configs with the global ConfigStore.

    Safe to call repeatedly; subsequent calls are no-ops.
    """
    global _registered  # noqa: PLW0603 - module-level idempotency guard
    if _registered:
        return

    store = ConfigStore.instance()
    store.store(name=SCHEMA_NAME, node=TinyEarthConfig)
    _registered = True
