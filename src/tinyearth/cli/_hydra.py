"""Shared Hydra wiring for console entry points.

``@hydra.main(config_path=...)`` interprets its argument relative to the
*module* that declares it, and resolves it as an importable module path when the
package is installed. TinyEarth keeps ``configs/`` at the repository root rather
than as package data -- research configs are edited constantly and should not
live inside a wheel -- so the relative form breaks under ``pip install -e .``.

Passing the absolute directory found by :func:`tinyearth.utils.paths.project_root`
works identically for source checkouts, editable installs and wheels. Every
entry point goes through :func:`run_with_hydra` so this decision lives in one
place.
"""

from __future__ import annotations

from collections.abc import Callable

import hydra
from omegaconf import DictConfig

from tinyearth.config.store import register_configs
from tinyearth.utils.paths import configs_dir

__all__ = ["run_with_hydra"]


def run_with_hydra(
    entrypoint: Callable[[DictConfig], None],
    *,
    config_name: str = "config",
) -> None:
    """Compose the TinyEarth config tree and invoke ``entrypoint`` with it.

    Registers the structured-config schema first, so composition is validated
    against :class:`~tinyearth.config.schema.TinyEarthConfig`.

    Args:
        entrypoint: Callable receiving the composed config. Command-line
            overrides are read from :data:`sys.argv` by Hydra.
        config_name: Root config file name, without the ``.yaml`` suffix.
    """
    register_configs()
    decorated = hydra.main(
        version_base=None,
        config_path=str(configs_dir()),
        config_name=config_name,
    )(entrypoint)
    decorated()
