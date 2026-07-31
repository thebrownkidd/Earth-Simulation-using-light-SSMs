"""Turning a composed config into resolved, usable objects.

Hydra hands back a :class:`~omegaconf.DictConfig`. Two things must happen before
the rest of the codebase touches it:

1. **Path resolution.** Config paths are written relative to the repository
   root, but Hydra changes the working directory at run start. Resolving once,
   here, keeps ``Path`` arithmetic out of every call site.
2. **Provenance capture.** A run is only reproducible if the *fully resolved*
   config -- interpolations expanded, overrides applied -- is stored alongside
   its outputs. :func:`save_config` does that, and :func:`config_fingerprint`
   gives a short stable id for grouping repeated runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from tinyearth.config.schema import TinyEarthConfig
from tinyearth.utils.paths import project_root

__all__ = [
    "ResolvedPaths",
    "config_fingerprint",
    "resolve_paths",
    "save_config",
    "to_container",
    "to_dataclass",
]


@dataclass(frozen=True)
class ResolvedPaths:
    """Absolute filesystem locations for a run.

    Attributes:
        root: Repository root.
        data: Dataset root.
        outputs: Root of all run outputs.
        cache: Cache root.
        run_dir: Directory for this specific run's artefacts.
    """

    root: Path
    data: Path
    outputs: Path
    cache: Path
    run_dir: Path

    def mkdirs(self) -> None:
        """Create the output and cache directories for this run.

        The data directory is deliberately not created: a missing dataset
        should surface as a clear error from the Phase 2 loader rather than as
        an empty directory that looks valid.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)


def _absolute(value: str, root: Path) -> Path:
    """Interpret ``value`` as absolute, or relative to ``root``.

    Args:
        value: Path string from the config.
        root: Base directory for relative paths.

    Returns:
        An absolute path.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def resolve_paths(cfg: TinyEarthConfig | DictConfig) -> ResolvedPaths:
    """Resolve configured paths to absolute locations.

    The run directory is ``<outputs>/<run.group>/<run.name>``, which keeps a
    sweep's runs adjacent on disk and makes globbing results straightforward.

    Args:
        cfg: A composed TinyEarth config.

    Returns:
        Absolute paths for the run. Directories are not created; call
        :meth:`ResolvedPaths.mkdirs`.
    """
    root = project_root()
    outputs = _absolute(str(cfg.paths.outputs), root)
    return ResolvedPaths(
        root=root,
        data=_absolute(str(cfg.paths.data), root),
        outputs=outputs,
        cache=_absolute(str(cfg.paths.cache), root),
        run_dir=outputs / str(cfg.run.group) / str(cfg.run.name),
    )


def to_container(cfg: DictConfig | TinyEarthConfig) -> dict[str, object]:
    """Convert a config to a plain dictionary with interpolations expanded.

    Args:
        cfg: A composed config.

    Returns:
        A JSON-serialisable nested dictionary.
    """
    structured = cfg if isinstance(cfg, DictConfig) else OmegaConf.structured(cfg)
    container = OmegaConf.to_container(structured, resolve=True)
    if not isinstance(container, dict):  # pragma: no cover - structurally impossible
        raise TypeError(f"Expected the config to resolve to a mapping, got {type(container)!r}.")
    return dict(container)


def to_dataclass(cfg: DictConfig) -> TinyEarthConfig:
    """Convert a composed :class:`~omegaconf.DictConfig` to a typed dataclass.

    Useful in library code and tests that want static typing and attribute
    completion rather than OmegaConf's dynamic access.

    Args:
        cfg: A config composed against :class:`TinyEarthConfig`.

    Returns:
        The equivalent typed dataclass instance.

    Raises:
        TypeError: If the config does not conform to the schema.
    """
    obj = OmegaConf.to_object(cfg)
    if not isinstance(obj, TinyEarthConfig):
        raise TypeError(
            "Config does not conform to TinyEarthConfig. Ensure `defaults` in "
            "configs/config.yaml lists the registered schema."
        )
    return obj


def config_fingerprint(cfg: DictConfig | TinyEarthConfig, length: int = 8) -> str:
    """Compute a short, stable hash of the resolved config.

    Fields under ``run`` are excluded, so two runs that differ only in name,
    notes or tags share a fingerprint. That is what makes it usable for
    detecting accidental duplicate experiments in a sweep.

    Args:
        cfg: A composed config.
        length: Number of hex characters to return.

    Returns:
        A truncated SHA-256 hex digest.
    """
    payload = to_container(cfg)
    payload.pop("run", None)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def save_config(cfg: DictConfig | TinyEarthConfig, path: Path | str) -> Path:
    """Write the fully resolved config to ``path`` as YAML.

    Args:
        cfg: A composed config.
        path: Destination file. Parent directories are created.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(to_container(cfg))
    destination.write_text(OmegaConf.to_yaml(resolved), encoding="utf-8")
    return destination
