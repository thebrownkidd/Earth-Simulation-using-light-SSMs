"""Console logging configuration.

Scope note: this module handles *human-readable console and file logging only*.
Experiment metric tracking (TensorBoard, Weights & Biases) is a separate
concern with a separate interface, introduced in Phase 3 under
:mod:`tinyearth.training`. Keeping them apart means a metric backend can be
swapped without touching diagnostics, and that library code can log freely
without pulling in a tracking dependency.

Library modules should never call :func:`setup_logging`; they call
:func:`get_logger` and let the entry point decide on handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

__all__ = ["DEFAULT_FORMAT", "get_logger", "log_section", "setup_logging"]

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
"""Format used for plain (non-rich) handlers and for file logs."""

_ROOT_LOGGER_NAME = "tinyearth"
_NOISY_LIBRARIES = ("matplotlib", "PIL", "fsspec", "urllib3")


def _reenable_hierarchy() -> None:
    """Undo any ``disable_existing_loggers`` that silenced our loggers.

    :func:`logging.config.dictConfig` defaults to ``disable_existing_loggers:
    True``, which switches off every logger that already exists. Module-level
    ``get_logger(__name__)`` calls run at *import* time, so they are always
    already present when a framework configures logging later -- Hydra's
    ``job_logging: disabled`` profile does exactly this.

    The failure mode is nasty: no error, no warning, and every library log line
    silently absent from both console and file. Re-enabling here means
    :func:`setup_logging` produces working logging regardless of what else has
    touched the logging system first.
    """
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and (
            name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}.")
        ):
            logger.disabled = False


def setup_logging(
    level: int | str = logging.INFO,
    *,
    rich: bool = True,
    log_file: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``tinyearth`` logger hierarchy.

    Call this exactly once, from an entry point. Handlers are attached to the
    ``tinyearth`` logger rather than the root logger so that TinyEarth never
    hijacks logging for an application that imports it as a library.

    Args:
        level: Threshold for the ``tinyearth`` logger, as a level name or number.
        rich: Use :class:`rich.logging.RichHandler` for colourised console output.
            Set ``False`` for CI or when redirecting to a file, where ANSI codes
            and column-width heuristics are unhelpful.
        log_file: Optional path for a plain-text log file. Parent directories are
            created. The file handler always logs at ``DEBUG`` regardless of
            ``level``, so a captured run retains full detail.
        force: Replace existing handlers. Without this, a second call is a no-op,
            which keeps repeated entry-point invocations (e.g. Hydra multirun)
            from duplicating every line.

    Returns:
        The configured ``tinyearth`` logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    _reenable_hierarchy()

    if logger.handlers and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(level)
    # Handlers live here, so do not also emit via the root logger.
    logger.propagate = False

    console: logging.Handler
    if rich:
        console = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
            log_time_format="[%H:%M:%S]",
        )
        console.setFormatter(logging.Formatter("%(name)s | %(message)s", datefmt="[%X]"))
    else:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    console.setLevel(level)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        # The logger itself must pass DEBUG through for the file handler to see
        # it; the console handler keeps its own, stricter threshold.
        logger.setLevel(logging.DEBUG)

    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the ``tinyearth`` hierarchy.

    Args:
        name: Usually ``__name__``. A module named ``tinyearth.models.ssm`` is
            returned as-is; any other name is nested under ``tinyearth.`` so it
            inherits the project's handlers. ``None`` returns the root
            ``tinyearth`` logger.

    Returns:
        The requested logger.
    """
    if name is None:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def log_section(logger: logging.Logger, title: str, width: int = 56) -> None:
    """Log a single-line section header.

    Emitted as one record rather than three rules, because rich renders each
    record inside a narrowed message column and long rules wrap unpleasantly.

    Args:
        logger: Logger to write to.
        title: Section title.
        width: Target total width, including the title and padding dashes.
    """
    # Rendered as "--- <title> <dashes>": 4 + len(title) + 1 + padding.
    padding = max(width - len(title) - 5, 0)
    logger.info("--- %s %s", title, "-" * padding)
