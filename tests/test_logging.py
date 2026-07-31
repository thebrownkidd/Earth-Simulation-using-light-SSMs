"""Tests for console logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tinyearth.utils.logging import get_logger, log_section, setup_logging


@pytest.fixture(autouse=True)
def _reset_logger():
    """Detach handlers between tests so state does not leak."""
    logger = logging.getLogger("tinyearth")
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)


def test_setup_returns_the_project_logger():
    assert setup_logging(rich=False, force=True).name == "tinyearth"


def test_setup_attaches_a_single_console_handler():
    logger = setup_logging(rich=False, force=True)
    assert len(logger.handlers) == 1


def test_repeated_setup_is_a_noop_without_force():
    setup_logging(rich=False, force=True)
    logger = setup_logging(rich=False)
    assert len(logger.handlers) == 1


def test_force_replaces_handlers_rather_than_appending():
    setup_logging(rich=False, force=True)
    logger = setup_logging(rich=False, force=True)
    assert len(logger.handlers) == 1


def test_does_not_propagate_to_the_root_logger():
    assert setup_logging(rich=False, force=True).propagate is False


def test_log_file_is_created_and_written(tmp_path: Path):
    log_file = tmp_path / "nested" / "run.log"
    logger = setup_logging(level="INFO", rich=False, log_file=log_file, force=True)
    logger.info("hello from the test")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.is_file()
    assert "hello from the test" in log_file.read_text(encoding="utf-8")


def test_file_handler_captures_debug_below_the_console_level(tmp_path: Path):
    log_file = tmp_path / "run.log"
    logger = setup_logging(level="WARNING", rich=False, log_file=log_file, force=True)
    logger.debug("debug detail")
    for handler in logger.handlers:
        handler.flush()

    assert "debug detail" in log_file.read_text(encoding="utf-8")


class TestDisabledLoggerRecovery:
    """Regression tests for library logs vanishing under `dictConfig`.

    `logging.config.dictConfig` defaults to `disable_existing_loggers: True`,
    which switches off every logger created at import time -- i.e. every
    module-level `get_logger(__name__)`. Hydra's `job_logging: disabled` profile
    does exactly this, and the symptom is total silence from library modules
    with no error anywhere.
    """

    def test_setup_reenables_a_disabled_child(self):
        child = get_logger("tinyearth.some.module")
        child.disabled = True

        setup_logging(rich=False, force=True)

        assert not child.disabled

    def test_dictconfig_disabling_is_undone(self):
        """The end-to-end symptom: a library log line must actually be emitted.

        Captured with a handler attached to the `tinyearth` logger rather than
        via caplog, because the dictConfig below also sets the *root* level to
        ERROR, which would filter caplog's root handler and mask the result.
        """
        import logging.config
        from io import StringIO

        child = get_logger("tinyearth.datasets.probe")
        logging.config.dictConfig(
            {"version": 1, "root": {"level": "ERROR"}, "disable_existing_loggers": True}
        )
        # Asserted via a local: asserting on `child.disabled` directly narrows it
        # to Literal[True] for the rest of the function, and mypy cannot see that
        # setup_logging mutates it, so it would call the remaining lines dead.
        disabled_before = child.disabled
        assert disabled_before, "precondition: dictConfig should have disabled it"

        logger = setup_logging(rich=False, force=True)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        child.info("library message")

        assert not child.disabled
        assert "library message" in stream.getvalue()

    def test_non_tinyearth_loggers_are_left_alone(self):
        """Re-enabling must not resurrect third-party loggers we did not disable."""
        other = logging.getLogger("some_third_party")
        other.disabled = True

        setup_logging(rich=False, force=True)

        assert other.disabled


def test_get_logger_nests_plain_names():
    assert get_logger("models.ssm").name == "tinyearth.models.ssm"


def test_get_logger_preserves_already_qualified_names():
    assert get_logger("tinyearth.models.ssm").name == "tinyearth.models.ssm"


def test_get_logger_without_a_name_returns_the_root_project_logger():
    assert get_logger().name == "tinyearth"


def test_level_is_applied():
    logger = setup_logging(level="WARNING", rich=False, force=True)
    assert logger.isEnabledFor(logging.WARNING)
    assert not logger.isEnabledFor(logging.DEBUG)


def test_log_section_emits_one_record_containing_the_title(caplog: pytest.LogCaptureFixture):
    logger = setup_logging(level="INFO", rich=False, force=True)
    logger.propagate = True  # let caplog observe the records
    with caplog.at_level(logging.INFO, logger="tinyearth"):
        log_section(logger, "Results", width=30)

    assert len(caplog.messages) == 1
    assert "Results" in caplog.messages[0]
    assert len(caplog.messages[0]) == 30


def test_log_section_does_not_pad_negatively_for_a_long_title(caplog: pytest.LogCaptureFixture):
    logger = setup_logging(level="INFO", rich=False, force=True)
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger="tinyearth"):
        log_section(logger, "a" * 100, width=20)

    assert caplog.messages[0] == f"--- {'a' * 100} "
