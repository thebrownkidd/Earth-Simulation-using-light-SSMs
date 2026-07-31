"""Package-level smoke tests.

These guard the scaffold itself: that every declared subpackage imports, that
the public surface is documented, and that the CLI entry points are callable.
"""

from __future__ import annotations

import importlib

import pytest

import tinyearth
from tinyearth.cli.info import render_report

SUBPACKAGES = [
    "tinyearth.bootstrap",
    "tinyearth.cli",
    "tinyearth.cli.info",
    "tinyearth.cli.inspect_config",
    "tinyearth.config",
    "tinyearth.datasets",
    "tinyearth.evaluation",
    "tinyearth.models",
    "tinyearth.models.decoders",
    "tinyearth.models.encoders",
    "tinyearth.models.losses",
    "tinyearth.models.temporal",
    "tinyearth.training",
    "tinyearth.utils",
]


def test_version_is_exposed():
    assert isinstance(tinyearth.__version__, str)
    assert tinyearth.__version__.count(".") == 2


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_imports(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_every_module_is_documented(module_name):
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} is missing a module docstring"


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_every_module_declares_its_public_surface(module_name):
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} does not declare __all__"


def test_utils_reexports_resolve():
    import tinyearth.utils as utils

    for name in utils.__all__:
        assert hasattr(utils, name), f"tinyearth.utils.__all__ names missing attribute {name}"


def test_config_reexports_resolve():
    import tinyearth.config as config

    for name in config.__all__:
        assert hasattr(config, name)


def test_info_report_renders():
    report = render_report("cpu")
    assert "TinyEarth" in report
    assert "torch version" in report
