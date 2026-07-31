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
    "tinyearth.cli.inspect_data",
    "tinyearth.config",
    "tinyearth.config.resolution",
    "tinyearth.config.schema",
    "tinyearth.config.store",
    "tinyearth.datasets",
    "tinyearth.datasets.earthnet2021",
    "tinyearth.datasets.factory",
    "tinyearth.datasets.loaders",
    "tinyearth.datasets.masking",
    "tinyearth.datasets.minicube",
    "tinyearth.datasets.normalization",
    "tinyearth.datasets.splits",
    "tinyearth.datasets.synthetic",
    "tinyearth.datasets.types",
    "tinyearth.datasets.windows",
    "tinyearth.evaluation",
    "tinyearth.models",
    "tinyearth.models.decoders",
    "tinyearth.models.encoders",
    "tinyearth.models.losses",
    "tinyearth.models.temporal",
    "tinyearth.training",
    "tinyearth.utils",
    "tinyearth.utils.device",
    "tinyearth.utils.logging",
    "tinyearth.utils.paths",
    "tinyearth.utils.registry",
    "tinyearth.utils.seed",
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


def test_datasets_reexports_resolve():
    import tinyearth.datasets as datasets

    for name in datasets.__all__:
        assert hasattr(datasets, name)


def test_registered_datasets_are_discoverable():
    """Configs select datasets by name, so the registry must be populated."""
    from tinyearth.datasets import DATASETS

    assert set(DATASETS.keys()) == {"earthnet2021", "synthetic"}


def test_info_report_renders():
    report = render_report("cpu")
    assert "TinyEarth" in report
    assert "torch version" in report
