"""TinyEarth test suite.

This file makes ``tests`` a package so that mypy resolves modules as
``tests.test_*`` and the per-module override in ``pyproject.toml`` applies. It
also keeps test module basenames from colliding as the suite grows.
"""
