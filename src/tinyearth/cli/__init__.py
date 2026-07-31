"""Console entry points.

Entry points orchestrate; they do not implement. Each one composes a config,
delegates to :func:`tinyearth.bootstrap.initialise_run`, and calls into library
code. This keeps training logic testable without a subprocess and avoids the
monolithic-script failure mode.
"""

from __future__ import annotations

__all__: list[str] = []
