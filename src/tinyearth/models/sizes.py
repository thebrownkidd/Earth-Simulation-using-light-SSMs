"""Calibrated model size tiers.

The scaling study compares backbones at **matched parameter budgets**. Different
architectures reach a given budget at very different widths, so the width for
each ``(backbone, tier)`` pair is calibrated here rather than guessed in YAML.

Design of the tiers
-------------------
Every tier fixes ``n_layers = 4`` and varies **only** ``hidden_dim``. Letting
depth and width move together would confound the scaling result: a difference
between 2M and 20M would be attributable to neither.

Calibration
-----------
Values were measured against the real model, with ``latent_dim=128`` and the
standard encoder/decoder (``base_channels=32, depth=2``, 228,548 parameters).
Each lands within about 5% of its nominal target. Reproduce or recalibrate with::

    python scripts/calibrate_sizes.py

The tolerance matters more than the exactness: a "2M model" that is really 2.08M
is fine, but the tiers must be *comparable across backbones*, which is what
:func:`resolve_hidden_dim` and the tests in ``tests/test_sizes.py`` protect.

What the table already shows
----------------------------
To reach 2M parameters the ConvLSTM affords ``hidden_dim=80``; S4D affords 272.
A ConvLSTM spends ``4·9·(H+H)·H`` parameters on its gate convolution, while a
diagonal SSM spends ``≈5H² + 4HN`` -- so at a matched budget the SSM gets roughly
3.4x the width. That difference *is* the parameter-efficiency question this
project asks, and it is visible before a single training step.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "SIZE_TIERS",
    "SIZE_TIERS_SKIP",
    "TARGET_PARAMETERS",
    "available_sizes",
    "resolve_hidden_dim",
]

TARGET_PARAMETERS: Final[dict[str, int]] = {
    "tiny": 2_000_000,
    "small": 5_000_000,
    "base": 10_000_000,
    "large": 20_000_000,
}
"""Nominal total-parameter target for each tier."""

SIZE_TIERS: Final[dict[str, dict[str, int]]] = {
    # backbone -> tier -> hidden_dim, at n_layers=4 and latent_dim=128.
    "s4d": {"tiny": 272, "small": 448, "base": 672, "large": 960},
    "mamba": {"tiny": 256, "small": 416, "base": 608, "large": 880},
    "convlstm": {"tiny": 80, "small": 128, "base": 192, "large": 272},
    "transformer": {"tiny": 128, "small": 208, "base": 288, "large": 416},
}
"""Calibrated ``hidden_dim`` per backbone and tier.

Measured, not estimated. See the module docstring for the calibration setup and
``tests/test_sizes.py`` for the check that keeps them honest.
"""

SIZE_TIERS_SKIP: Final[dict[str, dict[str, int]]] = {
    "s4d": {"tiny": 264, "small": 456, "base": 664, "large": 960},
    "mamba": {"tiny": 248, "small": 424, "base": 608, "large": 872},
    "convlstm": {"tiny": 72, "small": 128, "base": 184, "large": 272},
    "transformer": {"tiny": 120, "small": 200, "base": 296, "large": 416},
}
"""Calibrated ``hidden_dim`` for the encoder/decoder built with
``skip_connections=True``, which adds 43,232 fixed parameters (228,548 ->
271,780 at the standard ``base_channels=32, depth=2``) that :data:`SIZE_TIERS`
was not calibrated against. Reusing :data:`SIZE_TIERS` for a skip-connection
model would silently target the wrong parameter budget -- not by a lot (a few
percent at the "tiny" tier, less at larger ones), but enough that a "matched
2M" comparison would not, in fact, be matched: at :data:`SIZE_TIERS`'
``convlstm/tiny`` width of 80, the skip-connection total overshoots 2M by
13.2%, outside the tolerance ``tests/test_sizes.py`` enforces.

Calibrated at ``--step 8`` rather than the default 16 -- the fixed overhead is
a big enough share of the "tiny" budget that the default step's coarser grid
missed a close-enough width for ``convlstm``.

Recalibrate with::

    python scripts/calibrate_sizes.py --skip-connections --step 8 --emit-python
"""

CALIBRATION_LAYERS: Final = 4
"""Depth the tiers were calibrated at; the tier widths assume it."""

CALIBRATION_LATENT_DIM: Final = 128
"""Latent dimension the tiers were calibrated at."""


def available_sizes() -> tuple[str, ...]:
    """Return the tier names, smallest first."""
    return tuple(TARGET_PARAMETERS)


def resolve_hidden_dim(backbone: str, size: str, skip_connections: bool = False) -> int:
    """Look up the calibrated width for a backbone at a size tier.

    Args:
        backbone: Registry key, e.g. ``"s4d"``.
        size: Tier name from :func:`available_sizes`.
        skip_connections: Look up :data:`SIZE_TIERS_SKIP` instead of
            :data:`SIZE_TIERS`. Must match whether the model this width feeds
            was (or will be) built with ``model.skip_connections=True`` -- the
            two tables are calibrated against encoder/decoder pairs of
            different fixed size, so using the wrong one targets the wrong
            parameter budget.

    Returns:
        The calibrated ``hidden_dim``.

    Raises:
        KeyError: If the backbone or tier has no calibration. Adding a backbone
            means adding a row here, and the message says so -- silently guessing
            a width would produce a scaling study whose points are not comparable.
    """
    table = SIZE_TIERS_SKIP if skip_connections else SIZE_TIERS
    table_name = "SIZE_TIERS_SKIP" if skip_connections else "SIZE_TIERS"
    script_flag = " --skip-connections" if skip_connections else ""

    if backbone not in table:
        known = ", ".join(sorted(table))
        raise KeyError(
            f"No size calibration for backbone {backbone!r} in {table_name}. Calibrated: "
            f"{known}. Run `python scripts/calibrate_sizes.py{script_flag}` and add a row."
        )
    tiers = table[backbone]
    if size not in tiers:
        known = ", ".join(tiers)
        raise KeyError(f"Unknown size {size!r} for {backbone!r}. Available: {known}.")
    return tiers[size]
