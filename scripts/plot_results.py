#!/usr/bin/env python
"""Plot benchmark results.

Renders the cost comparison produced by ``scripts/benchmark_efficiency.py`` into
``docs/figures/``, in both light and dark variants so a README can serve the
right one via ``prefers-color-scheme``.

Design notes, since a chart is read by people:

* **Two panels, one measure each.** Never a dual y-axis -- two scales on one
  frame make the reader invent a relationship that is not in the data.
* **Every series is direct-labelled** as well as coloured, so identity never
  depends on colour alone.
* Palette is the validated categorical set; adjacent pairs clear the
  colour-vision-deficiency separation threshold.
* Log-scaled x, because the size tiers are roughly geometric (2/5/10/20M).

Requires the ``notebooks`` extra for matplotlib.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --input experiments/results/efficiency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root

logger = get_logger("tinyearth.scripts.plot")

# Validated categorical palette: light and dark steps of the same four hues.
PALETTE = {
    "light": {
        "s4d": "#2a78d6",
        "mamba": "#eb6834",
        "transformer": "#1baf7a",
        "convlstm": "#4a3aa7",
    },
    "dark": {"s4d": "#3987e5", "mamba": "#d95926", "transformer": "#199e70", "convlstm": "#9085e9"},
}
INK = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#dcdbd6",
    },
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7", "grid": "#3a3a38"},
}
ORDER = ("s4d", "mamba", "transformer", "convlstm")
LABEL = {
    "s4d": "S4D (SSM)",
    "mamba": "Mamba (SSM)",
    "transformer": "Transformer",
    "convlstm": "ConvLSTM",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _style(mode: str) -> dict[str, str]:
    """Return the ink tokens for a mode."""
    return INK[mode]


def _series(rows: list[dict[str, Any]], backbone: str, key: str) -> tuple[list[float], list[float]]:
    """Extract (parameters, measure) for one backbone, ordered by size.

    Args:
        rows: Benchmark rows.
        backbone: Backbone name.
        key: Measure to extract.

    Returns:
        ``(x, y)`` with parameters in millions.
    """
    picked = [row for row in rows if row["backbone"] == backbone and row.get(key) is not None]
    picked.sort(key=lambda row: row["parameters"])
    return (
        [row["parameters"] / 1e6 for row in picked],
        [float(row[key]) for row in picked],
    )


def _draw_endpoint_labels(
    axis: Any, endpoints: list[tuple[float, str, str]], ink: dict[str, str]
) -> None:
    """Direct-label each series at its right-hand end, without overlaps.

    Series that finish close together would otherwise print on top of each
    other -- ConvLSTM, Mamba and S4D land within a few percent on the compute
    panel. Labels are pushed apart in *display* space, then converted back, so
    the minimum gap is a real pixel distance rather than a data-space guess.

    Args:
        axis: Target axes.
        endpoints: ``(y_value, label, colour)`` per series.
        ink: Ink tokens for the current mode.
    """
    if not endpoints:
        return

    minimum_gap = 13.0  # points, a little over one line of 8.5pt text
    ordered = sorted(endpoints, key=lambda item: item[0])

    # Work in display coordinates so the gap is uniform on a log axis.
    right = axis.get_xlim()[1]
    positions = [axis.transData.transform((right, value))[1] for value, _, _ in ordered]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)

    inverse = axis.transData.inverted()
    anchor_x = 20.0  # the largest tier, where every series ends

    for (value, label, _colour), display_y in zip(ordered, positions, strict=True):
        _, data_y = inverse.transform((0.0, display_y))
        axis.annotate(
            label,
            xy=(anchor_x, value),
            xytext=(anchor_x * 1.12, data_y),
            color=ink["secondary"],
            fontsize=8.5,
            va="center",
            ha="left",
            annotation_clip=False,
        )


def render(record: dict[str, Any], mode: str, destination: Path) -> Path:
    """Render the two-panel cost figure.

    Args:
        record: Benchmark record.
        mode: ``"light"`` or ``"dark"``.
        destination: Output PNG path.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ink = _style(mode)
    colours = PALETTE[mode]
    rows = record["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=ink["surface"])
    panels = [
        ("gflops_per_sample", "Compute", "GFLOPs per sample", [2, 5, 10, 20, 50]),
        ("latency_ms", "Latency", "milliseconds per forward pass", [100, 300, 1000, 3000]),
    ]

    for axis, (key, title, ylabel, yticks) in zip(axes, panels, strict=True):
        axis.set_facecolor(ink["surface"])
        endpoints: list[tuple[float, str, str]] = []

        for backbone in ORDER:
            x, y = _series(rows, backbone, key)
            if not x:
                continue
            is_ssm = backbone in {"s4d", "mamba"}
            axis.plot(
                x,
                y,
                marker="o",
                markersize=6,
                linewidth=2.0,
                color=colours[backbone],
                linestyle="-" if is_ssm else "--",  # secondary encoding, not colour alone
                label=LABEL[backbone],
                zorder=3,
                markeredgecolor=ink["surface"],
                markeredgewidth=1.5,
            )
            endpoints.append((y[-1], LABEL[backbone], colours[backbone]))

        axis.set_xscale("log")
        axis.set_yscale("log")

        # Explicit ticks. The default log formatter renders 2x10^0 style labels
        # that collide with each other across a two-decade range.
        axis.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}M"))
        axis.yaxis.set_major_locator(FixedLocator(yticks))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

        axis.set_xlabel("total parameters", color=ink["secondary"], fontsize=9)
        axis.set_ylabel(ylabel, color=ink["secondary"], fontsize=9)
        axis.set_title(title, color=ink["primary"], fontsize=11, loc="left", pad=10)

        axis.grid(True, which="major", color=ink["grid"], linewidth=0.6, alpha=0.9, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(ink["grid"])
        axis.tick_params(colors=ink["secondary"], labelsize=8.5)

        axis.set_xlim(1.7, 20 * 3.4)  # headroom for the direct labels
        _draw_endpoint_labels(axis, endpoints, ink)

    setup = record["setup"]
    hardware = record["hardware"]
    fig.suptitle(
        "Cost at matched parameter budgets - the SSM buys width, and pays for it in compute",
        color=ink["primary"],
        fontsize=12.5,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.005,
        f"{setup['image_size']}x{setup['image_size']} input, T={setup['history']} -> K="
        f"{setup['horizon']}, {setup['n_layers']} layers, batch {setup['batch_size']}, "
        f"{hardware['device']}.  Solid = state space model, dashed = baseline.  "
        "Cost only - no forecast-quality claim.",
        color=ink["secondary"],
        fontsize=7.5,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor=ink["surface"])
    plt.close(fig)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code; 1 when the input is missing.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    source = args.input or (project_root() / "experiments" / "results" / "efficiency.json")
    if not source.is_file():
        logger.error(
            "no benchmark at %s. Generate it first:\n" "    python scripts/benchmark_efficiency.py",
            source,
        )
        return 1

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.error('matplotlib is required. Install with: pip install -e ".[notebooks]"')
        return 1

    record = json.loads(source.read_text(encoding="utf-8"))
    output_dir = args.output_dir or (project_root() / "docs" / "figures")

    for mode in ("light", "dark"):
        path = render(record, mode, output_dir / f"efficiency-{mode}.png")
        logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
