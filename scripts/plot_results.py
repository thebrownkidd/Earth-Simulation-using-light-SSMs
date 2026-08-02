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
    axis: Any,
    endpoints: list[tuple[float, str, str]],
    ink: dict[str, str],
    anchor_x: float = 20.0,
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
        anchor_x: Data-space x where every series ends.
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


def render_mixing(record: dict[str, Any], mode: str, destination: Path) -> Path:
    """Render the isolated temporal-mixing cost against sequence length.

    Args:
        record: A ``--sweep mixing`` benchmark record.
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
    lengths = sorted({row["history"] for row in rows})

    fig, axis = plt.subplots(figsize=(7.6, 4.8), facecolor=ink["surface"])
    axis.set_facecolor(ink["surface"])
    endpoints: list[tuple[float, str, str]] = []

    for backbone in ORDER:
        picked = sorted(
            (row for row in rows if row["backbone"] == backbone),
            key=lambda row: row["history"],
        )
        if not picked:
            continue
        x = [row["history"] for row in picked]
        y = [row["latency_ms"] for row in picked]
        is_ssm = backbone in {"s4d", "mamba"}
        exponent = record["exponents"][backbone]["latency"]
        label = f"{LABEL[backbone]}  k={exponent:.2f}"
        axis.plot(
            x,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            color=colours[backbone],
            linestyle="-" if is_ssm else "--",
            zorder=3,
            markeredgecolor=ink["surface"],
            markeredgewidth=1.5,
        )
        endpoints.append((y[-1], label, colours[backbone]))

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(FixedLocator(lengths))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    axis.set_xlabel("sequence length T", color=ink["secondary"], fontsize=9)
    axis.set_ylabel("milliseconds per forward pass", color=ink["secondary"], fontsize=9)
    axis.grid(True, which="major", color=ink["grid"], linewidth=0.6, alpha=0.9, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(ink["grid"])
    axis.tick_params(colors=ink["secondary"], labelsize=8.5)
    axis.set_xlim(lengths[0] * 0.85, lengths[-1] * 4.2)

    # This figure's labels sit at T = max, unlike the tier figure.
    _draw_endpoint_labels(axis, endpoints, ink, anchor_x=float(lengths[-1]))

    # Mark where this project's sequences actually live.
    axis.axvline(8, color=ink["secondary"], linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    axis.annotate(
        "TinyEarth: T <= 8",
        xy=(8.5, axis.get_ylim()[0] * 1.35),
        color=ink["secondary"],
        fontsize=8,
        ha="left",
    )

    fig.suptitle(
        "Temporal mixing in isolation - S4D beats attention at every T, and the gap widens",
        color=ink["primary"],
        fontsize=11.5,
        x=0.012,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.012,
        0.005,
        "Backbone only, encoder/decoder excluded. hidden=128, 2 layers, 8x8 latent grid, "
        f"{record['hardware']['device']}.  k = fitted exponent of latency ~ T^k.",
        color=ink["secondary"],
        fontsize=7.5,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.935))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor=ink["surface"])
    plt.close(fig)
    return destination


def render_quality(record: dict[str, Any], mode: str, destination: Path) -> Path:
    """Render the EarthNet2021 forecast-quality figure.

    Two panels, because two different questions are being asked:

    *Left* -- error against lead time. The shape matters more than the level: a
    model that has learned dynamics should separate from persistence as the
    horizon grows, since persistence is nearly exact at short lead and decays
    steadily. A learned curve that runs parallel to persistence has learned the
    static scene and nothing else.

    *Right* -- error against cost. This is the project's actual question, and it
    is why the parameter-free references appear here as horizontal lines: they
    cost nothing, so any model above the persistence line is worse than free.

    Args:
        record: The record written by ``scripts/evaluate_earthnet.py``.
        mode: ``"light"`` or ``"dark"``.
        destination: Output PNG path.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ink = _style(mode)
    colours = dict(PALETTE[mode])
    colours.setdefault("persistence", "#8a8880")
    colours.setdefault("climatology", "#b9b6ad" if mode == "light" else "#5f5e58")

    results = record["results"]
    setup = record["setup"]
    parameters = record["parameters"]
    revisit = setup.get("revisit_days", 5)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), facecolor=ink["surface"])
    left, right = axes
    for axis in axes:
        axis.set_facecolor(ink["surface"])

    # -- left: error against lead time -------------------------------------
    for name in (*ORDER, "persistence", "climatology"):
        entry = results.get(name)
        if entry is None:
            continue
        curve = entry["mae_by_lead"]
        days = [(index + 1) * revisit for index in range(len(curve))]
        is_reference = name not in parameters
        left.plot(
            days,
            curve,
            color=colours[name],
            linewidth=1.7 if is_reference else 2.1,
            linestyle=":" if is_reference else ("-" if name in {"s4d", "mamba"} else "--"),
            marker="" if is_reference else "o",
            markersize=4,
            markeredgecolor=ink["surface"],
            markeredgewidth=1.0,
            label=_display_name(name),
            zorder=3,
        )

    left.set_xlabel("days ahead", color=ink["secondary"], fontsize=9)
    left.set_ylabel("mean absolute error (reflectance)", color=ink["secondary"], fontsize=9)
    left.set_title(
        "Error grows with lead time", color=ink["primary"], fontsize=11, loc="left", pad=10
    )
    left.legend(frameon=False, fontsize=8.5, labelcolor=ink["secondary"], loc="upper left")

    # -- right: error against cost -----------------------------------------
    # Not against parameter count: the tiers put every model within 8% of 2M by
    # construction, so that axis has no spread to show and its tick labels
    # collapse to a row of identical "2.1M"s. What actually differs at a matched
    # budget is what the budget *costs to run*, which is the project's question.
    cost = record.get("cost", {})
    plotted = [
        (name, cost[name]["latency_ms"], results[name]["mae"])
        for name in ORDER
        if name in results and cost.get(name, {}).get("latency_ms") is not None
    ]
    plotted.sort(key=lambda item: item[1])

    for rank, (name, latency, mae) in enumerate(plotted):
        right.scatter(
            latency,
            mae,
            s=120,
            color=colours[name],
            edgecolor=ink["surface"],
            linewidth=1.5,
            zorder=4,
            marker="o" if name in {"s4d", "mamba"} else "s",
        )
        # Alternate left and right by latency rank. Two backbones can land
        # within a few milliseconds of each other -- ConvLSTM and the
        # transformer differ by 17 ms here -- while their labels are ~80 ms
        # wide, so stacking them above and below is not enough separation
        # either. Placing neighbours on opposite sides always is.
        leftward = rank % 2 == 0
        right.annotate(
            f"{LABEL[name]}\n{parameters[name] / 1e6:.2f}M params",
            xy=(latency, mae),
            xytext=(-12 if leftward else 12, 0),
            textcoords="offset points",
            color=ink["secondary"],
            fontsize=8.5,
            ha="right" if leftward else "left",
            va="center",
            linespacing=1.35,
        )

    # Margins first: the reference labels are placed against the axes edge, and
    # the data points must not already be sitting there.
    right.margins(x=0.32, y=0.26)

    for name in ("persistence", "climatology"):
        entry = results.get(name)
        if entry is None:
            continue
        right.axhline(entry["mae"], color=colours[name], linewidth=1.4, linestyle=":", zorder=2)
        right.annotate(
            f"{_display_name(name)} - free, 0 parameters",
            xy=(0.988, entry["mae"]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            color=ink["secondary"],
            fontsize=8,
            ha="right",
        )

    right.set_xlabel("inference latency (ms per forward pass)", color=ink["secondary"], fontsize=9)
    right.set_ylabel("mean absolute error (reflectance)", color=ink["secondary"], fontsize=9)
    right.set_title(
        "Same budget, very different cost", color=ink["primary"], fontsize=11, loc="left", pad=10
    )
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    for axis in axes:
        axis.grid(True, color=ink["grid"], linewidth=0.6, alpha=0.9, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(ink["grid"])
        axis.tick_params(colors=ink["secondary"], labelsize=8.5)

    fig.suptitle(
        "EarthNet2021 forecast quality - "
        f"{setup['history'] * revisit} days of context, {setup['horizon'] * revisit} days predicted",
        color=ink["primary"],
        fontsize=12.5,
        x=0.008,
        ha="left",
        y=0.985,
    )
    extent = (
        "full 128x128 scenes" if not setup.get("crop_size") else f"{setup['crop_size']}px crops"
    )
    fig.text(
        0.008,
        0.008,
        f"{setup['windows']} held-out validation windows, {extent}, cloud-masked metrics. "
        "Solid = state space model, dashed = baseline, dotted = parameter-free reference.\n"
        "Every model trained on identical data for an identical budget of 6 epochs; none to "
        "convergence. Latency measured on CPU with warmup, reported as a median.",
        color=ink["secondary"],
        fontsize=7.5,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )

    fig.tight_layout(rect=(0, 0.075, 1, 0.945))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor=ink["surface"])
    plt.close(fig)
    return destination


def render_v1_vs_v2(v1: dict[str, Any], v2: dict[str, Any], mode: str, destination: Path) -> Path:
    """Render the v1-vs-v2 (skip connections + GDL loss) comparison.

    Two panels, MAE and SSIM, because the two headline findings point in
    different directions: v2's MAE moves by a few percent in either direction
    depending on the backbone, while its SSIM improves for every backbone by
    a similar, larger margin. A single combined score would average that
    contrast away.

    Each version's own persistence baseline is drawn as a reference line in
    its own colour -- the two versions are NOT scored on the same validation
    cubes (v1: 161 windows on ~1,650 cubes; v2: 312 windows on ~3,150 cubes,
    grown by an unrelated data-collection task run in the same window), so
    the baseline itself moved between versions. Showing both baselines is
    what makes that visible rather than hidden.

    Args:
        v1: The record written by ``scripts/evaluate_earthnet.py`` for v1.
        v2: The same, for v2.
        mode: ``"light"`` or ``"dark"``.
        destination: Output PNG path.

    Returns:
        The path written.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = _style(mode)
    colours = PALETTE[mode]
    backbones = [name for name in ORDER if name in v1["results"] and name in v2["results"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=ink["surface"])
    positions = np.arange(len(backbones))
    width = 0.32

    for axis, metric, title, better in (
        (left, "mae", "Mean absolute error", "lower is better"),
        (right, "ssim", "Structural similarity (SSIM)", "higher is better"),
    ):
        axis.set_facecolor(ink["surface"])
        v1_values = [v1["results"][name][metric] for name in backbones]
        v2_values = [v2["results"][name][metric] for name in backbones]
        v1_persist = v1["results"]["persistence"][metric]
        v2_persist = v2["results"]["persistence"][metric]

        # Bars start from the data's own floor, not zero. All four quantities
        # per panel (two backbones' worth of bars, two persistence lines) sit
        # within a narrow band by construction -- MAE in [0.025, 0.031], SSIM
        # in [0.73, 0.81] -- and starting from 0 would compress every bar
        # into a sliver at the top, hiding the differences this figure exists
        # to show. Every value is still labelled directly on its bar, so
        # nothing here is deceptive about the true magnitude.
        floor = min([*v1_values, *v2_values, v1_persist, v2_persist])
        ceiling = max([*v1_values, *v2_values, v1_persist, v2_persist])
        span = ceiling - floor
        axis.set_ylim(floor - 0.22 * span, ceiling + 0.30 * span)

        axis.bar(
            positions - width / 2,
            v1_values,
            width,
            color=[colours[name] for name in backbones],
            alpha=0.45,
            label="v1 (baseline)",
            edgecolor=ink["surface"],
            zorder=3,
        )
        axis.bar(
            positions + width / 2,
            v2_values,
            width,
            color=[colours[name] for name in backbones],
            alpha=1.0,
            label="v2 (skip + GDL)",
            edgecolor=ink["surface"],
            zorder=3,
        )

        for position, value in zip(positions - width / 2, v1_values, strict=True):
            axis.annotate(
                f"{value:.3f}",
                xy=(position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=ink["secondary"],
                zorder=4,
            )
        for position, value in zip(positions + width / 2, v2_values, strict=True):
            axis.annotate(
                f"{value:.3f}",
                xy=(position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=ink["primary"],
                fontweight="bold",
                zorder=4,
            )

        axis.axhline(v1_persist, color=ink["secondary"], linestyle=":", linewidth=1.3, zorder=1)
        axis.axhline(v2_persist, color=ink["primary"], linestyle=":", linewidth=1.3, zorder=1)

        # The two persistence lines land within 0.002-0.006 of each other --
        # close enough that independent labels collide. One combined label,
        # placed above whichever line is higher, reads cleanly either way.
        upper_label, lower_label = (
            (f"v2 persistence {v2_persist:.3f}", f"v1 persistence {v1_persist:.3f}")
            if v2_persist >= v1_persist
            else (f"v1 persistence {v1_persist:.3f}", f"v2 persistence {v2_persist:.3f}")
        )
        axis.annotate(
            f"{upper_label}\n{lower_label}",
            xy=(0.99, max(v1_persist, v2_persist)),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=ink["secondary"],
            linespacing=1.4,
        )

        axis.set_xticks(positions)
        axis.set_xticklabels([LABEL[name] for name in backbones], fontsize=9)
        axis.set_title(f"{title} ({better})", color=ink["primary"], fontsize=11, loc="left", pad=10)
        axis.tick_params(colors=ink["secondary"])
        axis.grid(axis="y", color=ink["grid"], linewidth=0.7, zorder=0)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(ink["grid"])

    left.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=ink["secondary"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
    )

    fig.suptitle(
        "v1 vs v2: encoder-decoder skip connections + a gradient-difference loss term",
        color=ink["primary"],
        fontsize=12.5,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.008,
        f"v1: {v1['setup']['windows']} windows on ~1,650 cubes. v2: {v2['setup']['windows']} windows "
        "on ~3,150 cubes (dataset grew between runs -- see docs/v1-vs-v2.md). Faded = v1, solid = v2. "
        "Dotted lines are each version's own free persistence baseline.",
        color=ink["secondary"],
        fontsize=7.5,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )

    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor=ink["surface"])
    plt.close(fig)
    return destination


def _display_name(name: str) -> str:
    """Return a human label for a model or reference."""
    return LABEL.get(name, name.capitalize())


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
        logger.info("wrote %s", render(record, mode, output_dir / f"efficiency-{mode}.png"))

    mixing_source = source.with_name("scaling_mixing.json")
    if mixing_source.is_file():
        mixing = json.loads(mixing_source.read_text(encoding="utf-8"))
        for mode in ("light", "dark"):
            logger.info("wrote %s", render_mixing(mixing, mode, output_dir / f"mixing-{mode}.png"))
    else:
        logger.info(
            "no mixing sweep at %s; generate with "
            "`python scripts/benchmark_scaling.py --sweep mixing`",
            mixing_source,
        )

    quality_source = source.with_name("earthnet.json")
    quality = None
    if quality_source.is_file():
        quality = json.loads(quality_source.read_text(encoding="utf-8"))
        for mode in ("light", "dark"):
            logger.info(
                "wrote %s", render_quality(quality, mode, output_dir / f"quality-{mode}.png")
            )
    else:
        logger.info(
            "no EarthNet evaluation at %s; generate with " "`python scripts/evaluate_earthnet.py`",
            quality_source,
        )

    v2_source = source.with_name("earthnet_v2.json")
    if v2_source.is_file():
        quality_v2 = json.loads(v2_source.read_text(encoding="utf-8"))
        for mode in ("light", "dark"):
            logger.info(
                "wrote %s",
                render_quality(quality_v2, mode, output_dir / f"quality-v2-{mode}.png"),
            )
        if quality is not None:
            for mode in ("light", "dark"):
                logger.info(
                    "wrote %s",
                    render_v1_vs_v2(quality, quality_v2, mode, output_dir / f"v1-vs-v2-{mode}.png"),
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
