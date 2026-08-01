#!/usr/bin/env python
"""Render forecasts as pictures.

Produces the qualitative figures from trained checkpoints:

``forecast-<scene>.png``
    A filmstrip. What the model saw, what actually happened, what it predicted,
    and where it was wrong -- across the full 100-day horizon.

``ndvi-<scene>.png``
    The vegetation story. NDVI maps for truth and prediction, plus the scene's
    mean-greenness trajectory with every model and reference forecast on one
    axis. This is the figure that shows whether a model predicted the *season*
    rather than merely the average brightness of a landscape.

Trained on crops, rendered on whole scenes
------------------------------------------
Training uses 32x32 crops for cost reasons, but these figures are rendered at
the full 128x128 minicube size. That is not a trick: the encoder and decoder
are fully convolutional and the temporal backbone folds the latent grid into
the batch dimension, so no parameter in the model depends on height or width. A
checkpoint trained on crops therefore runs unchanged on whole scenes, and this
script exercises that directly.

Scene selection
---------------
Scenes are ranked by how much of the target window is cloud-free and the
clearest are drawn. This is stated on every figure. It is a presentational
choice, not a quantitative one -- the metrics in ``summary.json`` are computed
over the whole validation split, cloud and all, and are unaffected.

Requires the ``notebooks`` extra for matplotlib.

Usage:
    python scripts/visualize_forecasts.py
    python scripts/visualize_forecasts.py --group earthnet --scenes 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from tinyearth.config.resolution import from_container, resolve_paths
from tinyearth.datasets.factory import build_dataset
from tinyearth.datasets.splits import Split
from tinyearth.evaluation.references import REFERENCE_FORECASTS
from tinyearth.evaluation.visualization import NDVI_RANGE, composite_rgb, ndvi, stretch_limits
from tinyearth.models.factory import build_forecaster
from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root

logger = get_logger("tinyearth.scripts.visualize")

PALETTE = {
    "light": {
        "s4d": "#2a78d6",
        "mamba": "#eb6834",
        "transformer": "#1baf7a",
        "convlstm": "#4a3aa7",
        "persistence": "#8a8880",
        "climatology": "#a8a49a",
        "truth": "#0b0b0b",
    },
    "dark": {
        "s4d": "#3987e5",
        "mamba": "#d95926",
        "transformer": "#199e70",
        "convlstm": "#9085e9",
        "persistence": "#8a8880",
        "climatology": "#5f5e58",
        "truth": "#ffffff",
    },
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
LABEL = {
    "s4d": "S4D (SSM)",
    "mamba": "Mamba (SSM)",
    "transformer": "Transformer",
    "convlstm": "ConvLSTM",
    "persistence": "Persistence",
    "climatology": "Climatology",
}
ORDER = ("s4d", "mamba", "transformer", "convlstm")

REVISIT_DAYS = 5
"""Sentinel-2 revisit interval in the EarthNet2021 minicubes."""


@dataclass(frozen=True)
class Scene:
    """One validation window held in memory for rendering.

    Attributes:
        cube_id: Source cube identifier.
        images: Observed history, ``[T, C, H, W]``.
        target: Observed future, ``[K, C, H, W]``.
        target_mask: Validity of the future, ``[K, 1, H, W]``.
        images_mask: Validity of the history, ``[T, 1, H, W]``.
    """

    cube_id: str
    images: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    images_mask: torch.Tensor

    @property
    def valid_fraction(self) -> float:
        """Share of future pixels that are cloud-free."""
        return float(self.target_mask.mean())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--group", default="earthnet", help="Run group under outputs/.")
    parser.add_argument("--outputs", type=Path, default=None, help="Run output root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where figures are written.")
    parser.add_argument("--scenes", type=int, default=2, help="Scenes to render.")
    parser.add_argument(
        "--candidates",
        type=int,
        default=60,
        help="Validation windows scanned when ranking scenes by clarity.",
    )
    return parser.parse_args(argv)


def load_checkpoint(path: Path) -> tuple[Any, Any]:
    """Load a checkpoint and rebuild its model.

    Args:
        path: Path to a ``.ckpt`` written by the trainer.

    Returns:
        ``(model, config)`` with the model in eval mode.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_container(payload["config"])
    if cfg.model is None or cfg.data is None:  # pragma: no cover - defensive
        raise ValueError(f"{path} carries no model/data config.")

    channels = len(cfg.data.channels) if cfg.data.channels else 4
    model = build_forecaster(cfg.model, in_channels=channels, horizon=cfg.data.horizon)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def discover_runs(root: Path, group: str) -> dict[str, Path]:
    """Find the best checkpoint of every backbone in a run group.

    Args:
        root: Output root, normally ``outputs/``.
        group: Run group name.

    Returns:
        Backbone name to checkpoint path, ordered as :data:`ORDER`.
    """
    found: dict[str, Path] = {}
    for backbone in ORDER:
        checkpoint = root / group / backbone / "best.ckpt"
        if checkpoint.is_file():
            found[backbone] = checkpoint
        else:
            logger.warning("no checkpoint for %s at %s", backbone, checkpoint)
    return found


def collect_scenes(cfg: Any, candidates: int, wanted: int) -> list[Scene]:
    """Draw the clearest validation scenes, at full resolution.

    Args:
        cfg: A resolved config, used for the dataset settings.
        candidates: Windows to scan.
        wanted: Scenes to return.

    Returns:
        The clearest scenes, most cloud-free first.
    """
    # crop_size=None is the point of this function: the figures are rendered on
    # whole 128x128 scenes even though the checkpoint was trained on 32x32
    # crops. Nothing in the model depends on spatial size.
    data_cfg = replace(cfg.data, crop_size=None)
    dataset = build_dataset(data_cfg, resolve_paths(cfg), split=Split.VAL)
    logger.info("scanning %d of %d validation windows", min(candidates, len(dataset)), len(dataset))

    scenes: list[Scene] = []
    for index in range(min(candidates, len(dataset))):
        try:
            sample = dataset[index]
        except (OSError, ValueError, IndexError) as error:
            logger.warning("skipping window %d: %s", index, error)
            continue
        scenes.append(
            Scene(
                cube_id=sample["metadata"].cube_id,
                images=sample["images"],
                target=sample["target"],
                target_mask=sample.get("target_mask", torch.ones_like(sample["target"][:, :1])),
                images_mask=sample.get("images_mask", torch.ones_like(sample["images"][:, :1])),
            )
        )

    scenes.sort(key=lambda scene: scene.valid_fraction, reverse=True)
    return scenes[:wanted]


def predict(models: dict[str, Any], scene: Scene, horizon: int) -> dict[str, torch.Tensor]:
    """Run every model and reference on one scene.

    Args:
        models: Backbone name to model.
        scene: Scene to forecast.
        horizon: Frames to emit.

    Returns:
        Forecast name to ``[K, C, H, W]`` prediction.
    """
    history = scene.images.unsqueeze(0)
    mask = scene.images_mask.unsqueeze(0)

    forecasts: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, model in models.items():
            forecasts[name] = model(history, horizon=horizon)[0]
        for name, reference in REFERENCE_FORECASTS.items():
            forecasts[name] = reference(history, horizon, mask=mask)[0]
    return forecasts


def _frame_axes(axis: Any, ink: dict[str, str]) -> None:
    """Strip an image axis down to its frame."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(ink["grid"])
        spine.set_linewidth(0.6)


CLOUD_THRESHOLD = 0.35
"""Below this valid fraction a frame is labelled rather than left to puzzle."""


def _mark_if_clouded(axis: Any, valid_fraction: float) -> None:
    """Label a frame that is mostly cloud.

    Sentinel-2 frames are often almost entirely obscured, and the mask policy
    blanks those pixels to zero -- so the panel renders as a black square. Left
    unlabelled that looks like a rendering fault or a dead model. Saying so
    turns it into information: it is what the forecaster had to work with.

    The label carries its own colours rather than the figure's ink tokens: it
    has to stay legible against a black frame and against a neutral grey one.

    Args:
        axis: Panel to annotate.
        valid_fraction: Share of usable pixels in the frame.
    """
    if valid_fraction >= CLOUD_THRESHOLD:
        return
    # A near-total cloud frame renders as a black square, so the label needs its
    # own contrast rather than the figure's ink colour.
    axis.text(
        0.5,
        0.5,
        f"cloud\n{100 * (1 - valid_fraction):.0f}%",
        transform=axis.transAxes,
        color="#f2f1ec",
        fontsize=7.5,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#2b2b28",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )


def render_filmstrip(
    scene: Scene,
    forecasts: dict[str, torch.Tensor],
    backbone: str,
    mode: str,
    destination: Path,
) -> Path:
    """Render the context / truth / prediction / error filmstrip.

    Args:
        scene: The scene rendered.
        forecasts: Forecasts keyed by name.
        backbone: Which model's prediction to show.
        mode: ``"light"`` or ``"dark"``.
        destination: Output path.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    ink = INK[mode]
    prediction = forecasts[backbone]
    horizon = scene.target.shape[0]
    context_steps = scene.images.shape[0]

    # One stretch for every panel, computed from the observed sequence and
    # excluding cloud. Including the zeros the mask policy writes would pin the
    # black point at 0.0 and render the whole figure washed-out grey.
    low, high = stretch_limits(
        torch.cat([scene.images, scene.target]),
        mask=torch.cat([scene.images_mask, scene.target_mask]),
    )

    leads = _pick_leads(horizon, count=6)

    # Two independent column grids. The context strip has one panel per observed
    # frame and the forecast block one per lead time; forcing them onto a shared
    # grid would either misalign the labels or leave the forecast rows padding
    # out to the context's width.
    fig = plt.figure(figsize=(2.0 * len(leads) + 1.9, 8.6), facecolor=ink["surface"])
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 3.45],
        hspace=0.20,
        left=0.082,
        right=0.988,
        top=0.877,
        bottom=0.088,
    )
    top = GridSpecFromSubplotSpec(1, context_steps, subplot_spec=outer[0], wspace=0.06)
    bottom = GridSpecFromSubplotSpec(3, len(leads), subplot_spec=outer[1], hspace=0.07, wspace=0.05)

    context_rgb = composite_rgb(scene.images, low, high)
    context_valid = scene.images_mask[:, 0].mean(dim=(-2, -1))
    for step in range(context_steps):
        axis = fig.add_subplot(top[0, step])
        axis.imshow(context_rgb[step].numpy())
        _frame_axes(axis, ink)
        day = -(context_steps - step) * REVISIT_DAYS
        axis.set_title(f"{day:+d}d", color=ink["secondary"], fontsize=7.5, pad=2)
        _mark_if_clouded(axis, float(context_valid[step]))
        if step == 0:
            axis.set_ylabel("Observed\ncontext", color=ink["primary"], fontsize=9, labelpad=8)

    truth_rgb = composite_rgb(scene.target, low, high)
    prediction_rgb = composite_rgb(prediction, low, high)

    # Error only where the truth was actually observed. Cloud is blanked to zero
    # by the mask policy, so an unmasked error panel would light up bright over
    # every cloudy region and read as catastrophic model failure.
    valid = scene.target_mask[:, 0]
    error = (prediction - scene.target).abs().mean(dim=1)
    error = error.masked_fill(valid == 0, float("nan"))
    observed = error[~torch.isnan(error)]
    error_ceiling = float(observed.quantile(0.99).clamp(min=1e-3)) if observed.numel() else 1e-3

    target_valid = valid.mean(dim=(-2, -1))
    rows = (
        (0, truth_rgb, "What happened", None),
        (1, prediction_rgb, f"{LABEL[backbone]}\npredicted", None),
        (2, error, "Absolute error", "inferno"),
    )
    for row, data, label, cmap in rows:
        for column, lead in enumerate(leads):
            axis = fig.add_subplot(bottom[row, column])
            if cmap is None:
                axis.imshow(data[lead].numpy())
            else:
                image = axis.imshow(data[lead].numpy(), cmap=cmap, vmin=0.0, vmax=error_ceiling)
                image.cmap.set_bad(ink["grid"])  # masked cloud reads as neutral, not as error
            _frame_axes(axis, ink)
            if row == 0:
                axis.set_title(
                    f"+{(lead + 1) * REVISIT_DAYS}d", color=ink["secondary"], fontsize=7.5, pad=2
                )
                _mark_if_clouded(axis, float(target_valid[lead]))
            if column == 0:
                axis.set_ylabel(label, color=ink["primary"], fontsize=9, labelpad=8)

    fig.suptitle(
        f"Forecasting {scene.cube_id.split('_')[0]} - "
        f"{context_steps * REVISIT_DAYS} days of context, {horizon * REVISIT_DAYS} days predicted",
        color=ink["primary"],
        fontsize=13,
        x=0.075,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.075,
        0.945,
        "Sentinel-2 true colour. One contrast stretch across every panel, computed from the "
        "observed frames, so the prediction cannot be flattered by rescaling.",
        color=ink["secondary"],
        fontsize=8.5,
        ha="left",
    )
    fig.text(
        0.082,
        0.018,
        f"Error is the mean absolute reflectance difference over 4 bands, scaled to "
        f"0-{error_ceiling:.02f}; cloud-masked pixels are drawn flat grey rather than scored.\n"
        f"Scene {100 * scene.valid_fraction:.0f}% cloud-free. The clearest validation scenes "
        "are shown so the comparison is visible; reported metrics use the whole split.",
        color=ink["secondary"],
        fontsize=7.5,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=155, facecolor=ink["surface"])
    plt.close(fig)
    return destination


def render_ndvi(
    scene: Scene,
    forecasts: dict[str, torch.Tensor],
    backbone: str,
    mode: str,
    destination: Path,
) -> Path:
    """Render the NDVI maps and the mean-greenness trajectory.

    Args:
        scene: The scene rendered.
        forecasts: Forecasts keyed by name.
        backbone: Which model's NDVI maps to show.
        mode: ``"light"`` or ``"dark"``.
        destination: Output path.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ink = INK[mode]
    colours = PALETTE[mode]
    horizon = scene.target.shape[0]
    context_steps = scene.images.shape[0]
    leads = _pick_leads(horizon, count=5)

    truth_ndvi = ndvi(scene.target)
    prediction_ndvi = ndvi(forecasts[backbone])
    vmin, vmax = NDVI_RANGE

    # A dedicated final column for the colourbar. Letting `fig.colorbar` borrow
    # space from the map axes shrinks them unevenly and lands the bar in the
    # middle of the figure.
    fig = plt.figure(figsize=(13.4, 8.8), facecolor=ink["surface"])
    grid = GridSpec(
        3,
        len(leads) + 1,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.6],
        width_ratios=[*([1.0] * len(leads)), 0.055],
        hspace=0.22,
        wspace=0.06,
        left=0.072,
        right=0.955,
        top=0.878,
        bottom=0.105,
    )

    valid = scene.target_mask[:, 0]
    # Fully clouded frames carry no vegetation signal -- both bands are blanked
    # to zero, so NDVI evaluates to exactly 0 and would paint the panel solid
    # orange, reading as dead ground rather than as absent data.
    truth_display = truth_ndvi.masked_fill(valid == 0, float("nan"))
    frame_valid = valid.mean(dim=(-2, -1))

    image = None
    for row, (data, label) in enumerate(
        ((truth_display, "What happened"), (prediction_ndvi, f"{LABEL[backbone]}\npredicted"))
    ):
        for column, lead in enumerate(leads):
            axis = fig.add_subplot(grid[row, column])
            image = axis.imshow(data[lead].numpy(), cmap="RdYlGn", vmin=vmin, vmax=vmax)
            image.cmap.set_bad(ink["grid"])
            _frame_axes(axis, ink)
            if row == 0:
                axis.set_title(
                    f"+{(lead + 1) * REVISIT_DAYS}d", color=ink["secondary"], fontsize=8, pad=3
                )
                _mark_if_clouded(axis, float(frame_valid[lead]))
            if column == 0:
                axis.set_ylabel(label, color=ink["primary"], fontsize=9.5, labelpad=8)

    if image is not None:
        bar_axis = fig.add_subplot(grid[0:2, len(leads)])
        bar = fig.colorbar(image, cax=bar_axis)
        bar.set_label("NDVI  (bare ground -> dense canopy)", color=ink["secondary"], fontsize=8)
        bar.ax.tick_params(colors=ink["secondary"], labelsize=7.5)
        bar.outline.set_edgecolor(ink["grid"])

    # -- the trajectory ----------------------------------------------------
    axis = fig.add_subplot(grid[2, 0 : len(leads)])
    axis.set_facecolor(ink["surface"])

    context_days = [-(context_steps - step) * REVISIT_DAYS for step in range(context_steps)]
    future_days = [(step + 1) * REVISIT_DAYS for step in range(horizon)]

    # Mask-weighted spatial mean: cloud-blanked pixels are zeros and would drag
    # observed greenness toward bare ground if simply averaged in.
    context_curve = _masked_mean(ndvi(scene.images), scene.images_mask[:, 0])
    truth_curve = _masked_mean(truth_ndvi, scene.target_mask[:, 0])

    axis.plot(
        context_days + future_days[:1],
        context_curve + truth_curve[:1],
        color=colours["truth"],
        linewidth=2.2,
        zorder=5,
    )
    axis.plot(
        future_days,
        truth_curve,
        color=colours["truth"],
        linewidth=2.4,
        label="What happened",
        zorder=5,
    )

    for name in (*ORDER, "persistence", "climatology"):
        if name not in forecasts:
            continue
        curve = _masked_mean(ndvi(forecasts[name]), scene.target_mask[:, 0])
        is_reference = name in REFERENCE_FORECASTS
        axis.plot(
            future_days,
            curve,
            color=colours[name],
            linewidth=1.6 if is_reference else 2.0,
            linestyle=":" if is_reference else ("-" if name in {"s4d", "mamba"} else "--"),
            label=LABEL[name],
            zorder=3,
        )

    axis.axvline(0.0, color=ink["grid"], linewidth=1.0, zorder=1)
    axis.annotate(
        "forecast begins",
        xy=(0.0, axis.get_ylim()[1]),
        xytext=(3.0, 0.965),
        textcoords=("data", "axes fraction"),
        color=ink["secondary"],
        fontsize=8,
        va="top",
    )
    axis.set_xlabel("days from the last observation", color=ink["secondary"], fontsize=9)
    axis.set_ylabel("mean NDVI over the scene", color=ink["secondary"], fontsize=9)
    axis.grid(True, color=ink["grid"], linewidth=0.6, alpha=0.9, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(ink["grid"])
    axis.tick_params(colors=ink["secondary"], labelsize=8.5)

    # Horizontal, below the axis: a right-hand legend would either overflow the
    # figure or steal width from the trajectory, which is the panel that matters.
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=7,
        frameon=False,
        fontsize=8.5,
        labelcolor=ink["secondary"],
        columnspacing=1.6,
        handlelength=2.4,
    )
    legend.set_zorder(6)

    fig.suptitle(
        f"Did the model predict the season? - {scene.cube_id.split('_')[0]}",
        color=ink["primary"],
        fontsize=13,
        x=0.075,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.075,
        0.945,
        "NDVI tracks live vegetation. Getting average brightness right is easy; getting the "
        "greening and browning right is the actual task.",
        color=ink["secondary"],
        fontsize=8.5,
        ha="left",
    )
    fig.text(
        0.075,
        0.022,
        f"Scene {100 * scene.valid_fraction:.0f}% cloud-free. Curves are cloud-masked spatial "
        "means. Dotted = parameter-free reference, solid = state space model, dashed = baseline.",
        color=ink["secondary"],
        fontsize=7.5,
        ha="left",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=155, facecolor=ink["surface"])
    plt.close(fig)
    return destination


MIN_VALID_FOR_CURVE = 0.05
"""Below this valid fraction a frame's scene mean is not plotted."""


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Spatial mean per frame, ignoring masked pixels.

    Frames that are essentially all cloud return NaN, which matplotlib draws as
    a gap. The alternative is worse than it sounds: the mask policy blanks
    cloudy pixels to zero, so an unguarded mean over a fully clouded frame
    returns exactly 0.0 -- and a *plausible* 0.0, since NDVI can legitimately be
    near zero over bare ground. The observed curve would then plunge to the
    floor and recover several times across a season, inventing a dramatic
    vegetation collapse out of weather.

    Args:
        values: ``[T, H, W]``.
        mask: ``[T, H, W]``, 1 where usable.

    Returns:
        One mean per frame, NaN where the frame carries too little signal.
    """
    observed = mask.sum(dim=(-2, -1))
    weighted = (values * mask).sum(dim=(-2, -1))
    average = weighted / observed.clamp(min=1.0)
    enough = mask.mean(dim=(-2, -1)) >= MIN_VALID_FOR_CURVE
    return torch.where(enough, average, torch.full_like(average, float("nan"))).tolist()


def _pick_leads(horizon: int, count: int) -> list[int]:
    """Choose evenly spaced lead times, always including the last.

    Args:
        horizon: Total forecast length.
        count: How many to show.

    Returns:
        Frame indices, ascending.
    """
    if horizon <= count:
        return list(range(horizon))
    step = (horizon - 1) / (count - 1)
    return sorted({round(index * step) for index in range(count)})


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    root = args.outputs or project_root() / "outputs"
    output_dir = args.output_dir or project_root() / "docs" / "figures"

    runs = discover_runs(root, args.group)
    if not runs:
        logger.error("No checkpoints found under %s. Train first:", root / args.group)
        logger.error("    python scripts/run_earthnet_study.py")
        return 1

    models: dict[str, Any] = {}
    reference_cfg = None
    for backbone, checkpoint in runs.items():
        logger.info("loading %s from %s", backbone, checkpoint)
        models[backbone], reference_cfg = load_checkpoint(checkpoint)

    scenes = collect_scenes(reference_cfg, args.candidates, args.scenes)
    if not scenes:
        logger.error("No validation scenes could be read.")
        return 1

    headline = next(name for name in ORDER if name in models)
    horizon = scenes[0].target.shape[0]

    for position, scene in enumerate(scenes):
        logger.info(
            "scene %d: %s (%.0f%% clear)", position, scene.cube_id, 100 * scene.valid_fraction
        )
        forecasts = predict(models, scene, horizon)
        for mode in ("light", "dark"):
            written = render_filmstrip(
                scene, forecasts, headline, mode, output_dir / f"forecast-{position}-{mode}.png"
            )
            logger.info("  wrote %s", written)
            written = render_ndvi(
                scene, forecasts, headline, mode, output_dir / f"ndvi-{position}-{mode}.png"
            )
            logger.info("  wrote %s", written)

    return 0


if __name__ == "__main__":
    sys.exit(main())
