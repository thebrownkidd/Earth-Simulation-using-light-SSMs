"""Forecast-quality and efficiency metrics.

Both families are reported automatically at the end of every run, because
efficiency is this project's primary result rather than an afterthought.

Forecast quality (all mask-aware)
    MAE, RMSE, PSNR, SSIM, SAM -- see :mod:`tinyearth.evaluation.metrics`.

Efficiency
    parameter count, FLOPs, peak GPU memory, throughput, latency -- see
    :mod:`tinyearth.evaluation.efficiency`.

Reference forecasts
    Persistence and climatology -- see :mod:`tinyearth.evaluation.references`.
    A learned MAE means nothing until it is read against these.

Visualisation
    Band composition, contrast stretching and NDVI -- see
    :mod:`tinyearth.evaluation.visualization`.
"""

from __future__ import annotations

from tinyearth.evaluation.efficiency import (
    EfficiencyReport,
    measure_flops,
    measure_latency,
    measure_peak_memory,
    measure_throughput,
    profile_model,
)
from tinyearth.evaluation.metrics import (
    MetricAccumulator,
    forecast_metrics,
    masked_mae,
    masked_psnr,
    masked_rmse,
    masked_sam,
    masked_ssim,
)
from tinyearth.evaluation.references import (
    REFERENCE_FORECASTS,
    climatology_forecast,
    persistence_forecast,
)
from tinyearth.evaluation.visualization import (
    NDVI_RANGE,
    composite_rgb,
    ndvi,
    stretch_limits,
)

__all__ = [
    "NDVI_RANGE",
    "REFERENCE_FORECASTS",
    "EfficiencyReport",
    "MetricAccumulator",
    "climatology_forecast",
    "composite_rgb",
    "forecast_metrics",
    "masked_mae",
    "masked_psnr",
    "masked_rmse",
    "masked_sam",
    "masked_ssim",
    "measure_flops",
    "measure_latency",
    "measure_peak_memory",
    "measure_throughput",
    "ndvi",
    "persistence_forecast",
    "profile_model",
    "stretch_limits",
]
