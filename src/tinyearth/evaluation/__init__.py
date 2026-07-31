"""Forecast-quality and efficiency metrics.

Both families are reported automatically at the end of every run, because
efficiency is this project's primary result rather than an afterthought.

Forecast quality (all mask-aware)
    MAE, RMSE, PSNR, SSIM, SAM -- see :mod:`tinyearth.evaluation.metrics`.

Efficiency
    parameter count, FLOPs, peak GPU memory, throughput, latency -- see
    :mod:`tinyearth.evaluation.efficiency`.
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

__all__ = [
    "EfficiencyReport",
    "MetricAccumulator",
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
    "profile_model",
]
