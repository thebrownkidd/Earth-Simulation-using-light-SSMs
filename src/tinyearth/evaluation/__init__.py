"""Forecast-quality and efficiency metrics.

Phase 3 adds forecast quality (MAE, RMSE, SSIM, PSNR, SAM) and efficiency
(parameter count, FLOPs, peak GPU memory, throughput, latency). Both
families are reported automatically at the end of every run, since
efficiency is the project's primary result rather than an afterthought.
"""

from __future__ import annotations

__all__: list[str] = []
