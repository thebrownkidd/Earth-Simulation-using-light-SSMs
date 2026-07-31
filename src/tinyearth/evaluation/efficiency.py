"""Efficiency metrics: parameters, FLOPs, memory, throughput, latency.

These are TinyEarth's **primary results**, not diagnostics. The research
question is how small a model can get while staying competitive, so a quality
number without the matching efficiency numbers answers nothing.

Measurement discipline
----------------------
Naive timing of GPU code is wrong in two ways, and both inflate the reported
speed:

* **CUDA is asynchronous.** A kernel launch returns before the work finishes, so
  timing without :func:`torch.cuda.synchronize` measures launch overhead.
* **The first iterations are unrepresentative.** cuDNN autotuning, memory-pool
  growth and lazy module initialisation all land on the early passes.

:func:`measure_latency` and :func:`measure_throughput` handle both, with
warmup and explicit synchronisation. Latency is reported as a **median**, which
is far less sensitive to scheduler noise than a mean.

FLOPs come from :class:`torch.utils.flop_counter.FlopCounterMode`, which counts
actual dispatched operations. Note it counts multiply-accumulate as 2 FLOPs and
does not see elementwise work, so treat the figure as a comparable index across
architectures rather than an absolute cost.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from tinyearth.utils.logging import get_logger

__all__ = [
    "EfficiencyReport",
    "measure_flops",
    "measure_latency",
    "measure_peak_memory",
    "measure_throughput",
    "profile_model",
]

logger = get_logger(__name__)

DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 10


@dataclass(frozen=True)
class EfficiencyReport:
    """Efficiency measurements for one model on one device.

    Attributes:
        parameters: Trainable parameter count.
        flops_per_sample: Forward FLOPs for a single sample, or ``None`` if
            counting failed.
        peak_memory_bytes: Peak allocated memory during a forward pass. ``None``
            on CPU, where PyTorch does not track allocation this way.
        latency_ms: Median single-batch forward latency in milliseconds.
        throughput_samples_per_s: Samples per second at the measured batch size.
        batch_size: Batch size the throughput and latency were measured at.
        device: Device string.
    """

    parameters: int
    flops_per_sample: float | None
    peak_memory_bytes: int | None
    latency_ms: float
    throughput_samples_per_s: float
    batch_size: int
    device: str

    @property
    def peak_memory_mb(self) -> float | None:
        """Peak memory in MiB, or ``None`` on CPU."""
        if self.peak_memory_bytes is None:
            return None
        return self.peak_memory_bytes / (1024**2)

    def as_dict(self) -> dict[str, float]:
        """Flatten for metric tracking.

        Returns:
            Metrics under an ``efficiency/`` prefix; unavailable values omitted.
        """
        payload: dict[str, float] = {
            "efficiency/parameters": float(self.parameters),
            "efficiency/parameters_millions": self.parameters / 1e6,
            "efficiency/latency_ms": self.latency_ms,
            "efficiency/throughput_samples_per_s": self.throughput_samples_per_s,
        }
        if self.flops_per_sample is not None:
            payload["efficiency/gflops_per_sample"] = self.flops_per_sample / 1e9
        if self.peak_memory_mb is not None:
            payload["efficiency/peak_memory_mb"] = self.peak_memory_mb
        return payload

    def format_table(self) -> str:
        """Render a human-readable report.

        Returns:
            A multi-line table.
        """
        rows = [
            ("parameters", f"{self.parameters:,}"),
            ("parameters (M)", f"{self.parameters / 1e6:.3f}"),
            (
                "GFLOPs / sample",
                f"{self.flops_per_sample / 1e9:.3f}" if self.flops_per_sample else "n/a",
            ),
            (
                "peak memory (MiB)",
                f"{self.peak_memory_mb:.1f}" if self.peak_memory_mb is not None else "n/a",
            ),
            ("latency (ms)", f"{self.latency_ms:.2f}"),
            ("throughput (samp/s)", f"{self.throughput_samples_per_s:.1f}"),
            ("batch size", str(self.batch_size)),
            ("device", self.device),
        ]
        return "\n".join(f"{name:<22} {value:>16}" for name, value in rows)


def _synchronize(device: torch.device) -> None:
    """Block until queued device work has completed.

    Args:
        device: Device to synchronise.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def _inference(model: nn.Module) -> Iterator[None]:
    """Put a model in eval mode with gradients disabled, restoring afterwards.

    Args:
        model: Model to configure.

    Yields:
        ``None``.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def measure_flops(model: nn.Module, sample: torch.Tensor) -> float | None:
    """Count forward FLOPs for a single sample.

    Args:
        model: Model to profile.
        sample: A batch. FLOPs are divided by its batch size.

    Returns:
        FLOPs per sample, or ``None`` if counting is unsupported for this model.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # pragma: no cover - available in every supported torch
        logger.warning("FlopCounterMode unavailable; skipping FLOP measurement")
        return None

    batch_size = max(int(sample.shape[0]), 1)
    try:
        with _inference(model):
            counter = FlopCounterMode(display=False)
            with counter:
                model(sample)
            return counter.get_total_flops() / batch_size
    except Exception as error:
        logger.warning("FLOP counting failed (%s); reporting n/a", error)
        return None


def measure_peak_memory(model: nn.Module, sample: torch.Tensor) -> int | None:
    """Measure peak allocated memory during a forward pass.

    Args:
        model: Model to profile.
        sample: Input batch.

    Returns:
        Peak bytes, or ``None`` on non-CUDA devices.
    """
    device = sample.device
    if device.type != "cuda":
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with _inference(model):
        model(sample)
    _synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> float:
    """Measure median forward latency in milliseconds.

    Args:
        model: Model to profile.
        sample: Input batch.
        warmup: Untimed passes before measurement.
        iterations: Timed passes.

    Returns:
        Median latency in milliseconds.

    Raises:
        ValueError: If ``iterations`` is not positive.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}.")

    device = sample.device
    timings: list[float] = []

    with _inference(model):
        for _ in range(max(warmup, 0)):
            model(sample)
        _synchronize(device)

        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    return statistics.median(timings)


def measure_throughput(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> float:
    """Measure forward throughput in samples per second.

    Timed as one block rather than per-iteration, so per-call timing overhead
    does not inflate the reported cost.

    Args:
        model: Model to profile.
        sample: Input batch.
        warmup: Untimed passes before measurement.
        iterations: Timed passes.

    Returns:
        Samples per second.
    """
    device = sample.device
    batch_size = int(sample.shape[0])

    with _inference(model):
        for _ in range(max(warmup, 0)):
            model(sample)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return float("inf")
    return (batch_size * iterations) / elapsed


def profile_model(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> EfficiencyReport:
    """Run the full efficiency profile.

    Args:
        model: Model to profile.
        sample: A representative input batch, already on the target device.
        warmup: Untimed passes before each timed measurement.
        iterations: Timed passes per measurement.

    Returns:
        The assembled report.
    """
    device = sample.device
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return EfficiencyReport(
        parameters=parameters,
        flops_per_sample=measure_flops(model, sample),
        peak_memory_bytes=measure_peak_memory(model, sample),
        latency_ms=measure_latency(model, sample, warmup=warmup, iterations=iterations),
        throughput_samples_per_s=measure_throughput(
            model, sample, warmup=warmup, iterations=iterations
        ),
        batch_size=int(sample.shape[0]),
        device=str(device),
    )
