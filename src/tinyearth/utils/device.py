"""Device resolution and hardware reporting.

TinyEarth reports VRAM, throughput and latency as primary results, so the exact
hardware a run executed on is part of the experimental record. This module
resolves the device from config and captures that record.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import torch

__all__ = ["DeviceInfo", "describe_device", "resolve_device", "supports_amp"]


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device specification to a concrete :class:`torch.device`.

    Args:
        spec: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or ``"mps"``.
            ``"auto"`` prefers CUDA, then Apple MPS, then CPU.

    Returns:
        The resolved device. ``"cuda"`` without an index resolves to the current
        CUDA device so that logs record which GPU was actually used.

    Raises:
        RuntimeError: If a specific accelerator is requested but unavailable.
            This is deliberately loud: silently falling back to CPU would
            invalidate any efficiency measurement taken from the run.
    """
    normalised = spec.strip().lower()

    if normalised == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalised.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {spec!r} requested but CUDA is unavailable. "
                "Use device='auto' to fall back to CPU intentionally."
            )
        if normalised == "cuda":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device(normalised)

    if normalised == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"Device {spec!r} requested but MPS is unavailable.")
        return torch.device("mps")

    if normalised == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unrecognised device specification: {spec!r}.")


def supports_amp(device: torch.device) -> bool:
    """Return whether automatic mixed precision is usable on ``device``.

    Args:
        device: Target device.

    Returns:
        ``True`` for CUDA devices, ``False`` otherwise. CPU and MPS autocast
        exist but are not benchmarked in this project.
    """
    return device.type == "cuda"


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware and software provenance for a run.

    Attributes:
        device: Resolved device string, e.g. ``"cuda:0"``.
        device_name: Human-readable accelerator or CPU name.
        torch_version: Installed PyTorch version.
        cuda_version: CUDA toolkit version PyTorch was built against, if any.
        cudnn_version: cuDNN version, if available.
        total_memory_bytes: Total accelerator memory, ``None`` on CPU.
        platform: Operating system description.
        python_version: Interpreter version.
        cpu_count: Logical CPU count as seen by PyTorch.
        extra: Additional key/value provenance recorded by callers.
    """

    device: str
    device_name: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    total_memory_bytes: int | None
    platform: str
    python_version: str
    cpu_count: int
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def total_memory_gb(self) -> float | None:
        """Total accelerator memory in GiB, or ``None`` on CPU."""
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / (1024**3)

    def as_dict(self) -> dict[str, str]:
        """Flatten to a string mapping for logging and config dumps.

        Returns:
            A mapping with ``None`` values rendered as ``"n/a"``.
        """
        payload = {
            "device": self.device,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version or "n/a",
            "cudnn_version": str(self.cudnn_version) if self.cudnn_version else "n/a",
            "total_memory_gb": (
                f"{self.total_memory_gb:.2f}" if self.total_memory_gb is not None else "n/a"
            ),
            "platform": self.platform,
            "python_version": self.python_version,
            "cpu_count": str(self.cpu_count),
        }
        payload.update(self.extra)
        return payload


def describe_device(device: torch.device | str = "auto") -> DeviceInfo:
    """Collect provenance for ``device``.

    Args:
        device: A resolved device, or a specification accepted by
            :func:`resolve_device`.

    Returns:
        A populated :class:`DeviceInfo`.
    """
    resolved = resolve_device(device) if isinstance(device, str) else device

    name = platform.processor() or platform.machine() or "cpu"
    total_memory: int | None = None
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        name = properties.name
        total_memory = properties.total_memory
    elif resolved.type == "mps":
        name = "Apple Silicon (MPS)"

    # torch.backends.cudnn.version() is untyped in torch's stubs.
    cudnn_version: int | None = (
        torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        if torch.cuda.is_available()
        else None
    )

    return DeviceInfo(
        device=str(resolved),
        device_name=name,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=cudnn_version,
        total_memory_bytes=total_memory,
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=torch.get_num_threads(),
    )
