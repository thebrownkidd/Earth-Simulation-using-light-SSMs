"""Determinism utilities.

Reproducibility is a stated goal of this project, so seeding is centralised
here rather than scattered across scripts.

Two levels are distinguished:

**Seeding** (:func:`seed_everything`) makes a run repeatable given identical
software, hardware and kernel selection. It is cheap and always applied.

**Deterministic algorithms** (``deterministic=True``) additionally forbids
non-deterministic CUDA kernels, which makes results bit-reproducible across
runs on the same device at a measurable throughput cost. Because TinyEarth
reports *throughput and latency* as first-class results, benchmark runs should
set ``deterministic=False`` and ``cudnn_benchmark=True``; correctness and
ablation runs should leave determinism on. See ``docs/reproducibility.md``.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "MAX_SEED",
    "RngState",
    "capture_rng_state",
    "restore_rng_state",
    "seed_everything",
    "seeded_generator",
    "temporary_seed",
    "worker_init_fn",
]

MAX_SEED = 2**32 - 1
"""Largest seed accepted by NumPy's legacy generator; the binding constraint."""

_CUBLAS_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"


def _validate(seed: int) -> int:
    """Validate that ``seed`` lies within the supported range.

    Args:
        seed: Candidate seed.

    Returns:
        The validated seed.

    Raises:
        ValueError: If the seed is outside ``[0, MAX_SEED]``.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}.")
    return seed


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> int:
    """Seed Python, NumPy and PyTorch, and configure determinism.

    Args:
        seed: Base seed, in ``[0, MAX_SEED]``.
        deterministic: If ``True``, request deterministic kernels via
            :func:`torch.use_deterministic_algorithms` (``warn_only=True``, so
            an op without a deterministic implementation warns instead of
            raising) and disable cuDNN autotuning.
        cudnn_benchmark: Enable cuDNN autotuning. Faster for fixed input shapes
            but non-deterministic; ignored when ``deterministic`` is ``True``.

    Returns:
        The seed that was applied, so callers can log it.

    Raises:
        ValueError: If ``seed`` is out of range.
    """
    _validate(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic cuBLAS GEMMs; must be set before the first
        # CUDA context is created, hence the early assignment here.
        os.environ.setdefault(_CUBLAS_ENV_VAR, ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = cudnn_benchmark

    return seed


def seeded_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Create an explicitly seeded :class:`torch.Generator`.

    Prefer passing one of these to :class:`~torch.utils.data.DataLoader` and to
    stochastic ops over relying on global RNG state, so that shuffling order is
    independent of how many random numbers the model happened to draw.

    Args:
        seed: Seed for the generator.
        device: Device the generator lives on.

    Returns:
        A seeded generator.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(_validate(seed))
    return generator


def worker_init_fn(worker_id: int) -> None:
    """Seed per-worker RNGs for :class:`~torch.utils.data.DataLoader`.

    PyTorch seeds each worker's ``torch`` RNG automatically but leaves the
    ``random`` and ``numpy`` RNGs identical across workers, which silently
    duplicates augmentations. Pass this as ``worker_init_fn`` to fix that.

    Args:
        worker_id: Index of the worker process, supplied by PyTorch.
    """
    base_seed = torch.initial_seed() % (MAX_SEED + 1)
    worker_seed = (base_seed + worker_id) % (MAX_SEED + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class RngState:
    """Snapshot of every RNG stream TinyEarth controls.

    Fields carry a ``_state`` suffix so they do not shadow the ``random``,
    ``numpy`` and ``torch`` module names inside the class body, which would
    otherwise make the annotations here unresolvable.

    Attributes:
        python_state: State of the :mod:`random` module.
        numpy_state: State of the NumPy legacy global generator.
        torch_state: State of the CPU torch generator.
        cuda_states: Per-device CUDA generator states, empty without CUDA.
    """

    python_state: object
    numpy_state: object
    torch_state: torch.Tensor
    cuda_states: tuple[torch.Tensor, ...]


def capture_rng_state() -> RngState:
    """Capture the current global RNG state.

    Returns:
        A snapshot suitable for :func:`restore_rng_state`, e.g. for checkpoint
        save/resume or for making evaluation independent of training RNG draws.
    """
    return RngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(),
        cuda_states=tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def restore_rng_state(state: RngState) -> None:
    """Restore a snapshot produced by :func:`capture_rng_state`.

    Args:
        state: Previously captured RNG state.
    """
    random.setstate(state.python_state)  # type: ignore[arg-type]
    np.random.set_state(state.numpy_state)  # type: ignore[arg-type]
    torch.set_rng_state(state.torch_state)
    if state.cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(state.cuda_states))


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Seed the global RNGs inside the block, restoring prior state on exit.

    Useful for making a deterministic sub-computation (a fixed validation
    sample, a debug batch) without perturbing the training RNG stream.

    Args:
        seed: Seed applied for the duration of the block.

    Yields:
        ``None``.
    """
    saved = capture_rng_state()
    try:
        _validate(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng_state(saved)
