"""Tests for determinism utilities."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tinyearth.utils.seed import (
    MAX_SEED,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seeded_generator,
    temporary_seed,
    worker_init_fn,
)


def _draw() -> tuple[float, float, float]:
    """Draw one sample from each RNG stream."""
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_seed_everything_is_reproducible():
    seed_everything(123, deterministic=False)
    first = _draw()
    seed_everything(123, deterministic=False)
    assert _draw() == first


def test_different_seeds_diverge():
    seed_everything(1, deterministic=False)
    first = _draw()
    seed_everything(2, deterministic=False)
    assert _draw() != first


def test_seed_everything_returns_the_seed():
    assert seed_everything(7, deterministic=False) == 7


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
def test_out_of_range_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(seed)


def test_deterministic_flag_configures_cudnn():
    seed_everything(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_cudnn_benchmark_only_applies_when_nondeterministic():
    seed_everything(0, deterministic=False, cudnn_benchmark=True)
    assert torch.backends.cudnn.benchmark is True
    seed_everything(0, deterministic=True, cudnn_benchmark=True)
    assert torch.backends.cudnn.benchmark is False


def test_seeded_generator_is_independent_of_global_state():
    generator = seeded_generator(99)
    first = torch.rand(4, generator=generator)

    # Perturb global RNG; the explicit generator must be unaffected.
    torch.rand(1000)

    generator = seeded_generator(99)
    torch.testing.assert_close(torch.rand(4, generator=generator), first)


def test_temporary_seed_restores_previous_state():
    seed_everything(5, deterministic=False)
    before = _draw()

    seed_everything(5, deterministic=False)
    with temporary_seed(42):
        inner = _draw()
    after = _draw()

    assert after == before
    assert inner != before


def test_capture_and_restore_rng_state():
    seed_everything(11, deterministic=False)
    state = capture_rng_state()
    expected = _draw()

    _draw()  # advance the streams
    restore_rng_state(state)

    assert _draw() == expected


def test_worker_init_fn_gives_distinct_streams_per_worker():
    torch.manual_seed(0)

    worker_init_fn(0)
    worker_zero = (random.random(), float(np.random.rand()))

    worker_init_fn(1)
    worker_one = (random.random(), float(np.random.rand()))

    assert worker_zero != worker_one


def test_worker_init_fn_is_reproducible_for_a_fixed_base_seed():
    torch.manual_seed(0)
    worker_init_fn(3)
    first = (random.random(), float(np.random.rand()))

    torch.manual_seed(0)
    worker_init_fn(3)
    assert (random.random(), float(np.random.rand())) == first
