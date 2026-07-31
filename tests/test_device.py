"""Tests for device resolution and hardware reporting."""

from __future__ import annotations

import pytest
import torch

from tinyearth.utils.device import describe_device, resolve_device, supports_amp


def test_auto_resolves_to_an_available_device():
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_cpu_is_always_available():
    assert resolve_device("cpu") == torch.device("cpu")


def test_specification_is_case_insensitive():
    assert resolve_device("CPU") == torch.device("cpu")


def test_whitespace_is_tolerated():
    assert resolve_device("  cpu  ") == torch.device("cpu")


def test_unknown_specification_raises():
    with pytest.raises(ValueError, match="Unrecognised device"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a machine without CUDA")
def test_requesting_cuda_without_cuda_raises_rather_than_falling_back():
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_resolves_with_an_explicit_index():
    assert resolve_device("cuda").index is not None


def test_amp_is_only_supported_on_cuda():
    assert supports_amp(torch.device("cpu")) is False
    assert supports_amp(torch.device("cuda", 0)) is True


def test_describe_device_reports_the_resolved_device():
    info = describe_device("cpu")
    assert info.device == "cpu"
    assert info.torch_version == torch.__version__
    assert info.cpu_count >= 1


def test_describe_device_accepts_a_torch_device():
    assert describe_device(torch.device("cpu")).device == "cpu"


def test_cpu_reports_no_accelerator_memory():
    assert describe_device("cpu").total_memory_gb is None


def test_as_dict_renders_missing_values_as_na():
    payload = describe_device("cpu").as_dict()
    assert payload["total_memory_gb"] == "n/a"
    assert set(payload) >= {"device", "device_name", "torch_version", "platform"}


def test_as_dict_values_are_all_strings():
    assert all(isinstance(value, str) for value in describe_device("cpu").as_dict().values())
