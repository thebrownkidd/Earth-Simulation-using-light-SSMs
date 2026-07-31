"""Tests for forecast-quality and efficiency metrics.

Metrics are checked against closed-form values wherever one exists, because a
metric that is merely "plausible" is worse than none -- it produces numbers that
look publishable and are wrong.
"""

from __future__ import annotations

import math

import pytest
import torch

from tinyearth.evaluation.efficiency import (
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

SHAPE = (2, 2, 4, 16, 16)


@pytest.fixture
def target() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(*SHAPE)


class TestMAEandRMSE:
    def test_mae_matches_the_closed_form(self, target):
        prediction = target + 0.25
        assert masked_mae(prediction, target) == pytest.approx(0.25, abs=1e-6)

    def test_rmse_matches_the_closed_form(self, target):
        prediction = target + 0.2
        assert masked_rmse(prediction, target) == pytest.approx(0.2, abs=1e-6)

    def test_rmse_penalises_outliers_more_than_mae(self, target):
        prediction = target.clone()
        prediction[0, 0, 0, 0, 0] += 10.0
        assert masked_rmse(prediction, target) > masked_mae(prediction, target)

    def test_perfect_prediction_scores_zero(self, target):
        assert masked_mae(target, target) == pytest.approx(0.0)
        assert masked_rmse(target, target) == pytest.approx(0.0)

    def test_masked_pixels_are_ignored(self, target):
        mask = torch.ones(2, 2, 1, 16, 16)
        mask[..., :8, :] = 0.0
        prediction = target.clone()
        prediction[..., :8, :] += 100.0

        assert masked_mae(prediction, target, mask) == pytest.approx(0.0, abs=1e-6)

    def test_fully_masked_returns_zero(self, target):
        mask = torch.zeros(2, 2, 1, 16, 16)
        assert masked_mae(target + 1, target, mask) == 0.0


class TestPSNR:
    def test_matches_the_closed_form(self, target):
        prediction = target + 0.1
        expected = 10 * math.log10(1.0 / 0.01)
        assert masked_psnr(prediction, target) == pytest.approx(expected, abs=1e-4)

    def test_perfect_prediction_is_infinite(self, target):
        assert masked_psnr(target, target) == float("inf")

    def test_higher_is_better(self, target):
        close = masked_psnr(target + 0.01, target)
        far = masked_psnr(target + 0.5, target)
        assert close > far

    def test_data_range_shifts_the_result(self, target):
        prediction = target + 0.1
        assert masked_psnr(prediction, target, data_range=2.0) > masked_psnr(prediction, target)


class TestSSIM:
    def test_identical_images_score_one(self):
        torch.manual_seed(0)
        images = torch.rand(1, 1, 1, 32, 32)
        assert masked_ssim(images, images) == pytest.approx(1.0, abs=1e-4)

    def test_uncorrelated_images_score_far_below_one(self):
        torch.manual_seed(0)
        first = torch.rand(1, 1, 1, 32, 32)
        second = torch.rand(1, 1, 1, 32, 32)
        assert masked_ssim(first, second) < 0.5

    def test_degrades_monotonically_with_noise(self):
        torch.manual_seed(0)
        images = torch.rand(1, 1, 3, 32, 32)
        light = masked_ssim(images + torch.randn_like(images) * 0.01, images)
        heavy = masked_ssim(images + torch.randn_like(images) * 0.30, images)
        assert light > heavy

    def test_returns_zero_when_smaller_than_the_window(self):
        images = torch.rand(1, 1, 1, 4, 4)
        assert masked_ssim(images, images, window_size=11) == 0.0

    def test_masking_applies_to_the_map_not_the_inputs(self):
        """Masking inputs first would let the fill value bleed through the blur.

        Corrupting a masked region must leave the score for valid pixels intact.
        """
        torch.manual_seed(0)
        target = torch.rand(1, 1, 1, 32, 32)
        mask = torch.ones(1, 1, 1, 32, 32)
        mask[..., :16, :] = 0.0

        assert masked_ssim(target, target, mask) == pytest.approx(1.0, abs=1e-4)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="SSIM"):
            masked_ssim(torch.rand(1, 1, 1, 32, 32), torch.rand(1, 1, 1, 16, 16))


class TestSAM:
    def test_identical_spectra_score_zero_degrees(self, target):
        assert masked_sam(target, target) == pytest.approx(0.0, abs=1e-3)

    def test_is_invariant_to_brightness_scaling(self, target):
        """The defining property: SAM measures spectral shape, not magnitude."""
        assert masked_sam(target * 3.0, target) == pytest.approx(0.0, abs=1e-3)

    def test_orthogonal_spectra_score_ninety_degrees(self):
        prediction = torch.zeros(1, 1, 2, 2, 2)
        target = torch.zeros(1, 1, 2, 2, 2)
        prediction[:, :, 0] = 1.0
        target[:, :, 1] = 1.0
        assert masked_sam(prediction, target) == pytest.approx(90.0, abs=1e-3)

    def test_result_is_in_degrees(self):
        prediction = torch.tensor([1.0, 0.0]).view(1, 1, 2, 1, 1)
        target = torch.tensor([1.0, 1.0]).view(1, 1, 2, 1, 1)
        assert masked_sam(prediction, target) == pytest.approx(45.0, abs=1e-3)

    def test_zero_spectra_do_not_produce_nan(self):
        zeros = torch.zeros(1, 1, 3, 4, 4)
        assert not math.isnan(masked_sam(zeros, zeros))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="SAM"):
            masked_sam(torch.rand(1, 1, 4, 8, 8), torch.rand(1, 1, 4, 4, 4))


class TestForecastMetrics:
    def test_reports_all_five(self, target):
        metrics = forecast_metrics(target + 0.1, target)
        assert set(metrics) == {"mae", "rmse", "psnr", "ssim", "sam"}

    def test_values_are_plain_floats(self, target):
        metrics = forecast_metrics(target + 0.1, target)
        assert all(isinstance(value, float) for value in metrics.values())

    def test_mask_is_threaded_through_every_metric(self, target):
        mask = torch.ones(2, 2, 1, 16, 16)
        mask[..., :8, :] = 0.0
        prediction = target.clone()
        prediction[..., :8, :] += 100.0

        metrics = forecast_metrics(prediction, target, mask)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["sam"] == pytest.approx(0.0, abs=1e-3)


class TestMetricAccumulator:
    def test_weights_by_sample_count(self):
        """Averaging per-batch means would over-weight a small final batch."""
        accumulator = MetricAccumulator()
        accumulator.update({"mae": 1.0}, weight=9.0)
        accumulator.update({"mae": 11.0}, weight=1.0)
        assert accumulator.compute()["mae"] == pytest.approx(2.0)

    def test_empty_accumulator_computes_nothing(self):
        assert MetricAccumulator().compute() == {}

    def test_infinite_values_are_skipped(self):
        """A perfect-match PSNR of inf would destroy the running mean."""
        accumulator = MetricAccumulator()
        accumulator.update({"psnr": float("inf"), "mae": 1.0}, weight=1.0)
        accumulator.update({"psnr": 20.0, "mae": 1.0}, weight=1.0)

        computed = accumulator.compute()
        assert math.isfinite(computed["psnr"])
        assert computed["mae"] == pytest.approx(1.0)

    def test_nan_values_are_skipped(self):
        accumulator = MetricAccumulator()
        accumulator.update({"x": float("nan")}, weight=1.0)
        accumulator.update({"x": 4.0}, weight=1.0)
        assert accumulator.compute()["x"] == pytest.approx(2.0)

    def test_zero_weight_is_ignored(self):
        accumulator = MetricAccumulator()
        accumulator.update({"mae": 5.0}, weight=0.0)
        assert accumulator.compute() == {}

    def test_reset_clears_state(self):
        accumulator = MetricAccumulator()
        accumulator.update({"mae": 1.0}, weight=1.0)
        accumulator.reset()
        assert accumulator.compute() == {}


class TestEfficiency:
    @pytest.fixture
    def model(self) -> torch.nn.Module:
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 8))

    @pytest.fixture
    def sample(self) -> torch.Tensor:
        return torch.rand(4, 16)

    def test_flops_are_counted(self, model, sample):
        flops = measure_flops(model, sample)
        assert flops is not None
        # A 16x8 matmul is 128 MACs; the counter reports 2 FLOPs per MAC.
        assert flops == pytest.approx(256.0)

    def test_peak_memory_is_none_on_cpu(self, model, sample):
        assert measure_peak_memory(model, sample) is None

    def test_latency_is_positive(self, model, sample):
        assert measure_latency(model, sample, warmup=1, iterations=3) > 0

    def test_latency_rejects_zero_iterations(self, model, sample):
        with pytest.raises(ValueError, match="iterations must be"):
            measure_latency(model, sample, iterations=0)

    def test_throughput_is_positive(self, model, sample):
        assert measure_throughput(model, sample, warmup=1, iterations=3) > 0

    def test_profile_populates_the_report(self, model, sample):
        report = profile_model(model, sample, warmup=1, iterations=3)
        assert report.parameters == 16 * 8 + 8
        assert report.batch_size == 4
        assert report.device == "cpu"
        assert report.latency_ms > 0

    def test_report_flattens_for_tracking(self, model, sample):
        payload = profile_model(model, sample, warmup=1, iterations=3).as_dict()
        assert "efficiency/parameters" in payload
        assert "efficiency/latency_ms" in payload
        assert all(isinstance(value, float) for value in payload.values())

    def test_report_omits_unavailable_memory_on_cpu(self, model, sample):
        payload = profile_model(model, sample, warmup=1, iterations=3).as_dict()
        assert "efficiency/peak_memory_mb" not in payload

    def test_report_table_renders(self, model, sample):
        table = profile_model(model, sample, warmup=1, iterations=3).format_table()
        assert "parameters" in table
        assert "latency (ms)" in table

    def test_profiling_restores_training_mode(self, model, sample):
        model.train()
        profile_model(model, sample, warmup=1, iterations=2)
        assert model.training is True
