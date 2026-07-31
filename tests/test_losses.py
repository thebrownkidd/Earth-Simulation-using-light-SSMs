"""Tests for the loss system.

Masking is the load-bearing behaviour: on EarthNet2021 a large share of pixels
are cloudy, and optimising against them teaches the model to predict cloud.
The tests below pin down that masked pixels contribute *nothing*.
"""

from __future__ import annotations

import pytest
import torch

from tinyearth.models.losses import (
    LOSSES,
    CharbonnierLoss,
    CompositeLoss,
    ForecastLoss,
    L1Loss,
    L2Loss,
    expand_mask,
    masked_mean,
)

SHAPE = (2, 3, 4, 8, 8)
LOSS_NAMES = ("l1", "l2", "charbonnier")


@pytest.fixture
def prediction() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(*SHAPE)


@pytest.fixture
def target() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(*SHAPE)


class TestMaskedMean:
    def test_none_mask_is_a_plain_mean(self):
        values = torch.rand(4, 5)
        torch.testing.assert_close(masked_mean(values, None), values.mean())

    def test_masked_entries_are_excluded(self):
        values = torch.tensor([[1.0, 100.0], [1.0, 100.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        assert float(masked_mean(values, mask)) == pytest.approx(1.0)

    def test_fully_masked_returns_zero_not_nan(self):
        """A fully cloudy batch must contribute no gradient, not poison the run."""
        result = masked_mean(torch.rand(4, 4), torch.zeros(4, 4))
        assert float(result) == 0.0
        assert not torch.isnan(result)

    def test_divides_by_valid_count_not_total(self):
        """Dividing by the full count would scale the loss down with cloudiness."""
        values = torch.full((10,), 2.0)
        mask = torch.tensor([1.0] * 2 + [0.0] * 8)
        assert float(masked_mean(values, mask)) == pytest.approx(2.0)

    def test_expand_mask_broadcasts_the_channel_axis(self):
        mask = torch.ones(2, 3, 1, 8, 8)
        assert expand_mask(mask, torch.rand(*SHAPE)).shape == SHAPE

    def test_expand_mask_rejects_incompatible_shapes(self):
        with pytest.raises(ValueError, match="cannot broadcast"):
            expand_mask(torch.ones(5, 5), torch.rand(*SHAPE))

    def test_expand_mask_none_gives_all_ones(self):
        result = expand_mask(None, torch.rand(2, 2))
        torch.testing.assert_close(result, torch.ones(2, 2))


@pytest.mark.parametrize("name", LOSS_NAMES)
class TestLossContract:
    def test_is_registered(self, name):
        assert issubclass(LOSSES.get(name), ForecastLoss)

    def test_returns_a_scalar(self, name, prediction, target):
        value = LOSSES.build(name)(prediction, target)
        assert value.ndim == 0

    def test_is_zero_for_a_perfect_prediction(self, name, prediction):
        value = float(LOSSES.build(name)(prediction, prediction))
        assert value == pytest.approx(0.0, abs=2e-3)

    def test_is_positive_for_an_imperfect_prediction(self, name, prediction, target):
        assert float(LOSSES.build(name)(prediction, target)) > 0.0

    def test_masked_pixels_do_not_contribute(self, name, prediction, target):
        """Corrupting masked pixels must not change the loss at all."""
        mask = torch.ones(2, 3, 1, 8, 8)
        mask[..., :4, :] = 0.0

        loss = LOSSES.build(name)
        before = float(loss(prediction, target, mask))

        corrupted = prediction.clone()
        corrupted[..., :4, :] += 100.0
        after = float(loss(corrupted, target, mask))

        assert after == pytest.approx(before, rel=1e-5)

    def test_rejects_shape_mismatch(self, name, prediction):
        with pytest.raises(ValueError, match="does not match target shape"):
            LOSSES.build(name)(prediction, torch.rand(2, 3, 4, 4, 4))

    def test_gradients_flow(self, name, target):
        prediction = torch.rand(*SHAPE, requires_grad=True)
        LOSSES.build(name)(prediction, target).backward()
        assert prediction.grad is not None
        assert float(prediction.grad.abs().sum()) > 0


class TestLossValues:
    def test_l1_equals_mean_absolute_error(self, prediction, target):
        expected = (prediction - target).abs().mean()
        torch.testing.assert_close(L1Loss()(prediction, target), expected)

    def test_l2_equals_mean_squared_error(self, prediction, target):
        expected = (prediction - target).pow(2).mean()
        torch.testing.assert_close(L2Loss()(prediction, target), expected)

    def test_charbonnier_approaches_l1_for_large_errors(self):
        prediction = torch.zeros(1, 1, 1, 4, 4)
        target = torch.full((1, 1, 1, 4, 4), 5.0)
        charbonnier = float(CharbonnierLoss(epsilon=1e-4)(prediction, target))
        assert charbonnier == pytest.approx(float(L1Loss()(prediction, target)), rel=1e-4)

    def test_charbonnier_is_smooth_at_zero(self):
        """Unlike L1, its gradient at an exact match is finite and zero."""
        prediction = torch.zeros(1, 1, 1, 2, 2, requires_grad=True)
        CharbonnierLoss(epsilon=1e-2)(prediction, torch.zeros(1, 1, 1, 2, 2)).backward()
        assert prediction.grad is not None
        assert not torch.isnan(prediction.grad).any()

    def test_charbonnier_rejects_non_positive_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            CharbonnierLoss(epsilon=0.0)


class TestCompositeLoss:
    def test_single_term_matches_the_bare_loss(self, prediction, target):
        composite = CompositeLoss({"l1": (L1Loss(), 1.0)})
        torch.testing.assert_close(composite(prediction, target), L1Loss()(prediction, target))

    def test_weights_are_applied(self, prediction, target):
        weighted = CompositeLoss({"l1": (L1Loss(), 2.0)})
        expected = 2.0 * float(L1Loss()(prediction, target))
        assert float(weighted(prediction, target)) == pytest.approx(expected)

    def test_terms_are_summed(self, prediction, target):
        composite = CompositeLoss({"l1": (L1Loss(), 1.0), "l2": (L2Loss(), 1.0)})
        expected = float(L1Loss()(prediction, target)) + float(L2Loss()(prediction, target))
        assert float(composite(prediction, target)) == pytest.approx(expected)

    def test_per_term_values_are_exposed(self, prediction, target):
        composite = CompositeLoss({"l1": (L1Loss(), 1.0), "l2": (L2Loss(), 0.5)})
        composite(prediction, target)

        assert set(composite.last_terms) == {"l1", "l2", "total"}
        assert composite.last_terms["l1"] == pytest.approx(float(L1Loss()(prediction, target)))

    def test_rejects_empty_terms(self):
        with pytest.raises(ValueError, match="at least one term"):
            CompositeLoss({})

    def test_sub_losses_are_registered_modules(self):
        """Registration is what makes `.to(device)` and `state_dict` traverse them.

        Asserted via `named_modules` rather than `state_dict`: these losses are
        parameterless, so they contribute no entries to a state dict even when
        correctly registered.
        """
        composite = CompositeLoss({"charbonnier": (CharbonnierLoss(), 1.0)})
        names = dict(composite.named_modules())

        assert "losses.charbonnier" in names
        assert isinstance(names["losses.charbonnier"], CharbonnierLoss)

    def test_masking_propagates_to_every_term(self, prediction, target):
        mask = torch.ones(2, 3, 1, 8, 8)
        mask[..., :4, :] = 0.0
        composite = CompositeLoss({"l1": (L1Loss(), 1.0), "l2": (L2Loss(), 1.0)})

        before = float(composite(prediction, target, mask))
        corrupted = prediction.clone()
        corrupted[..., :4, :] += 50.0

        assert float(composite(corrupted, target, mask)) == pytest.approx(before, rel=1e-5)


def test_registry_exposes_all_phase3_losses():
    assert set(LOSSES.keys()) == set(LOSS_NAMES)
