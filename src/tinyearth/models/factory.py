"""Building models from configuration.

The single translation point from a validated
:class:`~tinyearth.config.schema.ModelConfig` to a live
:class:`~tinyearth.models.forecaster.Forecaster`, so the model classes stay
independent of Hydra and remain usable from a notebook or a plain script.

The backbone is selected by **name** from
:data:`~tinyearth.models.temporal.TEMPORAL_BACKBONES`. That is the mechanism
that makes the project's central claim -- only the temporal backbone changes --
a configuration edit rather than a code edit.
"""

from __future__ import annotations

from typing import Any

from tinyearth.config.schema import LossConfig, ModelConfig
from tinyearth.models.decoders import DECODERS
from tinyearth.models.encoders import ENCODERS
from tinyearth.models.forecaster import Forecaster
from tinyearth.models.losses import LOSSES, CompositeLoss, ForecastLoss
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.utils.logging import get_logger

__all__ = ["build_backbone", "build_forecaster", "build_loss"]

logger = get_logger(__name__)


def build_backbone(name: str, latent_dim: int, **kwargs: Any) -> Any:
    """Construct a temporal backbone by registry name.

    Args:
        name: Registry key, e.g. ``"convlstm"`` or ``"transformer"``.
        latent_dim: Latent channel count; must match the encoder and decoder.
        **kwargs: Backbone-specific arguments.

    Returns:
        The constructed backbone.

    Raises:
        UnknownComponentError: If ``name`` is not registered. The message lists
            what is available, since this is almost always a config typo.
    """
    return TEMPORAL_BACKBONES.build(name, latent_dim=latent_dim, **kwargs)


def build_forecaster(cfg: ModelConfig, in_channels: int, horizon: int) -> Forecaster:
    """Construct the full model described by ``cfg``.

    Args:
        cfg: Model configuration.
        in_channels: Imagery channels, taken from the data config so that a
            channel-subset experiment cannot silently mismatch the encoder.
        horizon: Default forecast horizon, taken from the data config for the
            same reason.

    Returns:
        The assembled forecaster.

    Raises:
        ValueError: If encoder and decoder depths disagree, which would leave
            the forecast at a different resolution from the input.
    """
    if cfg.encoder.depth != cfg.decoder.depth:
        raise ValueError(
            f"encoder.depth={cfg.encoder.depth} and decoder.depth={cfg.decoder.depth} "
            "must match, or the forecast will not have the input resolution."
        )

    encoder = ENCODERS.build(
        cfg.encoder.name,
        in_channels=in_channels,
        latent_dim=cfg.latent_dim,
        base_channels=cfg.encoder.base_channels,
        depth=cfg.encoder.depth,
        norm=cfg.encoder.norm,
        activation=cfg.encoder.activation,
    )
    decoder = DECODERS.build(
        cfg.decoder.name,
        out_channels=in_channels,
        latent_dim=cfg.latent_dim,
        base_channels=cfg.decoder.base_channels,
        depth=cfg.decoder.depth,
        norm=cfg.decoder.norm,
        activation=cfg.decoder.activation,
        output_activation=cfg.decoder.output_activation,
    )
    backbone = build_backbone(
        cfg.backbone.name,
        latent_dim=cfg.latent_dim,
        **dict(cfg.backbone.kwargs),
    )

    model = Forecaster(encoder=encoder, backbone=backbone, decoder=decoder, horizon=horizon)
    breakdown = model.parameter_breakdown()
    logger.info(
        "built %r forecaster: %s parameters (%.1f%% in the backbone)",
        cfg.backbone.name,
        f"{breakdown.total:,}",
        100 * breakdown.backbone_fraction,
    )
    return model


def build_loss(cfg: LossConfig) -> ForecastLoss:
    """Construct the training objective.

    A single-term objective still goes through :class:`CompositeLoss`, so the
    trainer has one code path and per-term logging works uniformly.

    Args:
        cfg: Loss configuration mapping names to weights.

    Returns:
        The constructed loss.

    Raises:
        ValueError: If no terms are configured.
    """
    if not cfg.terms:
        raise ValueError("No loss terms configured. Set e.g. `model.loss.terms={l1: 1.0}`.")

    terms: dict[str, tuple[ForecastLoss, float]] = {}
    for name, weight in cfg.terms.items():
        kwargs = dict(cfg.kwargs.get(name, {}))
        terms[name] = (LOSSES.build(name, **kwargs), float(weight))

    logger.info("loss: %s", ", ".join(f"{name}x{weight}" for name, (_, weight) in terms.items()))
    return CompositeLoss(terms)
