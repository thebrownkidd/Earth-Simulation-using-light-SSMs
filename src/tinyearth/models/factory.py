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

import inspect
from typing import Any

from tinyearth.config.schema import LossConfig, ModelConfig
from tinyearth.models.decoders import DECODERS
from tinyearth.models.encoders import ENCODERS
from tinyearth.models.forecaster import Forecaster
from tinyearth.models.losses import LOSSES, CompositeLoss, ForecastLoss
from tinyearth.models.sizes import resolve_hidden_dim
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.utils.logging import get_logger

__all__ = ["build_backbone", "build_forecaster", "build_loss"]

logger = get_logger(__name__)


def _accepted_arguments(name: str) -> set[str]:
    """Return the keyword arguments a registered backbone accepts.

    Args:
        name: Registry key.

    Returns:
        Parameter names of the class constructor, excluding ``self``.
    """
    signature = inspect.signature(TEMPORAL_BACKBONES.get(name).__init__)
    return {parameter for parameter in signature.parameters if parameter != "self"}


def _partition_kwargs(name: str, kwargs: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Split kwargs into those this backbone accepts and those it does not.

    A sweep runs one config across several architectures, so a config that sets
    ``state_dim`` for an SSM will also reach the ConvLSTM, which has no such
    argument. Those must be dropped, or no cross-architecture sweep is possible.

    But dropping *everything* unrecognised would silently swallow typos, and a
    misspelled ``hidden_dim`` that trains a default-width model is exactly the
    kind of error that wastes a sweep. So an argument is only droppable if some
    *other* registered backbone accepts it -- that is what marks it as
    architecture-specific rather than wrong.

    Args:
        name: Registry key of the backbone being built.
        kwargs: Requested arguments.

    Returns:
        ``(accepted, droppable)``.

    Raises:
        ValueError: If an argument is accepted by no registered backbone.
    """
    accepted_here = _accepted_arguments(name)
    accepted_anywhere: set[str] = set()
    for other in TEMPORAL_BACKBONES:
        accepted_anywhere |= _accepted_arguments(other)

    accepted = {key: value for key, value in kwargs.items() if key in accepted_here}
    unknown = set(kwargs) - accepted_here

    typos = unknown - accepted_anywhere
    if typos:
        raise ValueError(
            f"Unknown backbone argument(s) {sorted(typos)} for {name!r}. "
            f"No registered backbone accepts them, so this is a typo rather than "
            f"an architecture-specific option. {name!r} accepts: "
            f"{', '.join(sorted(accepted_here - {'latent_dim'}))}."
        )
    return accepted, unknown


def build_backbone(name: str, latent_dim: int, **kwargs: Any) -> Any:
    """Construct a temporal backbone by registry name.

    Arguments that belong to a different architecture are dropped, so one sweep
    config can run across all four backbones. Genuine typos still raise -- see
    :func:`_partition_kwargs`.

    Args:
        name: Registry key, e.g. ``"convlstm"``, ``"s4d"``.
        latent_dim: Latent channel count; must match the encoder and decoder.
        **kwargs: Backbone-specific arguments.

    Returns:
        The constructed backbone.

    Raises:
        UnknownComponentError: If ``name`` is not registered.
        ValueError: If an argument is accepted by no registered backbone.
    """
    accepted, dropped = _partition_kwargs(name, kwargs)
    if dropped:
        logger.info(
            "%r does not take %s; dropping (they belong to another architecture)",
            name,
            ", ".join(sorted(dropped)),
        )
    return TEMPORAL_BACKBONES.build(name, latent_dim=latent_dim, **accepted)


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
        skip_connections=cfg.skip_connections,
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
        # Wired from the already-built encoder rather than recomputed here, so
        # the two can never disagree about channel counts even if a future
        # encoder implementation computes skip_channels differently.
        skip_channels=encoder.skip_channels if cfg.skip_connections else None,
    )
    backbone_kwargs = dict(cfg.backbone.kwargs)
    if cfg.backbone.size is not None:
        width = resolve_hidden_dim(
            cfg.backbone.name, cfg.backbone.size, skip_connections=cfg.skip_connections
        )
        # `hidden_dim: null` means "let the tier decide". Every model config
        # ships a concrete default width, and without a way to stand that
        # default down, `size` could never take effect from a composed config --
        # the tier would be computed, logged, and then silently discarded, which
        # is exactly how the matched-budget comparison came to be run at
        # mismatched budgets.
        explicit = backbone_kwargs.get("hidden_dim")
        if explicit is not None:
            logger.warning(
                "backbone.size=%r resolves to hidden_dim=%d for %r, but "
                "kwargs.hidden_dim=%s is set and wins. The size tier is NOT in "
                "effect; set kwargs.hidden_dim=null to use it.",
                cfg.backbone.size,
                width,
                cfg.backbone.name,
                explicit,
            )
        else:
            backbone_kwargs["hidden_dim"] = width
            logger.info(
                "size tier %r -> hidden_dim=%d for %r",
                cfg.backbone.size,
                width,
                cfg.backbone.name,
            )
    elif backbone_kwargs.get("hidden_dim") is None:
        raise ValueError(
            f"Backbone {cfg.backbone.name!r} has neither backbone.size nor "
            "backbone.kwargs.hidden_dim set, so its width is undefined. Set one."
        )

    backbone = build_backbone(
        cfg.backbone.name,
        latent_dim=cfg.latent_dim,
        **backbone_kwargs,
    )

    model = Forecaster(encoder=encoder, backbone=backbone, decoder=decoder, horizon=horizon)
    breakdown = model.parameter_breakdown()
    logger.info(
        "built %r forecaster [%s]: %s parameters (%.1f%% in the backbone), skip_connections=%s",
        cfg.backbone.name,
        cfg.architecture_version,
        f"{breakdown.total:,}",
        100 * breakdown.backbone_fraction,
        cfg.skip_connections,
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
