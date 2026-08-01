"""Model components: encoders, temporal backbones, decoders and losses.

The architecture is fixed::

    image encoder  ->  temporal backbone  ->  decoder  ->  forecast

**Only the temporal backbone varies across experiments.** That control is what
makes the efficiency comparison meaningful, and the interfaces in
:mod:`tinyearth.models.base` are what enforce it.

Four backbones are available: the ``convlstm`` and ``transformer`` baselines,
and the ``s4d`` and ``mamba`` state space models the research question is about.
All four can be built at any calibrated size tier -- tiny ~2M, small ~5M,
base ~10M, large ~20M -- so the scaling study compares them at matched budgets.

Example:
    >>> from tinyearth.models import TEMPORAL_BACKBONES, available_sizes
    >>> sorted(TEMPORAL_BACKBONES.keys())
    ['convlstm', 'mamba', 's4d', 'transformer']
    >>> available_sizes()
    ('tiny', 'small', 'base', 'large')
"""

from __future__ import annotations

from tinyearth.models.base import Decoder, Encoder, TemporalBackbone
from tinyearth.models.decoders import DECODERS, CNNDecoder
from tinyearth.models.encoders import ENCODERS, CNNEncoder
from tinyearth.models.factory import build_backbone, build_forecaster, build_loss
from tinyearth.models.forecaster import Forecaster, ParameterBreakdown, count_parameters
from tinyearth.models.losses import (
    LOSSES,
    CharbonnierLoss,
    CompositeLoss,
    ForecastLoss,
    GDLLoss,
    L1Loss,
    L2Loss,
)
from tinyearth.models.sizes import (
    SIZE_TIERS,
    SIZE_TIERS_SKIP,
    TARGET_PARAMETERS,
    available_sizes,
    resolve_hidden_dim,
)
from tinyearth.models.temporal import (
    TEMPORAL_BACKBONES,
    ConvLSTMBackbone,
    MambaBackbone,
    S4DBackbone,
    TemporalTransformerBackbone,
)

__all__ = [
    "DECODERS",
    "ENCODERS",
    "LOSSES",
    "SIZE_TIERS",
    "SIZE_TIERS_SKIP",
    "TARGET_PARAMETERS",
    "TEMPORAL_BACKBONES",
    "CNNDecoder",
    "CNNEncoder",
    "CharbonnierLoss",
    "CompositeLoss",
    "ConvLSTMBackbone",
    "Decoder",
    "Encoder",
    "ForecastLoss",
    "Forecaster",
    "GDLLoss",
    "L1Loss",
    "L2Loss",
    "MambaBackbone",
    "ParameterBreakdown",
    "S4DBackbone",
    "TemporalBackbone",
    "TemporalTransformerBackbone",
    "available_sizes",
    "build_backbone",
    "build_forecaster",
    "build_loss",
    "count_parameters",
    "resolve_hidden_dim",
]
