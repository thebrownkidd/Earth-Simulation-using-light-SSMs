"""Temporal backbones -- the only component that varies across experiments.

Every backbone implements :class:`~tinyearth.models.base.TemporalBackbone`,
mapping ``[B, T, D, h, w]`` to ``[B, K, D, h, w]``, and registers itself in
:data:`TEMPORAL_BACKBONES` so configs select it by name.

Phase 3 provides two baselines whose purpose is correctness and a reference
point, not benchmark numbers:

``convlstm``
    Shi et al. (2015). Strictly sequential in time; the cost profile a state
    space model is expected to improve on.
``transformer``
    Attention over the temporal axis only, non-autoregressive.

``s4d``
    Diagonal state space model (Gu et al., 2022). Parallel in time via an FFT
    convolution. **The backbone this project exists to study.**
``mamba``
    Selective state space model (Gu and Dao, 2023). Input-dependent dynamics, at
    the cost of a sequential scan.
"""

from __future__ import annotations

from tinyearth.models.temporal.convlstm import (
    TEMPORAL_BACKBONES,
    ConvLSTMBackbone,
    ConvLSTMCell,
)
from tinyearth.models.temporal.selective_ssm import (
    MambaBackbone,
    SelectiveSSMBlock,
    selective_scan,
)
from tinyearth.models.temporal.ssm import S4DBackbone, S4DBlock, S4DKernel
from tinyearth.models.temporal.transformer import (
    SinusoidalPositionalEncoding,
    TemporalTransformerBackbone,
)

__all__ = [
    "TEMPORAL_BACKBONES",
    "ConvLSTMBackbone",
    "ConvLSTMCell",
    "MambaBackbone",
    "S4DBackbone",
    "S4DBlock",
    "S4DKernel",
    "SelectiveSSMBlock",
    "SinusoidalPositionalEncoding",
    "TemporalTransformerBackbone",
    "selective_scan",
]
