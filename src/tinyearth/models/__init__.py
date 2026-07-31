"""Model components: encoders, temporal backbones, decoders and losses.

The architecture is fixed as::

    image encoder -> temporal backbone -> decoder -> forecast

Only the temporal backbone varies across experiments, which is what makes
the efficiency comparison a controlled one. Phase 3 adds the ConvLSTM and
temporal-transformer baselines; Phase 4 adds the State Space Model
backbones and the ``TemporalBackbone`` interface they share.
"""

from __future__ import annotations

__all__: list[str] = []
