"""Decoders mapping a latent grid back to pixel space.

Held fixed across temporal-backbone experiments, and free of temporal
parameters.
"""

from __future__ import annotations

from tinyearth.models.decoders.cnn import DECODERS, CNNDecoder

__all__ = ["DECODERS", "CNNDecoder"]
