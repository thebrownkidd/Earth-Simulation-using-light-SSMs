"""Image encoders mapping a frame to a latent grid.

Held fixed across temporal-backbone experiments, and deliberately free of
temporal parameters -- every bit of sequence-modelling capacity belongs to the
backbone under study.

A frozen foundation encoder is a later comparison point; the registry is what
makes swapping one in a config change.
"""

from __future__ import annotations

from tinyearth.models.encoders.cnn import ENCODERS, CNNEncoder

__all__ = ["ENCODERS", "CNNEncoder"]
