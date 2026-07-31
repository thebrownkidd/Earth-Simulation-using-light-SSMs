"""Optimiser and learning-rate schedule construction.

Two decisions worth stating.

**Weight decay is not applied to norm and bias parameters.** Decaying a
normalisation scale pulls it toward zero, which fights the layer's purpose; the
effect is small but systematic, and it would differ between backbones because
they contain different numbers of norm layers. That would leak into the
comparison.

**Warmup is applied per step, not per epoch.** With short epochs -- and the
smoke configs here are very short -- an epoch-granular warmup either does
nothing or consumes the whole run. Per-step warmup behaves the same regardless
of epoch length.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tinyearth.config.schema import OptimizerConfig, SchedulerConfig
from tinyearth.utils.logging import get_logger

__all__ = ["build_optimizer", "build_scheduler", "parameter_groups"]

logger = get_logger(__name__)

_NO_DECAY_MODULES = (nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split parameters into decayed and non-decayed groups.

    Biases and normalisation parameters are excluded from weight decay.

    Args:
        model: Model whose parameters to group.
        weight_decay: Decay applied to the eligible group.

    Returns:
        Two parameter groups suitable for a :class:`~torch.optim.Optimizer`.
    """
    no_decay: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _NO_DECAY_MODULES):
            for param_name, _ in module.named_parameters(recurse=False):
                no_decay.add(f"{module_name}.{param_name}" if module_name else param_name)

    decayed: list[nn.Parameter] = []
    excluded: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in no_decay or name.endswith(".bias") or param.ndim <= 1:
            excluded.append(param)
        else:
            decayed.append(param)

    logger.debug(
        "parameter groups: %d decayed, %d excluded from weight decay",
        len(decayed),
        len(excluded),
    )
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": excluded, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    """Construct the optimiser.

    Args:
        model: Model to optimise.
        cfg: Optimiser configuration.

    Returns:
        The constructed optimiser.

    Raises:
        ValueError: If ``cfg.name`` is unknown.
    """
    groups = parameter_groups(model, cfg.weight_decay)
    name = cfg.name.strip().lower()

    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas))  # type: ignore[arg-type]
    if name == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, betas=tuple(cfg.betas))  # type: ignore[arg-type]
    if name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)

    raise ValueError(f"Unknown optimizer {cfg.name!r}. Expected 'adamw', 'adam' or 'sgd'.")


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> LRScheduler | None:
    """Construct the learning-rate schedule.

    Args:
        optimizer: Optimiser to schedule.
        cfg: Scheduler configuration.
        steps_per_epoch: Optimiser steps in one epoch, used to convert the
            warmup and cosine lengths to steps.
        epochs: Total epochs.

    Returns:
        The scheduler, stepped **per optimiser step**, or ``None`` for
        ``name: none`` and for ``plateau`` (which the trainer steps per epoch
        with a validation metric instead).

    Raises:
        ValueError: If ``cfg.name`` is unknown.
    """
    name = cfg.name.strip().lower()
    if name == "none":
        return None
    if name == "plateau":
        # Driven by a validation metric, so the trainer owns its stepping.
        return None

    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = max(steps_per_epoch * cfg.warmup_epochs, 0)

    if name == "cosine":
        base_lrs: list[float] = [float(group["lr"]) for group in optimizer.param_groups]

        def cosine(step: int) -> float:
            """Return the LR multiplier at ``step``."""
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Interpolate between the base LR and min_lr rather than scaling to
            # zero, so min_lr is honoured exactly.
            floor = cfg.min_lr / max(base_lrs[0], 1e-12)
            return floor + (1.0 - floor) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine)

    if name == "step":
        step_every = max(steps_per_epoch * cfg.step_size, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_every, gamma=cfg.gamma)

    raise ValueError(
        f"Unknown scheduler {cfg.name!r}. Expected 'cosine', 'step', 'plateau' or 'none'."
    )


def current_lr(optimizer: Optimizer) -> float:
    """Return the learning rate of the first parameter group.

    Args:
        optimizer: Optimiser to inspect.

    Returns:
        The current learning rate.
    """
    groups: Iterable[dict[str, Any]] = optimizer.param_groups
    for group in groups:
        return float(group["lr"])
    return 0.0
