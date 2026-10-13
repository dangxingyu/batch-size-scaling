"""Two additive optimizer channels with independent input/output projection flags.

With output projection disabled the channels can overlap. In particular, at
k=1 off/off is TWICE the vanilla raw update, not a vanilla control. There is
no hidden averaging/rescaling: every factorial cell has its own k=1 control.
Weight decay is applied ONCE to the summed physical parameter per base step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from bsz_lm.branchout.spectral import Layout, TwoSpaces
from bsz_lm.optimizers import zeropower_via_newton_schulz


@dataclass(frozen=True)
class CoreConfig:
    algorithm: str
    momentum: float = 0.95

    def __post_init__(self):
        if self.algorithm != "muon":
            raise ValueError("branch-out supports Muon only")
        if not 0 <= self.momentum < 1:
            raise ValueError("invalid momentum coefficient")


class MatrixCore:
    """Track-3 matrix map with replicated moments and owner-sharded Muon NS."""

    def __init__(self, layout: Layout, config: CoreConfig, device: torch.device):
        self.layout, self.config = layout, config
        self.m = torch.zeros(layout.dimension, device=device)
        self.count = 0

    def state_dict(self) -> dict:
        return {"m": self.m.clone(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        if state["m"].shape != self.m.shape:
            raise ValueError("optimizer state/layout mismatch")
        self.m.copy_(state["m"])
        self.count = int(state["count"])

    @torch.no_grad()
    def direction(self, gradient: Tensor) -> Tensor:
        if gradient.shape != self.m.shape or not torch.isfinite(gradient).all():
            raise FloatingPointError("invalid optimizer gradient")
        self.count += 1
        cfg = self.config
        self.m.lerp_(gradient, 1 - cfg.momentum)
        source = gradient.lerp(self.m, cfg.momentum)
        direction = torch.zeros_like(source)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        pieces = self.layout.unpack(direction)
        for index, (name, value) in enumerate(self.layout.unpack(source).items()):
            if index % world == rank:
                # Muon's sqrt(max(1, m/n)) shape scale, applied in BF16 before conversion to FP32.
                pieces[name].copy_(zeropower_via_newton_schulz(value)
                                   * math.sqrt(max(1.0, value.shape[0] / value.shape[1])))
        if world > 1:
            dist.all_reduce(direction, op=dist.ReduceOp.SUM)
        return direction


class SplitAccumulator:
    def __init__(
        self, parent: MatrixCore, spaces: TwoSpaces, *,
        project_input: bool, project_output: bool, remainder_period: int,
        lr_power: float = 1.0,
    ):
        if remainder_period < 1 or int(remainder_period) != remainder_period:
            raise ValueError("accumulation period must be a positive integer")
        if not math.isfinite(lr_power):
            raise ValueError("invalid LR exponent")
        self.spaces = spaces
        self.project_input, self.project_output = project_input, project_output
        self.periods = {"top": 1, "remainder": remainder_period}
        self.lr_power = lr_power
        self.cores, self.accumulators = {}, {}
        for band in self.periods:
            core = MatrixCore(parent.layout, parent.config, parent.m.device)
            core.load_state_dict(parent.state_dict())
            if project_input:
                core.m.copy_(spaces.project(core.m, band))
            self.cores[band] = core
            self.accumulators[band] = torch.zeros_like(parent.m)
        self.base_steps = 0
        self.lr_accumulator = 0.0

    @torch.no_grad()
    def advance(self, gradient: Tensor, learning_rate: float) -> tuple[Tensor, dict]:
        self.base_steps += 1
        self.lr_accumulator += learning_rate
        delta = torch.zeros_like(gradient)
        metrics = {}
        for band, period in self.periods.items():
            incoming = self.spaces.project(gradient, band) if self.project_input else gradient
            self.accumulators[band].add_(incoming)
            if self.base_steps % period:
                continue
            averaged = self.accumulators[band] / period
            raw = self.cores[band].direction(averaged)
            projected = self.spaces.project(raw, band)
            direction = projected if self.project_output else raw
            # Sum token-clock LRs for p=1; use their mean * k^p otherwise.
            mean_lr = learning_rate if band == "top" else self.lr_accumulator / period
            effective_lr = mean_lr * period**self.lr_power
            delta.add_(direction, alpha=-effective_lr)
            metrics[band] = {
                "period": period,
                "effective_lr": effective_lr,
                "gradient_norm": float(averaged.norm()),
                "momentum_norm": float(self.cores[band].m.norm()),
                "raw_update_norm": float(raw.norm()),
                "applied_update_norm": float(direction.norm()),
                "raw_leakage": float((raw - projected).square().sum()
                                     / raw.square().sum().clamp_min(1e-30)),
                "applied_leakage": self.spaces.leakage(direction, band),
                "momentum_leakage": self.spaces.leakage(self.cores[band].m, band),
                "optimizer_updates": self.cores[band].count,
            }
            self.accumulators[band].zero_()
        if self.base_steps % self.periods["remainder"] == 0:
            self.lr_accumulator = 0.0
        return delta, metrics

    def assert_flushed(self) -> None:
        if self.base_steps % self.periods["remainder"]:
            raise ValueError("branch ends with unconsumed accumulation; horizon must divide period")


@torch.no_grad()
def apply_matrix_delta(parameters: dict[str, Tensor], layout: Layout,
                       delta: Tensor, *, learning_rate: float, weight_decay: float) -> None:
    if not 0 <= learning_rate * weight_decay < 1:
        raise ValueError("invalid token-clock decay factor")
    if not torch.isfinite(delta).all():
        raise FloatingPointError("nonfinite parameter update")
    for name, value in layout.unpack(delta).items():
        parameters[name].mul_(1 - learning_rate * weight_decay)
        parameters[name].add_(value.to(parameters[name].dtype))

