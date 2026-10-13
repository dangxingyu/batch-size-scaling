"""Band channel engines with their own clocks, and channel-clock weight decay."""

import time

import torch

from bsz_lm.branchout.optimization import MatrixCore, apply_matrix_delta
from bsz_lm.branchout.spectral import TwoSpaces


class RankBands:
    """Contiguous ranges inside an accepted Ritz basis, plus its complement."""

    def __init__(self, basis, widths):
        self.full = TwoSpaces(basis)
        if not widths or any(not isinstance(w, int) or w <= 0 for w in widths) or sum(widths) != basis.shape[1]:
            raise ValueError("positive band widths must partition the retained basis")
        self.bases = {}
        start = 0
        for index, width in enumerate(widths):
            self.bases[f"rank_band_{index}"] = self.full.basis[:, start:start + width]
            start += width

    def project(self, value, band):
        if band == "remainder":
            return self.full.project(value, band)
        basis = self.bases[band]
        return basis @ (basis.T @ value.float())

    def leakage(self, value, band):
        return float((value - self.project(value, band)).square().sum()
                     / value.square().sum().clamp_min(1e-30))


class BandAccumulator:
    """Independent inherited states and clocks; no implicit channel averaging."""

    def __init__(self, parent, spaces, periods, *, project_input, project_output, lr_power,
                 reference_period=1, lr_scale=1.0):
        if not float(lr_scale) > 0:
            raise ValueError("lr_scale must be positive")
        # Extra multiplier on every channel's effective learning rate (LR scans at fixed k).
        self.lr_scale = float(lr_scale)
        if set(periods) != {*spaces.bases, "remainder"}:
            raise ValueError("one clock is required for each spectral band")
        if any(not isinstance(k, int) or k < 1 for k in periods.values()):
            raise ValueError("band periods must be positive integers")
        self.spaces, self.periods = spaces, periods
        self.project_input, self.project_output = project_input, project_output
        self.lr_power, self.base_steps = lr_power, 0
        self.reference_period = reference_period
        self.cores, self.accumulators, self.lr_accumulator = {}, {}, {}
        for band in periods:
            core = MatrixCore(parent.layout, parent.config, parent.m.device)
            core.load_state_dict(parent.state_dict())
            if project_input:
                core.m.copy_(spaces.project(core.m, band))
            self.cores[band] = core
            self.accumulators[band] = torch.zeros_like(parent.m)
            self.lr_accumulator[band] = 0.0

    @torch.no_grad()
    def advance(self, gradient, learning_rate):
        self.base_steps += 1
        delta = torch.zeros_like(gradient)
        metrics = {}
        for band, period in self.periods.items():
            self.lr_accumulator[band] += learning_rate
            self.accumulators[band].add_(self.spaces.project(gradient, band) if self.project_input else gradient)
            if self.base_steps % period:
                continue
            average = self.accumulators[band] / period
            raw = self.cores[band].direction(average)
            projected = self.spaces.project(raw, band)
            direction = projected if self.project_output else raw
            effective_lr = (self.lr_accumulator[band] / period * (period / self.reference_period)**self.lr_power
                            * self.lr_scale)
            delta.add_(direction, alpha=-effective_lr)
            metrics[band] = {
                "period": period, "effective_lr": effective_lr,
                "gradient_norm": float(average.norm()), "momentum_norm": float(self.cores[band].m.norm()),
                "raw_update_norm": float(raw.norm()), "applied_update_norm": float(direction.norm()),
                "raw_leakage": self.spaces.leakage(raw, band),
                "applied_leakage": self.spaces.leakage(direction, band),
                "momentum_leakage": self.spaces.leakage(self.cores[band].m, band),
                "optimizer_updates": self.cores[band].count,
            }
            self.accumulators[band].zero_()
            self.lr_accumulator[band] = 0.0
        return delta, metrics

    def assert_flushed(self):
        if any(self.base_steps % period for period in self.periods.values()):
            raise ValueError("unconsumed band accumulation")


@torch.no_grad()
def channel_decay(weights, split, metrics, weight_decay, wd_power, reference_period=1):
    """Apply each component's decay only when that component updates.

The input is the physical parameter BEFORE either channel update. Thus channel
ordering does not affect this additive decay. wd_power=1 scales lambda by the
channel period; wd_power=0 keeps lambda fixed. Auxiliary parameters are excluded.
"""
    decay = torch.zeros_like(weights)
    for band, record in metrics.items():
        period = split.periods[band]
        ratio = period / reference_period
        coefficient = record["effective_lr"] * weight_decay * ratio**wd_power
        if not 0 <= coefficient < 1:
            raise ValueError("invalid per-channel decay factor")
        decay.add_(split.spaces.project(weights, band), alpha=-coefficient)
        record["effective_weight_decay"] = weight_decay * ratio**wd_power
        record["decay_coefficient"] = coefficient
    return decay


def advance_channel_decay(exp, split, wd_power):
    """Same gradients/moments/auxiliary updates; explicit different WD clock."""
    if split is None:
        raise ValueError("channel decay requires split optimizer state")
    torch.cuda.synchronize()
    start = time.perf_counter()
    lr, wd = exp.schedule()
    gradient, training_loss = exp.gradients()
    torch.cuda.synchronize()
    after_gradient = time.perf_counter()
    delta, metrics = split.advance(gradient, lr)
    weights = exp.layout.pack(exp.parameters)
    delta.add_(channel_decay(weights, split, metrics, wd, wd_power))
    apply_matrix_delta(exp.parameters, exp.layout, delta, learning_rate=lr, weight_decay=0.0)
    exp.auxiliary.step()
    exp.model.zero_grad(set_to_none=True)
    exp.step += 1
    torch.cuda.synchronize()
    end = time.perf_counter()
    return {
        "step": exp.step, "tokens": exp.step * exp.cfg.base_batch_tokens,
        "training_loss": training_loss, "matrix_lr": lr, "matrix_wd": wd,
        "gradient_seconds": after_gradient - start, "optimizer_seconds": end - after_gradient,
        "step_seconds": end - start, "max_memory_bytes": torch.cuda.max_memory_allocated(),
        "bands": metrics, "decay_clock": "channel_update", "wd_power": wd_power,
    }
