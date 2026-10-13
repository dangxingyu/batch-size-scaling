"""Directional collapse driver: anchors with fixed-weight block statistics and
token-matched branches under one declared transport.

Transport for the channel whose batch is k times the base batch: learning rate
is the mean scheduled LR over the accumulation window times k**lr_power
(default 0.5), per-update moment retentions are unchanged, matrix weight decay
stays on the physical base-step clock (per-token decay preserved), and in
global mode the auxiliary AdamW decay coefficient is multiplied by
k**aux_wd_power (default 0.5) so its per-token decay is also preserved.

Intervention modes:
  split  : top-16 channel every base step, complement channel every k base
           steps, both projections ON. k=1 is the split control.
  global : the unsplit optimizer at global batch k*B0; matrix and auxiliary
           parameters both accumulate k base gradients. k=1 reproduces the
           vanilla continuation bitwise.
  top    : top-16 channel every k base steps, complement every base step.
  matrix-out : both channels at period k with unprojected inputs, i.e. the
           hidden matrices at k*B0 while auxiliary parameters keep the base
           clock; attributes the global branch's penalty.
  split-out / top-out : as split / top, but channel inputs (gradients and
           inherited first moments) are NOT projected; only the returned
           directions are. At k=1 the two channels carry identical state and
           P Orth(m) + R Orth(m) = Orth(m), so the control is the vanilla map.
  random-complement-out : as complement-out with the anchor's fixed random basis in the
           place of the Ritz basis: random-r directions keep the base clock and their
           complement is slowed (the control for "the top-r must be the kept directions").
  hold   : one vanilla optimizer and clock; the held subspace (remainder, top or
           all) uses the mean of the previous complete k-step window of projected
           gradients, refreshed every k steps. Equal steps, no transport.
  extra  : one vanilla optimizer and clock; the held subspace averages the base
           batch with k-1 extra never-consumed batches each step. Equal steps,
           no transport, k times the gradient compute.

Every base step logs projections of the raw minibatch gradient, the relevant
momenta and the applied matrix update onto the 16 retained Hessian directions
and 16 fixed random complement directions.
"""

import argparse
import copy
import dataclasses
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from directional_extension_math import BandAccumulator, RankBands, channel_decay
from directional_noise_profile import chi_buckets, chi_values, resolve_columns
from directional_wide_basis import (
    ShardedBasis,
    WideRankBands,
    basis_drift,
    basis_overlap,
    map_and_orthonormalize,
    random_orthonormal_basis,
    refresh_tracked_basis,
    subspace_refresh,
    wide_lanczos,
)
from muon_preconditioner import MatrixPolarPreconditioner
from torch.nn.attention import SDPBackend, sdpa_kernel

from bsz_lm.branchout.config import load_config
from bsz_lm.branchout.optimization import SplitAccumulator
from bsz_lm.branchout.runner import (
    Experiment,
    barrier,
    curve_event,
    event,
    json_write,
    prepare_output,
    restore_rng,
    rng_state,
    run_anchor,
)
from bsz_lm.branchout.spectral import TwoSpaces, top_ritz
from bsz_lm.canonical import digest_file, digest_value

DEFAULT_BLOCKS = 64
COMPLEMENT_DIRECTIONS = 16
DRIVER_SHA256 = digest_file(Path(__file__))


def shared_digest(exp, path, expected=None):
    values = [digest_file(path) if exp.rank == 0 else None]
    if exp.world > 1:
        dist.broadcast_object_list(values, src=0)
    if expected is not None and values[0] != expected:
        raise ValueError(f"input digest changed: {path}")
    return values[0]


class GlobalAccumulator:
    """Unsplit optimizer at global batch k*B0 with the declared transport."""

    def __init__(self, exp, period, lr_power, aux_wd_power, aux_lr_power=None, lr_scale=1.0,
                 aux_lr_scale=1.0, aux_beta2=None):
        if int(period) != period or period < 1:
            raise ValueError("period must be a positive integer")
        if not float(lr_scale) > 0 or not float(aux_lr_scale) > 0:
            raise ValueError("lr_scale and aux_lr_scale must be positive")
        # Auxiliary diagnostics: an extra multiplier on the auxiliary AdamW learning rate
        # (on top of k**aux_lr_power) and an optional second-moment retention override,
        # applied to every auxiliary group from the first branch step (state inherited).
        self.aux_lr_scale = float(aux_lr_scale)
        self.aux_beta2 = None if aux_beta2 is None else float(aux_beta2)
        if self.aux_beta2 is not None:
            if not 0.0 < self.aux_beta2 < 1.0:
                raise ValueError("aux_beta2 must lie in (0, 1)")
            for group in exp.auxiliary.param_groups:
                group["betas"] = (group["betas"][0], self.aux_beta2)
        self.exp, self.period = exp, int(period)
        self.lr_power, self.aux_wd_power = float(lr_power), float(aux_wd_power)
        # Extra multiplier on the hidden-matrix learning rate only (k=1 LR controls);
        # weight decay and the auxiliary optimizer are untouched.
        self.lr_scale = float(lr_scale)
        # Auxiliary (embedding/head/norm) AdamW LR exponent; defaults to the matrix exponent.
        self.aux_lr_power = float(lr_power if aux_lr_power is None else aux_lr_power)
        self.accumulator = torch.zeros_like(exp.core.m)
        self.aux_grads = [[torch.zeros_like(p) for p in g["params"]]
                          for g in exp.auxiliary.param_groups]
        self.aux_lr = [0.0] * len(exp.auxiliary.param_groups)
        self.lr_accumulator, self.base_steps, self.updates, self.due = 0.0, 0, 0, False

    @torch.no_grad()
    def matrix_update(self, gradient, learning_rate):
        self.base_steps += 1
        self.lr_accumulator += learning_rate
        self.accumulator.add_(gradient)
        for index, (buffers, group) in enumerate(zip(self.aux_grads, self.exp.auxiliary.param_groups)):
            self.aux_lr[index] += float(group["lr"])
            for buffer, parameter in zip(buffers, group["params"]):
                if parameter.grad is None:
                    raise RuntimeError("missing auxiliary gradient")
                buffer.add_(parameter.grad)
        self.due = self.base_steps % self.period == 0
        if not self.due:
            return torch.zeros_like(gradient), {}
        averaged = self.accumulator / self.period
        raw = self.exp.core.direction(averaged)
        mean_lr = self.lr_accumulator / self.period
        effective = mean_lr * self.period ** self.lr_power * self.lr_scale
        self.accumulator.zero_()
        self.lr_accumulator = 0.0
        self.updates += 1
        metrics = {"global": {
            "period": self.period, "effective_lr": effective,
            "gradient_norm": float(averaged.norm()), "momentum_norm": float(self.exp.core.m.norm()),
            "raw_update_norm": float(raw.norm()), "applied_update_norm": float(raw.norm()),
            "optimizer_updates": self.exp.core.count,
        }}
        return raw * (-effective), metrics

    @torch.no_grad()
    def auxiliary_update(self):
        if not self.due:
            return
        auxiliary = self.exp.auxiliary
        for index, (buffers, group) in enumerate(zip(self.aux_grads, auxiliary.param_groups)):
            mean_lr = self.aux_lr[index] / self.period
            group["lr"] = mean_lr * self.period ** self.aux_lr_power * self.aux_lr_scale
            group["weight_decay"] = group["initial_weight_decay"] * self.period ** self.aux_wd_power
            for buffer, parameter in zip(buffers, group["params"]):
                parameter.grad.copy_(buffer / self.period)
                buffer.zero_()
            self.aux_lr[index] = 0.0
        auxiliary.step()

    def assert_flushed(self):
        if self.base_steps % self.period:
            raise ValueError("global accumulation not flushed; horizon must divide period")


class AuxiliaryClock:
    """Auxiliary AdamW groups (embeddings, head, scalars) on the k-clock with the matrix
    transport, for the channel modes: gradients averaged over `period` base steps, one
    AdamW step per period with LR x k**aux_lr_power (x aux_lr_scale), weight decay
    x k**aux_wd_power, retention fixed per update. Period 1 is the base recipe bitwise.
    Same arithmetic as GlobalAccumulator's auxiliary path."""

    def __init__(self, exp, period, aux_lr_power=0.5, aux_wd_power=0.5, aux_lr_scale=1.0, aux_beta2=None):
        if int(period) != period or period < 1:
            raise ValueError("period must be a positive integer")
        if not float(aux_lr_scale) > 0:
            raise ValueError("aux_lr_scale must be positive")
        self.exp, self.period = exp, int(period)
        self.aux_lr_power, self.aux_wd_power = float(aux_lr_power), float(aux_wd_power)
        self.aux_lr_scale = float(aux_lr_scale)
        self.aux_beta2 = None if aux_beta2 is None else float(aux_beta2)
        if self.aux_beta2 is not None:
            if not 0.0 < self.aux_beta2 < 1.0:
                raise ValueError("aux_beta2 must lie in (0, 1)")
            for group in exp.auxiliary.param_groups:
                group["betas"] = (group["betas"][0], self.aux_beta2)
        self.aux_grads = [[torch.zeros_like(p) for p in g["params"]] for g in exp.auxiliary.param_groups]
        self.aux_lr = [0.0] * len(exp.auxiliary.param_groups)
        self.base_steps, self.updates = 0, 0

    @torch.no_grad()
    def base_step(self):
        """Called once per base step after the gradients: accumulate; step when due."""
        self.base_steps += 1
        auxiliary = self.exp.auxiliary
        for index, (buffers, group) in enumerate(zip(self.aux_grads, auxiliary.param_groups)):
            self.aux_lr[index] += float(group["lr"])
            for buffer, parameter in zip(buffers, group["params"]):
                if parameter.grad is None:
                    raise RuntimeError("missing auxiliary gradient")
                buffer.add_(parameter.grad)
        if self.base_steps % self.period:
            return False
        for index, (buffers, group) in enumerate(zip(self.aux_grads, auxiliary.param_groups)):
            mean_lr = self.aux_lr[index] / self.period
            group["lr"] = mean_lr * self.period ** self.aux_lr_power * self.aux_lr_scale
            group["weight_decay"] = group["initial_weight_decay"] * self.period ** self.aux_wd_power
            for buffer, parameter in zip(buffers, group["params"]):
                parameter.grad.copy_(buffer / self.period)
                buffer.zero_()
            self.aux_lr[index] = 0.0
        auxiliary.step()
        self.updates += 1
        return True

    def assert_flushed(self):
        if self.base_steps % self.period:
            raise ValueError("auxiliary accumulation not flushed; horizon must divide period")


class HeldEstimator:
    """Directional gradient estimator with one vanilla optimizer and one clock.

    The held subspace ("remainder", "top" or "all") uses the mean of the last
    complete k-step window of projected gradients, refreshed every k steps; the
    other subspace uses the fresh base-batch gradient. Learning rate, moments and
    update count are untouched. The first window (no complete history yet) uses
    the fresh gradient everywhere; k=1 is the vanilla estimator bitwise.
    """

    def __init__(self, exp, spaces, period, held):
        if int(period) != period or period < 1:
            raise ValueError("period must be a positive integer")
        if held not in {"remainder", "top", "all"}:
            raise ValueError("held must be remainder, top or all")
        self.exp, self.spaces, self.period, self.held = exp, spaces, int(period), held
        self.accumulator = torch.zeros_like(exp.core.m)
        self.average = None
        self.base_steps = 0

    def _held_part(self, gradient):
        return gradient if self.held == "all" else self.spaces.project(gradient, self.held)

    @torch.no_grad()
    def estimate(self, gradient):
        self.base_steps += 1
        held_fresh = self._held_part(gradient)
        self.accumulator.add_(held_fresh)
        if self.period == 1:
            estimate, stale = gradient, False
        elif self.average is None:
            estimate, stale = gradient, False
        else:
            estimate, stale = gradient - held_fresh + self.average, True
        if self.base_steps % self.period == 0:
            self.average = self.accumulator / self.period
            self.accumulator.zero_()
        return estimate, {"estimator": "hold", "held": self.held, "period": self.period,
                          "stale_window_used": stale,
                          "held_fresh_norm": float(held_fresh.norm()),
                          "held_average_norm": float(self.average.norm()) if self.average is not None else None}

    def assert_flushed(self):
        if self.base_steps % self.period:
            raise ValueError("held-window accumulation not flushed")


class ExtraDataEstimator:
    """Directional estimator whose held subspace averages k batches per step.

    The k-1 extra batches come from training positions never consumed by the
    13,000-step schedule (indices from `extra_base_step` onward), so no branch
    or parent data is reused. The fresh subspace uses the base batch only.
    """

    def __init__(self, exp, spaces, period, held, extra_base_step):
        if int(period) != period or period < 1:
            raise ValueError("period must be a positive integer")
        if held not in {"remainder", "top", "all"}:
            raise ValueError("held must be remainder, top or all")
        self.exp, self.spaces, self.period, self.held = exp, spaces, int(period), held
        self.extra_base_step, self.extra_used = int(extra_base_step), 0
        self.extra_sum = torch.zeros_like(exp.core.m)
        self.base_steps = 0

    def before_gradients(self):
        """Compute the extra gradients first so the auxiliary .grad ends on the base batch."""
        self.extra_sum.zero_()
        original = self.exp.step
        try:
            for _ in range(self.period - 1):
                self.exp.step = self.extra_base_step + self.extra_used
                self.extra_used += 1
                gradient, _ = self.exp.gradients()
                self.extra_sum.add_(gradient)
        finally:
            self.exp.step = original
            self.exp.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def estimate(self, gradient):
        self.base_steps += 1
        if self.period == 1:
            return gradient, {"estimator": "extra", "held": self.held, "period": 1}
        averaged = (gradient + self.extra_sum) / self.period
        if self.held == "all":
            estimate = averaged
        else:
            estimate = gradient - self.spaces.project(gradient, self.held) + self.spaces.project(averaged, self.held)
        return estimate, {"estimator": "extra", "held": self.held, "period": self.period,
                          "extra_batches_used": self.extra_used,
                          "held_averaged_norm": float(self.spaces.project(averaged, self.held).norm())
                          if self.held != "all" else float(averaged.norm())}

    def assert_flushed(self):
        return None


def block_statistics(exp, basis, complement, count, ritz_values):
    """Fixed-weight statistics over `count` unconsumed 128K blocks after the anchor.

    Gradients are the training-sum gradients (131,072-token sums) in the joint
    hidden-matrix coordinates, so ratios are meaningful and absolute scales are
    in Track-3 training-sum units. The Hessian block (step-1) is excluded by construction.
    """
    before = rng_state()
    original_step = exp.step
    dimension = exp.layout.dimension
    total = torch.zeros(dimension, device=exp.device)
    squares = 0.0
    top_rows, complement_rows = [], []
    try:
        for block in range(count):
            exp.step = original_step + block
            gradient, _ = exp.gradients()
            total += gradient
            squares += float(gradient.dot(gradient))
            top_rows.append(basis.T @ gradient)
            complement_rows.append(complement.T @ gradient)
    finally:
        exp.step = original_step
        exp.model.zero_grad(set_to_none=True)
        restore_rng(before)
    mean = total / count
    top = torch.stack(top_rows)
    rest = torch.stack(complement_rows)
    mean_norm_sq = float(mean.dot(mean))
    total_variance = (squares - count * mean_norm_sq) / (count - 1)
    signal = mean_norm_sq - total_variance / count
    top_mean = top.mean(0)
    top_mean_sq = float(top_mean.dot(top_mean))
    top_variance = float((top - top_mean).square().sum()) / (count - 1)
    top_signal = top_mean_sq - top_variance / count
    rest_mean_sq = mean_norm_sq - top_mean_sq
    rest_variance = total_variance - top_variance
    rest_signal = rest_mean_sq - rest_variance / count

    def per_direction(rows, values):
        means = rows.mean(0)
        variances = rows.var(0, unbiased=True)
        out = []
        for i in range(rows.shape[1]):
            m, v = float(means[i]), float(variances[i])
            sig = m * m - v / count
            out.append({"mean": m, "variance": v, "signal": sig,
                        "snr": (m * m / v) if v > 0 else None,
                        "b_simple_blocks": (v / sig) if sig > 0 else None,
                        "ritz_value": values[i] if values is not None else None,
                        "variance_over_ritz_sq": (v / values[i] ** 2) if values is not None and values[i] else None})
        return out

    return {
        "blocks": count, "block_tokens": exp.cfg.base_batch_tokens,
        "first_block_base_step": original_step, "gradient_units": "131072-token training sums",
        "full": {"mean_norm_sq": mean_norm_sq, "variance": total_variance, "signal": signal,
                 "b_simple_blocks": (total_variance / signal) if signal > 0 else None},
        "top": {"mean_norm_sq": top_mean_sq, "variance": top_variance, "signal": top_signal,
                "b_simple_blocks": (top_variance / top_signal) if top_signal > 0 else None},
        "remainder": {"mean_norm_sq": rest_mean_sq, "variance": rest_variance, "signal": rest_signal,
                      "b_simple_blocks": (rest_variance / rest_signal) if rest_signal > 0 else None},
        "top_directions": per_direction(top, ritz_values),
        "complement_directions": per_direction(rest, None),
    }


def run_anchor_stage(cfg, task, root):
    checkpoint = Path(task["checkpoint"])
    basis_path = run_anchor(cfg, task["algorithm"], checkpoint, root / "anchor")
    gc.collect()
    torch.cuda.empty_cache()
    barrier()
    exp = Experiment(cfg, task["algorithm"])
    exp.restore(checkpoint)
    payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    basis = payload["basis"].to(exp.device)
    metadata = payload["metadata"]
    del payload
    generator = torch.Generator(device=exp.device).manual_seed(cfg.seed * 1000003 + exp.step + 4242)
    complement = torch.randn(exp.layout.dimension, COMPLEMENT_DIRECTIONS,
                             device=exp.device, generator=generator)
    for _ in range(2):
        complement -= basis @ (basis.T @ complement)
    complement = torch.linalg.qr(complement).Q.contiguous()
    if exp.world > 1:
        dist.broadcast(complement, src=0)
    statistics = block_statistics(exp, basis, complement, int(task.get("blocks", DEFAULT_BLOCKS)),
                                  metadata["eigenvalues"])
    basis_hash = shared_digest(exp, basis_path)
    record = {"schema": "directional_collapse_anchor_v1", "status": "complete",
              "algorithm": task["algorithm"], "parent_step": exp.step,
              "checkpoint": str(checkpoint), "checkpoint_sha256": metadata["checkpoint_sha256"],
              "basis": str(basis_path), "basis_sha256": basis_hash,
              "ritz_values": metadata["eigenvalues"], "max_relative_residual": metadata["max_relative_residual"],
              "complement_directions": COMPLEMENT_DIRECTIONS,
              "complement_seed": cfg.seed * 1000003 + exp.step + 4242,
              "block_statistics": statistics, "source_id": exp.source,
              "config_id": digest_value(cfg.to_dict()), "driver_sha256": DRIVER_SHA256}
    if exp.rank == 0:
        torch.save({"directions": complement.cpu(), "metadata": {
            "basis_sha256": basis_hash, "checkpoint_sha256": metadata["checkpoint_sha256"],
            "parent_step": exp.step, "seed": record["complement_seed"]}},
            root / "complement_directions.pt")
        json_write(root / "result.json", record)
    barrier()
    return record


def make_hvp(exp, cfg, batch_step=None, tokens=None, micro=None):
    """Hessian-vector product of the mean token loss over the batch preceding exp.step (or
    the training batch of `batch_step`) on an fp32 copy of the current model (the same
    construction as the src anchor stage).  `tokens` overrides cfg.hessian_tokens, which
    sets how many tokens the curvature is estimated on.  Returns (hvp, release, state)
    where release() frees the copy and state["calls"] counts HVPs."""
    model = copy.deepcopy(exp.model).float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [dict(model.named_parameters())[name] for name in exp.layout.names]
    for parameter in selected:
        parameter.requires_grad_(True)
    hessian_tokens = int(cfg.hessian_tokens if tokens is None else tokens)
    inputs, targets = exp.data.batch(exp.step - 1 if batch_step is None else int(batch_step),
                                     tokens=hessian_tokens)
    micro = int(micro or cfg.microbatch_sequences)
    if micro < 1:
        raise ValueError("hvp micro-batch must be a positive number of sequences")
    state = {"model": model, "selected": selected, "inputs": inputs, "targets": targets, "calls": 0, "micro": micro}

    def hvp(vector):
        state["calls"] += 1
        vector_parts = tuple(exp.layout.unpack(vector).values())
        total = torch.zeros_like(vector)
        for start in range(0, len(state["inputs"]), micro):
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                loss = state["model"](state["inputs"][start:start + micro], state["targets"][start:start + micro]) / hessian_tokens
                first = torch.autograd.grad(loss, state["selected"], create_graph=True)
                dot = sum((g * v).sum() for g, v in zip(first, vector_parts, strict=True))
                second = torch.autograd.grad(dot, state["selected"], allow_unused=True)
                values = {name: (torch.zeros_like(p) if g is None else g)
                          for name, p, g in zip(exp.layout.names, state["selected"], second, strict=True)}
                total += exp.layout.pack(values)
                del first, second, dot, loss, values
        if exp.world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return total

    def release():
        for key in ("model", "selected", "inputs", "targets"):
            state[key] = None
        gc.collect()
        torch.cuda.empty_cache()

    return hvp, release, state


def gauss_newton_product(logits, parameters, vector_parts, token_count):
    """One Gauss-Newton product on a batch of softmax-cross-entropy logits.

    `logits` must carry a graph back to `parameters`.  Returns the tuple
    J^T H_z J v, one tensor per parameter, where J = d logits / d parameters and H_z is the
    per-token softmax Hessian diag(p) - p p^T divided by `token_count`.  H_z does not involve
    the labels, so neither does this product.  Three passes: J v by the double-backward
    trick, the closed-form H_z, then J^T of the result."""
    seed = torch.zeros_like(logits, requires_grad=True)
    transposed = torch.autograd.grad(logits, parameters, grad_outputs=seed, create_graph=True)
    directional = torch.autograd.grad(transposed, seed, grad_outputs=vector_parts,
                                      retain_graph=True)[0]
    with torch.no_grad():
        probability = torch.softmax(logits.detach(), dim=-1)
        weighted = probability * directional
        curved = (weighted - probability * weighted.sum(-1, keepdim=True)) / token_count
    return torch.autograd.grad(logits, parameters, grad_outputs=curved, allow_unused=True)


def make_ggn_hvp(exp, cfg, batch_step=None, tokens=None, micro=None):
    """Generalized Gauss-Newton product of the same mean token loss, in place of the exact
    Hessian.  Splitting the loss as L = l(z(w)) at the softcapped logits z, the exact Hessian
    is J^T H_z J + sum_k (dl/dz_k) grad^2 z_k; the GGN keeps only the first term, which is
    positive semidefinite because H_z is.  For softmax cross entropy H_z is block diagonal
    with blocks diag(p) - p p^T per token, so the GGN does not depend on the labels at all and
    equals the Fisher information of the model's own predictive distribution.

    Each product is three passes: J v by the double-backward trick, then the closed-form
    H_z applied to it, then J^T of that.  Returns (hvp, release, state) like `make_hvp`."""
    model = copy.deepcopy(exp.model).float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [dict(model.named_parameters())[name] for name in exp.layout.names]
    for parameter in selected:
        parameter.requires_grad_(True)
    hessian_tokens = int(cfg.hessian_tokens if tokens is None else tokens)
    inputs, _ = exp.data.batch(exp.step - 1 if batch_step is None else int(batch_step),
                               tokens=hessian_tokens)
    micro = int(micro or cfg.microbatch_sequences)
    if micro < 1:
        raise ValueError("hvp micro-batch must be a positive number of sequences")
    state = {"model": model, "selected": selected, "inputs": inputs, "calls": 0, "micro": micro}

    def hvp(vector):
        state["calls"] += 1
        vector_parts = tuple(exp.layout.unpack(vector).values())
        total = torch.zeros_like(vector)
        for start in range(0, len(state["inputs"]), micro):
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                logits = state["model"].logits(state["inputs"][start:start + micro])
                out = gauss_newton_product(logits, state["selected"], vector_parts, hessian_tokens)
                values = {name: (torch.zeros_like(p) if g is None else g)
                          for name, p, g in zip(exp.layout.names, state["selected"], out, strict=True)}
                total += exp.layout.pack(values)
                del logits, out, values
        if exp.world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return total

    def release():
        for key in ("model", "selected", "inputs"):
            state[key] = None
        gc.collect()
        torch.cuda.empty_cache()

    return hvp, release, state


def base_clock_core(engine, exp):
    """The optimizer core whose clock is the base clock: the period-1 channel of a branch.

    Its state is what the k=1 control's step is taken in, so it is the state to rebuild the
    preconditioner from during a branch.  The parent core `exp.core` is not stepped by the
    channel engine, so it holds the anchor state for the whole window, and a P^{1/2} built
    from it would stay frozen at the anchor."""
    periods, cores = getattr(engine, "periods", None), getattr(engine, "cores", None)
    if not periods or not cores:
        return exp.core, "parent"
    band = min(periods, key=periods.get)
    return cores[band], band


def make_preconditioner(exp, task, core=None):
    """P^{1/2} as a linear map, from task['preconditioner'].

    The linearized dynamics of a preconditioned step is d <- (I - lr P H) d, so the spectrum
    that governs it is that of P H, equivalently of the symmetric P^{1/2} H P^{1/2}.  Raw-H
    eigendirections are not the ones the optimizer struggles with.

      none          : identity, i.e. the raw curvature.
      muon-polar    : P = the Jacobian of Orth at the momentum, which is the Hessian of the
                      nuclear norm and so symmetric PSD; its exact square root comes from the
                      momentum's SVD (see scripts/muon_preconditioner.py).  Each matrix's P
                      carries Muon's sqrt(max(1, m/n)) shape scale.

    `core` is the optimizer core to read the state from; the parent core by default, the
    base-clock channel core during a tracked branch (see base_clock_core).

    Returns (apply, metadata); `apply` is None for the identity."""
    kind = str(task.get("preconditioner", "none"))
    if kind == "none":
        return None, {"preconditioner": "none"}
    core = exp.core if core is None else core
    if kind == "muon-polar":
        damping = float(task.get("preconditioner_damping", 1e-3))
        momenta, per_matrix_scale = exp.layout.unpack(core.m), {}
        for name, value in momenta.items():
            # Muon's sqrt(max(1, m/n)) shape scale is P's per-matrix weight
            per_matrix_scale[name] = math.sqrt(max(1.0, value.shape[0] / value.shape[1]))
        # one batched SVD per matrix shape instead of 72 sequential ones
        factors = MatrixPolarPreconditioner.batch(momenta, damping=damping, scales=per_matrix_scale,
                                                  method=str(task.get("polar_factorization", "svd")))
        spectra = [(float(f.s.min()), float(f.s.max())) for f in factors.values()]
        scales = list(per_matrix_scale.values())
        names = list(momenta)

        def apply_muon(vector):
            pieces = exp.layout.unpack(vector)
            return exp.layout.pack({name: factors[name].half(part) for name, part in pieces.items()})

        def apply_muon_block(matrix):
            """P^{1/2} on the columns of a [dimension, k] block: the per-matrix GEMMs run once
            on a [k, m, n] batch instead of k times, which is what makes re-mapping a wide basis
            cheap."""
            columns = [exp.layout.unpack(matrix[:, j]) for j in range(matrix.shape[1])]
            mapped = {name: factors[name].half_batch(torch.stack([c[name] for c in columns]))
                      for name in names}
            return torch.stack([exp.layout.pack({name: mapped[name][j] for name in names})
                                for j in range(matrix.shape[1])], dim=1)

        apply_muon.block = apply_muon_block

        metadata = {"preconditioner": kind, "damping": damping,
                    "factorization": str(task.get("polar_factorization", "svd")),
                    "matrices": len(factors),
                    "scale_min": min(scales), "scale_max": max(scales),
                    "singular_min": min(a for a, _ in spectra),
                    "singular_max": max(b for _, b in spectra)}
        return apply_muon, metadata
    raise ValueError(f"unknown preconditioner {kind!r}")


def curvature_operator(exp, cfg, task, batch_step=None, tokens=None):
    """The curvature operator a task asks for: exact Hessian or Gauss-Newton, optionally
    conjugated by the optimizer's preconditioner as P^{1/2} (.) P^{1/2}."""
    kind = str(task.get("hessian_operator", "exact"))
    # Systems knobs (task keys, default = the anchors' construction): `hvp_microbatch_sequences`
    # sets how many sequences each HVP pass takes at once (the config's 2 keeps a 124M model
    # latency-bound), `hvp_matmul_precision` the fp32 matmul mode for the product ("high" =
    # TF32 tensor cores, ~1e-3 relative; anchors keep "highest").  Both are recorded.
    micro = task.get("hvp_microbatch_sequences")
    if kind == "exact":
        base, release, state = make_hvp(exp, cfg, batch_step, tokens, micro=micro)
    elif kind == "ggn":
        base, release, state = make_ggn_hvp(exp, cfg, batch_step, tokens, micro=micro)
    else:
        raise ValueError(f"unknown hessian_operator {kind!r}")
    precision = task.get("hvp_matmul_precision")
    state["matmul_precision"] = str(precision) if precision else torch.get_float32_matmul_precision()
    if precision:
        if str(precision) not in {"highest", "high", "medium"}:
            raise ValueError("hvp_matmul_precision must be highest, high or medium")
        inner = base

        def base(vector):
            previous = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision(str(precision))
            try:
                return inner(vector)
            finally:
                torch.set_float32_matmul_precision(previous)
    half, metadata = make_preconditioner(exp, task)
    state["preconditioner"] = metadata
    if half is None:
        return base, release, state

    def conjugated(vector):
        return half(base(half(vector)))

    return conjugated, release, state


def run_anchor_wide_stage(cfg, task, root):
    """Wide Ritz basis for the rank sweep: sharded bf16 Lanczos with `iterations` steps
    keeping `top_rank` vectors, an independent HVP audit of each retained pair, a
    Hutchinson trace estimate, and an orthonormal random basis of `random_rank`
    columns as the control. Writes one shard file per rank plus result.json."""
    checkpoint = Path(task["checkpoint"])
    exp = Experiment(cfg, task["algorithm"])
    exp.restore(checkpoint)
    if exp.step < 1:
        raise ValueError("Hessian anchors require a preceding training batch")
    checkpoint_hash = shared_digest(exp, checkpoint, task.get("checkpoint_sha256"))
    top_rank, iterations = int(task["top_rank"]), int(task["iterations"])
    prepare_output(exp, root, "anchor_wide", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                   top_rank=top_rank, iterations=iterations)
    before_rng = rng_state()
    hvp, release_hvp, hvp_state = curvature_operator(exp, cfg, task, tokens=task.get("hessian_tokens"))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    generator = torch.Generator(device=exp.device).manual_seed(cfg.seed + exp.step + 913)
    probes = [torch.randn(exp.layout.dimension, device=exp.device, generator=generator) for _ in range(2)]
    probes = [v / v.norm() for v in probes]
    products = [hvp(v) for v in probes]
    symmetry = float((torch.dot(probes[0], products[1]) - torch.dot(probes[1], products[0])).abs()
                     / (products[0].norm() + products[1].norm()).clamp_min(1e-20))
    del probes, products
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[task.get("krylov_dtype", "bf16")]
    basis, report = wide_lanczos(hvp, dimension=exp.layout.dimension, top_rank=top_rank, iterations=iterations,
                                 device=exp.device, seed=cfg.seed + exp.step, krylov_dtype=dtype,
                                 audit_rank=int(task.get("audit_rank", top_rank)),
                                 trace_probes=int(task.get("trace_probes", 16)))
    calls = hvp_state["calls"]
    release_hvp()
    random_rank = int(task.get("random_rank", top_rank))
    random_seed = cfg.seed * 7919 + exp.step + 17
    random = random_orthonormal_basis(exp.layout.dimension, random_rank, exp.device, random_seed)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gate_rank = int(task.get("gate_rank", min(16, top_rank)))
    gate = max(report["relative_residuals"][:gate_rank])
    accepted = symmetry <= 1e-4 and gate <= cfg.ritz_relative_tolerance
    metadata = {"schema": "directional_cbs_anchor_wide_v1", "accepted": accepted, "checkpoint_sha256": checkpoint_hash,
                "parent_step": exp.step, "source_id": exp.source, "config_id": digest_value(cfg.to_dict()),
                "layout": exp.layout.metadata(), "world": exp.world, "top_rank": top_rank, "iterations": iterations,
                "krylov_dtype": task.get("krylov_dtype", "bf16"), "random_rank": random_rank, "random_seed": random_seed,
                "gate_rank": gate_rank, "gate_max_relative_residual": gate, "hvp_symmetry_error": symmetry}
    report.update(metadata)
    report.update({"hvp_calls": calls, "seconds": elapsed, "max_memory_bytes": torch.cuda.max_memory_allocated(),
                   "random_orthogonality_error": random.gram_error(),
                   "hessian_tokens": int(task.get("hessian_tokens") or cfg.hessian_tokens),
                   "hessian_batch_base_step": exp.step - 1, "driver_sha256": DRIVER_SHA256,
                   "hessian_precision": "FP32 mathematical-model proxy at saved weights",
                   "hessian_operator": str(task.get("hessian_operator", "exact")),
                   "preconditioner": hvp_state.get("preconditioner")})
    restore_rng(before_rng)
    if accepted:
        torch.save({"basis": basis.state(), "random": random.state(), "metadata": metadata},
                   root / f"basis-shard{exp.rank:02d}.pt")
    if exp.rank == 0:
        json_write(root / "result.json", report)
    barrier()
    if not accepted:
        raise RuntimeError(f"wide Hessian gate failed: symmetry={symmetry}, gate residual={gate}")
    return report


def run_noise_profile_stage(cfg, task, root):
    """Gradient-noise variance along every direction of a stored wide basis, at the anchor.

    For each of `profile_batches` consecutive base batches after the anchor (the stream a
    branch would consume), the mean-loss gradient of the hidden matrices is taken, pushed
    through P^{1/2} when the task names a preconditioner, and projected onto the stored basis
    U, which for a preconditioned anchor lives in preconditioned coordinates.  The variance
    over batches of u^T P^{1/2} g_B is the noise the base-batch optimizer sees along u;
    divided by lam_u^2 it is the chi of that direction, and the buckets of lowest and highest
    chi are what a branch can slow.  The random basis gives the same statistic for directions
    with no curvature ordering at all."""
    checkpoint = Path(task["checkpoint"])
    exp = Experiment(cfg, task["algorithm"])
    exp.restore(checkpoint)
    checkpoint_hash = shared_digest(exp, checkpoint, task.get("checkpoint_sha256"))
    path = Path(task["wide_basis"]) / f"basis-shard{exp.rank:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    for key, expected in {"checkpoint_sha256": checkpoint_hash, "parent_step": exp.step, "source_id": exp.source,
                          "layout": exp.layout.metadata(), "world": exp.world, "accepted": True}.items():
        if meta[key] != expected:
            raise ValueError(f"wide basis identity mismatch: {key}")
    basis = ShardedBasis.from_state(payload["basis"], exp.device)
    random = ShardedBasis.from_state(payload["random"], exp.device)
    del payload
    anchor_report = json.loads((Path(task["wide_basis"]) / "result.json").read_text())
    eigenvalues = [float(x) for x in anchor_report["eigenvalues"][:basis.width]]
    batches = int(task.get("profile_batches", 128))
    tokens = int(task.get("profile_tokens", cfg.base_batch_tokens))
    prepare_output(exp, root, "noise_profile", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                   batches=batches, tokens=tokens, width=basis.width)
    half, pre_meta = make_preconditioner(exp, task)
    model = copy.deepcopy(exp.model).float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [dict(model.named_parameters())[name] for name in exp.layout.names]
    for parameter in selected:
        parameter.requires_grad_(True)
    micro = cfg.microbatch_sequences
    torch.cuda.synchronize()
    start = time.perf_counter()
    coefficients, random_coefficients, norms = [], [], []
    for offset in range(batches):
        inputs, targets = exp.data.batch(exp.step + offset, tokens=tokens)
        gradient = torch.zeros(exp.layout.dimension, device=exp.device, dtype=torch.float32)
        for first in range(0, len(inputs), micro):
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                loss = model(inputs[first:first + micro], targets[first:first + micro]) / tokens
                parts = torch.autograd.grad(loss, selected, allow_unused=True)
                values = {name: (torch.zeros_like(p) if g is None else g)
                          for name, p, g in zip(exp.layout.names, selected, parts, strict=True)}
                gradient += exp.layout.pack(values)
                del loss, parts, values
        if exp.world > 1:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        vector = half(gradient) if half is not None else gradient
        norms.append(float(vector.norm()))
        coefficients.append(basis.coefficients(vector).cpu())
        random_coefficients.append(random.coefficients(vector).cpu())
        del gradient, vector
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stacked = torch.stack(coefficients).double()
    random_stacked = torch.stack(random_coefficients).double()
    variance = stacked.var(dim=0, unbiased=True)
    chi = chi_values(eigenvalues, variance.tolist())
    size = int(task.get("bucket_size", min(256, basis.width // 2)))
    buckets = chi_buckets(chi, size)
    buckets["top_lambda"] = list(range(size))
    buckets["bottom_lambda"] = list(range(basis.width - size, basis.width))
    report = {"schema": "directional_cbs_noise_profile_v1", "checkpoint_sha256": checkpoint_hash,
              "parent_step": exp.step, "source_id": exp.source, "layout": exp.layout.metadata(),
              "wide_basis": str(task["wide_basis"]), "batches": batches, "tokens": tokens, "width": basis.width,
              "first_batch_step": exp.step, "eigenvalues": eigenvalues,
              "mean": stacked.mean(dim=0).tolist(), "variance": variance.tolist(), "chi": chi,
              "buckets": buckets, "bucket_size": size,
              "random_variance": random_stacked.var(dim=0, unbiased=True).tolist(),
              "gradient_norm_mean": float(sum(norms) / len(norms)),
              "preconditioner": pre_meta, "hessian_operator": anchor_report.get("hessian_operator"),
              "seconds": elapsed, "max_memory_bytes": torch.cuda.max_memory_allocated(),
              "driver_sha256": DRIVER_SHA256}
    if exp.rank == 0:
        json_write(root / "result.json", report)
    barrier()
    return report


# Modes whose retained band is the anchor's random basis rather than the Ritz basis.
RANDOM_MODES = {"random-out", "random-complement-out"}
# Channel clocks of the wide-basis modes: the bands that accumulate k base gradients per update.
WIDE_MODE_SLOWED = {"top-out": ("rank_band_0",), "random-out": ("rank_band_0",),
                    "complement-out": ("remainder",), "random-complement-out": ("remainder",),
                    "matrix-out": ("rank_band_0", "remainder")}


def wide_channel_periods(mode, period):
    """Base-step period of each band of a wide-basis channel mode (1 = the base clock)."""
    slowed = WIDE_MODE_SLOWED[mode]
    return {band: (int(period) if band in slowed else 1) for band in ("rank_band_0", "remainder")}


def load_wide_basis(exp, task):
    """This rank's shard of the wide Ritz (or random) basis, narrowed to task['top_rank']."""
    path = Path(task["wide_basis"]) / f"basis-shard{exp.rank:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    for key, expected in {"checkpoint_sha256": shared_digest(exp, Path(task["checkpoint"]), task.get("checkpoint_sha256")),
                          "parent_step": exp.step, "source_id": exp.source, "layout": exp.layout.metadata(),
                          "world": exp.world, "accepted": True}.items():
        if meta[key] != expected:
            raise ValueError(f"wide basis identity mismatch: {key}")
    kind = "random" if task["mode"] in RANDOM_MODES else "basis"
    columns = resolve_columns(task) if kind == "basis" else None
    # Memory at rank 768: the stored shard is 1,536 bf16 columns (32 GB per rank), the narrowed
    # basis 16 GB and the fp32 image of the mapping 33 GB, so nothing may be duplicated.  The
    # 16 logging columns come from the stored Ritz basis before it is dropped.
    if kind == "basis":
        stored = ShardedBasis.from_state(payload["basis"], exp.device)
        logging_basis = stored.gather_columns(min(16, stored.width))
    else:
        ritz = ShardedBasis.from_state(payload["basis"], exp.device)
        logging_basis = ritz.gather_columns(min(16, ritz.width))
        del ritz
        stored = ShardedBasis.from_state(payload["random"], exp.device)
    del payload
    if columns is not None:
        # A subspace chosen by a measurement (e.g. the lowest-chi directions) rather than the
        # top-r prefix; the rank must say how many columns were chosen so labels stay honest.
        if len(columns) != int(task["top_rank"]):
            raise ValueError("top_rank must equal the number of selected basis columns")
        sharded = stored.select(columns)
        meta = dict(meta, basis_columns=list(columns))
    else:
        sharded = stored.narrow(int(task["top_rank"]))
    del stored
    gc.collect()
    torch.cuda.empty_cache()
    preconditioned = None
    if task.get("preconditioner", "none") != "none" and kind == "basis":
        # The anchor solved P^{1/2} H P^{1/2} u = lam u, so the parameter-space subspace is
        # spanned by P^{1/2} u.  Map and re-orthonormalize before it becomes a projector.  Only
        # a tracked branch needs u itself afterwards (it refreshes u and re-maps); a frozen
        # branch drops it, which at rank 768 is 16 GB per rank.
        half, _ = make_preconditioner(exp, task)
        mapped, mapping = map_and_orthonormalize(sharded, half, matmul_precision=task.get("basis_matmul_precision"))
        preconditioned = sharded if task.get("refresh_interval") else None
        sharded = mapped
        del mapped
        gc.collect()
        torch.cuda.empty_cache()
        meta = dict(meta, parameter_space_mapping=mapping)
    return sharded, logging_basis, meta, preconditioned


def measure_validation(exp, root, tokens):
    value = exp.evaluate(split="val", tokens=tokens)
    if exp.rank == 0:
        with (root / "validation_curve.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "step": exp.step, "tokens": exp.step * exp.cfg.base_batch_tokens,
                "validation_loss": value, "validation_tokens": tokens,
                "validation_id": digest_value(exp.data.metadata.get("val", exp.data.metadata)),
                "recorded_at": time.time()}, sort_keys=True, allow_nan=False) + "\n")
    barrier()


def projections(gradient, delta, momentum_top, momentum_rest, basis, complement):
    parts = [basis.T @ gradient, complement.T @ gradient,
             basis.T @ delta, complement.T @ delta,
             basis.T @ momentum_top, complement.T @ momentum_rest]
    pieces = torch.cat(parts).cpu().split([part.numel() for part in parts])
    names = ("top_gradient", "complement_gradient", "top_delta", "complement_delta",
             "top_momentum", "complement_momentum")
    return {name: piece.tolist() for name, piece in zip(names, pieces)}


def advance(exp, engine, mode, basis, complement, decay=("physical", 0.0)):
    """One base step. decay=("physical", _) applies (1 - lr*wd) to the whole parameter
    every base step; decay=("channel", q) instead subtracts lr_eff * wd * k**q * P_j W on
    each channel's own update (q=0: coefficient fixed per update; q=0.5: per-token decay
    preserved, equivalent to the physical clock up to second order); in global mode the
    whole matrix block decays on the k*B0 update only (global_channel_decay)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    lr, wd = exp.schedule()
    if hasattr(engine, "before_gradients"):
        engine.before_gradients()
    gradient, training_loss = exp.gradients()
    torch.cuda.synchronize()
    after_gradient = time.perf_counter()
    if mode == "global":
        delta, metrics = engine.matrix_update(gradient, lr)
        momentum_top = momentum_rest = exp.core.m
    elif mode in {"hold", "extra"}:
        estimate, info = engine.estimate(gradient)
        delta = exp.core.direction(estimate) * (-lr)
        metrics = {"estimator": dict(info, effective_lr=lr, optimizer_updates=exp.core.count,
                                     raw_update_norm=float(delta.norm() / lr) if lr else None)}
        momentum_top = momentum_rest = exp.core.m
    else:
        delta, metrics = engine.advance(gradient, lr)
        top_key = "top" if mode in {"split", "split-out"} else "rank_band_0"
        momentum_top, momentum_rest = engine.cores[top_key].m, engine.cores["remainder"].m
    logged = projections(gradient, delta, momentum_top, momentum_rest, basis, complement)
    logged.update({"gradient_norm": float(gradient.norm()), "delta_norm": float(delta.norm())})
    applied_wd = wd
    if decay[0] == "channel":
        if mode in {"hold", "extra"}:
            raise ValueError("channel decay requires a channel engine or global mode")
        weights = exp.layout.pack(exp.parameters)
        if mode == "global":
            delta = delta + global_channel_decay(weights, engine, metrics, wd, float(decay[1]))
        else:
            delta = delta + channel_decay(weights, engine, metrics, wd, float(decay[1]))
        applied_wd = 0.0
    exp.apply_delta(delta, lr, applied_wd)
    if mode == "global":
        engine.auxiliary_update()
    elif getattr(engine, "aux_clock", None) is not None:
        engine.aux_clock.base_step()
    else:
        exp.auxiliary.step()
    exp.model.zero_grad(set_to_none=True)
    exp.step += 1
    torch.cuda.synchronize()
    end = time.perf_counter()
    record = {"step": exp.step, "tokens": exp.step * exp.cfg.base_batch_tokens,
              "training_loss": training_loss, "matrix_lr": lr, "matrix_wd": wd,
              "decay_clock": decay[0], "wd_power": float(decay[1]),
              "gradient_seconds": after_gradient - start,
              "optimizer_seconds": end - after_gradient, "step_seconds": end - start,
              "max_memory_bytes": torch.cuda.max_memory_allocated(), "bands": metrics}
    logged["step"] = exp.step
    return record, logged


MATRIX_HPARAM_KEYS = {"matrix_mu": "momentum"}


def apply_matrix_hparam_overrides(cores, task):
    """Per-branch override of the Muon momentum, applied to every channel core after the
    parent state is inherited. Used for the EMA-transport arms beta -> 1-k(1-beta)."""
    for key, field in MATRIX_HPARAM_KEYS.items():
        if task.get(key) is None:
            continue
        value = float(task[key])
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{key} must lie in [0, 1)")
        for core in cores:
            core.config = dataclasses.replace(core.config, **{field: value})


SCALED_HPARAM_KEYS = {key + "_scaled": key for key in MATRIX_HPARAM_KEYS}


def apply_scaled_channel_overrides(engine, exp, task):
    """Retention override for the SCALED channels only: `matrix_mu_scaled` is applied to every
    channel core whose period is the task period, i.e. the slowed band(s) of a channel engine or the single core of global
    mode, so a rule such as mu -> mu^k transports the momentum of the k*B0 channel while a
    period-1 channel keeps the base recipe.  `matrix_mu` (no suffix) still hits every channel."""
    given = {base: task[key] for key, base in SCALED_HPARAM_KEYS.items() if task.get(key) is not None}
    if not given:
        return
    period = int(task["period"])
    if period == 1:
        raise ValueError("scaled-channel overrides need a period above 1")
    if hasattr(engine, "cores") and hasattr(engine, "periods"):
        cores = [engine.cores[band] for band, p in engine.periods.items() if int(p) == period]
    elif task["mode"] == "global":
        cores = [exp.core]
    else:
        raise ValueError("scaled-channel overrides are defined for channel engines and global mode")
    if not cores:
        raise ValueError("no channel runs at the task period")
    apply_matrix_hparam_overrides(cores, given)


def global_channel_decay(weights, engine, metrics, weight_decay, wd_power):
    """Global mode on the channel clock: the hidden matrices decay only on the base steps where
    the k*B0 update fires, by lr_eff * wd * k**wd_power (wd_power=0 keeps the coefficient
    lambda fixed per update, the paper's convention W <- (1 - eta lambda) W - eta U at the
    transported eta; wd_power=1 scales lambda by k).  Off-cycle steps decay nothing."""
    if not metrics:
        return torch.zeros_like(weights)
    record = metrics["global"]
    ratio = float(engine.period)
    coefficient = record["effective_lr"] * weight_decay * ratio ** wd_power
    if not 0 <= coefficient < 1:
        raise ValueError("invalid global decay factor")
    record["effective_weight_decay"] = weight_decay * ratio ** wd_power
    record["decay_coefficient"] = coefficient
    return weights * (-coefficient)


def build_engine(exp, mode, basis, task, spaces=None):
    unused = [key for key in ("matrix_beta1", "matrix_beta2", "matrix_beta1_scaled",
                              "matrix_beta2_scaled") if task.get(key) is not None]
    if unused:
        raise ValueError(f"{unused} are not hyperparameters of Muon")
    engine = _build_engine(exp, mode, basis, task, spaces)
    cores = list(engine.cores.values()) if hasattr(engine, "cores") else [exp.core]
    apply_matrix_hparam_overrides(cores, task)
    apply_scaled_channel_overrides(engine, exp, task)
    if task.get("aux_scaled"):
        # Auxiliary parameters on the channel period with the matrix transport ("all
        # parameters scaled"); the global mode already does this internally.
        if mode in {"global", "hold", "extra"}:
            raise ValueError("aux_scaled is defined for the channel modes only")
        engine.aux_clock = AuxiliaryClock(exp, int(task["period"]), task.get("aux_lr_power", task.get("lr_power", 0.5)),
                                          task.get("aux_wd_power", 0.5), task.get("aux_lr_scale", 1.0),
                                          task.get("aux_beta2"))
    return engine


def _build_engine(exp, mode, basis, task, spaces=None):
    period, power = int(task["period"]), float(task.get("lr_power", 0.5))
    if mode in WIDE_MODE_SLOWED and spaces is not None:
        # Wide (sharded) basis: one band of task["top_rank"] columns plus its complement.
        periods = wide_channel_periods(mode, period)
        return BandAccumulator(exp.core, spaces, periods, project_input=False, project_output=True,
                               lr_power=power, lr_scale=task.get("lr_scale", 1.0))
    if mode in RANDOM_MODES:
        raise ValueError(f"{mode} requires a wide basis (task['wide_basis'])")
    if mode == "complement-out":
        spaces = RankBands(basis, [basis.shape[1]])
        return BandAccumulator(exp.core, spaces, {"rank_band_0": 1, "remainder": period},
                               project_input=False, project_output=True, lr_power=power,
                               lr_scale=task.get("lr_scale", 1.0))
    if mode in {"split", "split-out"}:
        return SplitAccumulator(exp.core, TwoSpaces(basis), project_input=(mode == "split"),
                                project_output=True, remainder_period=period, lr_power=power)
    if mode in {"top", "top-out"}:
        spaces = RankBands(basis, [basis.shape[1]])
        return BandAccumulator(exp.core, spaces, {"rank_band_0": period, "remainder": 1},
                               project_input=(mode == "top"), project_output=True, lr_power=power,
                               lr_scale=task.get("lr_scale", 1.0))
    if mode == "matrix-out":
        # Both channels slowed together: hidden matrices at k*B0 with the same rule,
        # auxiliary parameters still every base step. Isolates the auxiliary part of
        # the global branch's penalty.
        spaces = RankBands(basis, [basis.shape[1]])
        return BandAccumulator(exp.core, spaces, {"rank_band_0": period, "remainder": period},
                               project_input=False, project_output=True, lr_power=power,
                               lr_scale=task.get("lr_scale", 1.0))
    if mode == "global":
        return GlobalAccumulator(exp, period, power, task.get("aux_wd_power", 0.5),
                                 task.get("aux_lr_power"), lr_scale=task.get("lr_scale", 1.0),
                                 aux_lr_scale=task.get("aux_lr_scale", 1.0),
                                 aux_beta2=task.get("aux_beta2"))
    if mode == "hold":
        return HeldEstimator(exp, TwoSpaces(basis), period, task.get("held", "remainder"))
    if mode == "extra":
        return ExtraDataEstimator(exp, TwoSpaces(basis), period, task.get("held", "remainder"),
                                  task.get("extra_base_step", 13000))
    raise ValueError(f"unknown mode {mode}")


def recompute_basis(exp, cfg, previous_basis):
    """Top-16 Ritz basis of the joint hidden-matrix Hessian at the current weights,
    on the preceding training batch, with the anchor-stage solver and gates. Returns
    (basis or None if the gate failed, report)."""
    before = rng_state()
    model = copy.deepcopy(exp.model).float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [dict(model.named_parameters())[name] for name in exp.layout.names]
    for parameter in selected:
        parameter.requires_grad_(True)
    inputs, targets = exp.data.batch(exp.step - 1, tokens=cfg.hessian_tokens)
    micro = cfg.microbatch_sequences

    def hvp(vector):
        parts = tuple(exp.layout.unpack(vector).values())
        total = torch.zeros_like(vector)
        for start in range(0, len(inputs), micro):
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                loss = model(inputs[start:start + micro], targets[start:start + micro]) / cfg.hessian_tokens
                first = torch.autograd.grad(loss, selected, create_graph=True)
                dot = sum((g * v).sum() for g, v in zip(first, parts, strict=True))
                second = torch.autograd.grad(dot, selected, allow_unused=True)
                total += exp.layout.pack({name: (torch.zeros_like(p) if g is None else g)
                                          for name, p, g in zip(exp.layout.names, selected, second, strict=True)})
                del first, second, dot, loss
        if exp.world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return total

    torch.cuda.synchronize()
    start = time.perf_counter()
    basis, report = top_ritz(hvp, dimension=exp.layout.dimension, top_rank=cfg.top_rank,
                             iterations=cfg.lanczos_iterations, device=exp.device,
                             seed=cfg.seed + exp.step)
    torch.cuda.synchronize()
    restore_rng(before)
    accepted = report["max_relative_residual"] <= cfg.ritz_relative_tolerance
    overlap = (previous_basis.T @ basis.to(previous_basis.device)).float()
    report = {"step": exp.step, "accepted": accepted, "seconds": time.perf_counter() - start,
              "ritz_values": report["eigenvalues"], "max_relative_residual": report["max_relative_residual"],
              "overlap_with_previous": float((overlap * overlap).sum() / basis.shape[1]),
              "max_principal_angle_deg": float(torch.rad2deg(torch.arccos(
                  torch.linalg.svdvals(overlap).clamp(max=1.0))).max())}
    return (basis if accepted else None), report


def run_branch_stage(cfg, task, root):
    checkpoint = Path(task["checkpoint"])
    anchor_root = Path(task["anchor"])
    basis_path, complement_path = anchor_root / "anchor/basis.pt", anchor_root / "complement_directions.pt"
    mode, period, horizon = task["mode"], int(task["period"]), int(task["branch_steps"])
    exp = Experiment(cfg, task["algorithm"])
    exp.restore(checkpoint)
    checkpoint_hash = shared_digest(exp, checkpoint, task.get("checkpoint_sha256"))
    basis_hash = shared_digest(exp, basis_path, task.get("basis_sha256"))
    payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    for key, expected in {"checkpoint_sha256": checkpoint_hash, "parent_step": exp.step,
                          "source_id": exp.source, "config_id": digest_value(cfg.to_dict()),
                          "layout": exp.layout.metadata(), "accepted": True}.items():
        if payload["metadata"][key] != expected:
            raise ValueError(f"basis identity mismatch: {key}")
    basis = payload["basis"].to(exp.device)
    del payload
    wide_spaces, wide_meta, wide_preconditioned = None, None, None
    if int(task.get("remap_interval", 0)) and (not task.get("wide_basis") or task.get("preconditioner", "none") == "none"):
        raise ValueError("remap_interval needs a preconditioned wide basis (the map is P^{1/2})")
    if task.get("wide_basis"):
        sharded, basis, wide_meta, wide_preconditioned = load_wide_basis(exp, task)
        wide_spaces = WideRankBands(sharded, [sharded.width])
    saved = torch.load(complement_path, map_location="cpu", weights_only=False)
    if saved["metadata"]["basis_sha256"] != basis_hash or saved["metadata"]["parent_step"] != exp.step:
        raise ValueError("complement directions do not match this basis")
    complement = saved["directions"].to(exp.device)
    del saved
    if horizon <= 0 or horizon % period or exp.step + horizon > cfg.schedule_steps:
        raise ValueError("invalid or unflushed branch horizon")
    engine = build_engine(exp, mode, basis, task, wide_spaces)
    decay = (task.get("decay_rule", "physical"), float(task.get("wd_power", 0.0)))
    if decay[0] not in {"physical", "channel"}:
        raise ValueError(f"unknown decay rule {decay[0]}")
    curve_interval = int(task.get("curve_interval", 32))
    validation_tokens = int(task.get("validation_curve_tokens", cfg.curve_eval_tokens))
    prepare_output(exp, root, "collapse_branch", task=task, parent_step=exp.step, branch_steps=horizon,
                   checkpoint_sha256=checkpoint_hash, basis_sha256=basis_hash,
                   driver_sha256=DRIVER_SHA256,
                   transport=({"lr_power": 0.0, "moments": "unchanged", "matrix_wd_clock": "physical base step",
                               "note": "equal-step directional estimator; no batch transport",
                               "held": task.get("held", "remainder")} if mode in {"hold", "extra"} else
                              {"lr_power": float(task.get("lr_power", 0.5)),
                               "moments": "fixed per channel update",
                               "matrix_wd_clock": decay[0],
                               "matrix_wd_power": decay[1],
                               "aux_wd_power": float(task.get("aux_wd_power", 0.5)) if mode == "global" else None,
                               "aux_lr_power": float(task.get("aux_lr_power", task.get("lr_power", 0.5))) if mode == "global" else None}))
    curve_event(exp, root)
    measure_validation(exp, root, validation_tokens)
    origin = json.loads((root / "curve.jsonl").read_text().splitlines()[0])
    parent_rows = [json.loads(line) for line in Path(task["parent_curve"]).read_text().splitlines()]
    parent_origin = next(r for r in parent_rows if r["step"] == exp.step)
    origin_error = abs(origin["holdout_loss"] - parent_origin["holdout_loss"])
    if origin["holdout_id"] != parent_origin["holdout_id"] or origin_error > float(task.get("origin_atol", 1e-5)):
        raise AssertionError(f"parent origin mismatch: {origin_error}")
    torch.cuda.reset_peak_memory_stats()
    timings = []
    refresh_interval = int(task.get("refresh_interval", 0))
    if refresh_interval and mode not in {"split-out", "top-out", "matrix-out", "complement-out"}:
        raise ValueError("basis refresh is defined for the unprojected-input channel modes only")
    # The shift keeps a one-pass block iteration aimed at the algebraically largest
    # eigenvalues, so it has to live on the operator's own scale: 0.25 is a quarter of lam_1 for
    # the raw Hessian at anchor 1,000 (lam_1 ~ 1.1) but sixteen times lam_1 for P^{1/2}HP^{1/2}
    # (lam_1 ~ 0.015), where it would turn the iteration into (nearly) the identity and the
    # "tracked" basis would never move.  Default to a fraction of the anchor's own lam_1; an
    # explicit absolute `refresh_shift` still wins.
    if "refresh_shift" in task:
        refresh_shift = float(task["refresh_shift"])
    else:
        refresh_shift = float(task.get("refresh_shift_relative", 0.25))
        if refresh_interval and task.get("wide_basis"):
            report = json.loads((Path(task["wide_basis"]) / "result.json").read_text())
            refresh_shift *= float(report["eigenvalues"][0])
    refreshes = []
    # Re-map cadence: between Gauss-Newton refreshes the stored eigenvectors U are pushed through
    # the LIVE P^{1/2} every `remap_interval` base steps, so the parameter-space projector orth(P_t^{1/2} U) follows the
    # optimizer state at its own (momentum) time scale while the eigenproblem is solved only
    # every `refresh_interval` steps.  The preconditioner rebuild is a per-matrix SVD of the
    # momentum (about 3 s on the 124M-parameter model) plus the r-column mapping.
    remap_interval = int(task.get("remap_interval", 0))
    remaps = []
    drift_steps = sorted({int(d) for d in task.get("drift_steps", [])})
    # Batch control: at the anchor weights, the same measurement against the Hessian of a
    # different training batch (step - b); separates Hessian-batch noise from weight motion.
    drift_batch_offsets = sorted({int(b) for b in task.get("drift_batch_offsets", [])})
    if (drift_steps or drift_batch_offsets) and (wide_spaces is None or period != 1 or refresh_interval):
        raise ValueError("eigenspace drift is measured on a period-1 (vanilla) branch with a frozen wide basis")
    if any(b < 1 for b in drift_batch_offsets):
        raise ValueError("drift batch offsets count back from the current step (1 = the anchor's own batch)")
    drifts, drift_controls = [], []

    def measure_drift(offset, batch_offset=None):
        # Frozen anchor basis vs the Hessian of the batch just consumed (or of the control
        # batch), at the current weights.
        hvp, release_hvp, _ = curvature_operator(exp, cfg, task,
                                                 None if batch_offset is None else exp.step - batch_offset,
                                                 tokens=task.get("drift_hessian_tokens"))
        report = basis_drift(hvp, engine.spaces.full, ranks=task.get("drift_ranks", [16, 64, 128, 256]),
                             iterations=int(task.get("drift_iterations", 2 * engine.spaces.full.width)),
                             seed=cfg.seed * 1000003 + exp.step + 777 + 31 * (batch_offset or 0),
                             audit=int(task.get("drift_audit", 16)),
                             trace_probes=int(task.get("drift_trace_probes", 16)))
        release_hvp()
        gc.collect()
        torch.cuda.empty_cache()
        report.update({"step": exp.step, "offset": offset, "batch_offset": batch_offset if batch_offset else 1,
                       "control": batch_offset is not None,
                       "hessian_tokens": int(task.get("drift_hessian_tokens") or cfg.hessian_tokens)})
        (drift_controls if batch_offset is not None else drifts).append(report)
        if exp.rank == 0:
            with (root / "drift.jsonl").open("a") as handle:
                handle.write(json.dumps(report, allow_nan=False) + "\n")

    if 0 in drift_steps:
        measure_drift(0)
    for batch_offset in drift_batch_offsets:
        measure_drift(0, batch_offset)
    for offset in range(horizon):
        record, logged = advance(exp, engine, mode, basis, complement, decay)
        record["wall"] = time.time()
        timings.append(record["step_seconds"])
        if offset + 1 in drift_steps:
            measure_drift(offset + 1)
        if refresh_interval and wide_spaces is not None and (offset + 1) % refresh_interval == 0 and offset + 1 < horizon:
            # Wide basis: one warm-started block subspace-iteration pass on H + shift*I at the
            # current weights (width HVPs + a 16-column audit); channel states are untouched.
            phases, mark, last = {}, time.perf_counter(), [time.perf_counter()]

            def lap(name):
                now = time.perf_counter()
                phases[name] = now - last[0]
                last[0] = now
            # `refresh_hessian_tokens` (default cfg.hessian_tokens, the anchor's batch) trades
            # Gauss-Newton sampling noise for refresh time: the block pass costs width + audit
            # HVPs, each linear in the tokens (272 HVPs at 128K tokens are ~290 s at rank 256).
            refresh_tokens = int(task.get("refresh_hessian_tokens") or cfg.hessian_tokens)
            hvp, release_hvp, hvp_state = curvature_operator(exp, cfg, task, tokens=refresh_tokens)
            lap("curvature_setup")
            if wide_preconditioned is not None:
                # Track in the coordinates of the eigenproblem, then re-map, as at load time;
                # P^{1/2} is rebuilt at the current optimizer state, which is the geometry the
                # step is taking right now.
                live_core, source = base_clock_core(engine, exp)
                half, pre_meta = make_preconditioner(exp, task, core=live_core)
                lap("preconditioner")
                wide_preconditioned, new_basis, report = refresh_tracked_basis(
                    hvp, wide_preconditioned, half, shift=refresh_shift, audit=16,
                    map_precision=task.get("basis_matmul_precision"))
                report["preconditioner"] = dict(pre_meta, source=source)
            else:
                new_basis, report = subspace_refresh(hvp, engine.spaces.full, shift=refresh_shift, audit=16)
            lap("refresh_and_map")
            release_hvp()
            engine.spaces = WideRankBands(new_basis, [new_basis.width])
            wide_spaces = engine.spaces
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            lap("release")
            phases["total"] = time.perf_counter() - mark
            report["step"] = exp.step
            report["phase_seconds"] = phases
            report["hessian_tokens"] = refresh_tokens
            report["matmul_precision"] = hvp_state["matmul_precision"]
            report["hvp_microbatch_sequences"] = hvp_state["micro"]
            record["basis_refresh"] = report
            refreshes.append(report)
        elif refresh_interval and (offset + 1) % refresh_interval == 0 and offset + 1 < horizon:
            # Free the projector bases on the GPU (10.8 GB) before the 128-vector Lanczos
            # workspace (40.5 GB) is allocated; rebuild them afterwards.
            previous_cpu, complement_cpu = basis.cpu(), complement.cpu()
            engine.spaces = None
            del basis, complement
            gc.collect()
            torch.cuda.empty_cache()
            new_basis, report = recompute_basis(exp, cfg, previous_cpu)
            basis = (new_basis if new_basis is not None else previous_cpu.to(exp.device)).contiguous()
            complement = complement_cpu.to(exp.device)
            if new_basis is not None:
                complement -= basis @ (basis.T @ complement)
                complement -= basis @ (basis.T @ complement)
                complement = torch.linalg.qr(complement).Q.contiguous()
                if exp.world > 1:
                    dist.broadcast(complement, src=0)
            engine.spaces = (TwoSpaces(basis) if mode == "split-out"
                             else RankBands(basis, [basis.shape[1]]))
            del previous_cpu, complement_cpu
            record["basis_refresh"] = report
            refreshes.append(report)
            gc.collect()
            torch.cuda.empty_cache()
        if remap_interval and wide_preconditioned is not None and (offset + 1) % remap_interval == 0 \
                and offset + 1 < horizon and not (refresh_interval and (offset + 1) % refresh_interval == 0):
            started = time.perf_counter()
            previous = engine.spaces.full
            live_core, source = base_clock_core(engine, exp)
            half, pre_meta = make_preconditioner(exp, task, core=live_core)
            mapped, mapping = map_and_orthonormalize(wide_preconditioned, half,
                                                     matmul_precision=task.get("basis_matmul_precision"))
            report = {"step": exp.step, "preconditioner_source": source,
                      "map_phase_seconds": mapping.get("phase_seconds"),
                      "matmul_precision": mapping.get("matmul_precision"),
                      "gram_condition": mapping["gram_condition"],
                      "orthogonality_error": mapping["orthogonality_error"]}
            report["overlap_with_previous"] = basis_overlap(previous, mapped)
            engine.spaces = WideRankBands(mapped, [mapped.width])
            wide_spaces = engine.spaces
            del previous
            gc.collect()
            torch.cuda.empty_cache()
            report["seconds"] = time.perf_counter() - started
            record["basis_remap"] = report
            remaps.append(report)
        event(exp, root, record)
        if exp.rank == 0:
            with (root / "directions.jsonl").open("a") as handle:
                handle.write(json.dumps(logged, allow_nan=False) + "\n")
        if (offset + 1) % curve_interval == 0 or offset + 1 == horizon:
            curve_event(exp, root)
            measure_validation(exp, root, validation_tokens)
    engine.assert_flushed()
    if getattr(engine, "aux_clock", None) is not None:
        engine.aux_clock.assert_flushed()
    saved_checkpoint = None
    if task.get("save_checkpoint"):
        # A global-mode branch steps the parent core itself, so its endpoint checkpoint is an
        # exact continuation point (a channel engine keeps its own cores, which this would not
        # save).  Restores like a parent checkpoint under the same config.
        if mode != "global":
            raise ValueError("save_checkpoint is defined for global mode only")
        saved_checkpoint = root / "final.pt"
        exp.checkpoint(saved_checkpoint)
        barrier()
    endpoint = exp.evaluate()
    result = {"schema": "directional_collapse_branch_v1", "status": "complete",
              "checkpoint": str(saved_checkpoint) if saved_checkpoint else None,
              "algorithm": task["algorithm"], "mode": mode, "period": period,
              "held": task.get("held", "remainder") if mode in {"hold", "extra"} else None,
              "decay_rule": decay[0], "wd_power": decay[1],
              "aux_lr_power": float(task.get("aux_lr_power", task.get("lr_power", 0.5))) if mode == "global" or task.get("aux_scaled") else None,
              "aux_wd_power": float(task.get("aux_wd_power", 0.5)) if mode == "global" or task.get("aux_scaled") else None,
              "aux_scaled": bool(task.get("aux_scaled", False)),
              "lr_power": 0.0 if mode in {"hold", "extra"} else float(task.get("lr_power", 0.5)),
              "lr_scale": float(task.get("lr_scale", 1.0)),
              "top_rank": int(task["top_rank"]) if task.get("top_rank") else None,
              "wide_basis": task.get("wide_basis"),
              "wide_basis_iterations": wide_meta["iterations"] if wide_meta else None,
              "matrix_mu": task.get("matrix_mu"),
              "matrix_mu_scaled": task.get("matrix_mu_scaled"),
              "aux_lr_scale": float(task.get("aux_lr_scale", 1.0)) if mode == "global" or task.get("aux_scaled") else None,
              "aux_beta2": task.get("aux_beta2") if mode == "global" or task.get("aux_scaled") else None,
              "parent_step": task["step"],
              "branch_steps": horizon, "additional_tokens": horizon * cfg.base_batch_tokens,
              "final_data_next_base_step": exp.step, "validation_loss": endpoint,
              "validation_tokens": cfg.validation_tokens, "origin_absolute_error": origin_error,
              "parent_checkpoint_sha256": checkpoint_hash, "basis_sha256": basis_hash,
              "source_id": exp.source, "config_id": digest_value(cfg.to_dict()),
              "driver_sha256": DRIVER_SHA256,
              "refresh_interval": refresh_interval, "basis_refreshes": refreshes,
              "remap_interval": remap_interval, "basis_remaps": remaps,
              "hessian_operator": str(task.get("hessian_operator", "exact")),
              "drift_steps": drift_steps, "basis_drift": drifts,
              "drift_batch_offsets": drift_batch_offsets, "basis_drift_controls": drift_controls,
              "median_step_seconds": float(np.median(timings)), "mean_step_seconds": float(np.mean(timings)),
              "max_memory_bytes": torch.cuda.max_memory_allocated()}
    if exp.rank == 0:
        json_write(root / "result.json", result)
    barrier()
    return result, exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    task, = [t for t in manifest["tasks"] if t["id"] == args.task]
    snapshot = Path(task.get("snapshot", manifest["snapshot"]))
    cfg = load_config(snapshot / task["config"])
    try:
        if task["kind"] == "anchor":
            run_anchor_stage(cfg, task, args.output)
        elif task["kind"] == "anchor_wide":
            run_anchor_wide_stage(cfg, task, args.output)
        elif task["kind"] == "noise_profile":
            run_noise_profile_stage(cfg, task, args.output)
        elif task["kind"] == "branch":
            run_branch_stage(cfg, task, args.output)
        else:
            raise ValueError(f"driver does not run task kind {task['kind']}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
