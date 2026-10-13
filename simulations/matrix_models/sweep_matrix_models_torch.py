#!/usr/bin/env python3
"""Tune matrix optimizers on three finite-budget regression models.

The data distribution is a finite rank-one design for which minibatches can be
sampled exactly through multinomial and binomial sufficient statistics.  This
keeps the stochastic experiment faithful to minibatch sampling while avoiding
an explicit loop over the B examples in every update.

Each (model, batch, optimizer) task performs:

1. a broad Sobol/anchor hyperparameter search with common random numbers;
2. dense local refinement around several broad-search winners;
3. higher-chain selection among the combined finalists;
4. float64 finalist validation; and
5. an independent float64 report run for the selected winner.

The reported risk is exact population excess risk evaluated at the stochastic
terminal iterate.  Only the expectation over training trajectories is Monte
Carlo estimated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


MODELS = ("linear", "logistic", "normalized_linear")
OPTIMIZERS = ("sgd_hb", "signsgd_hb", "adam", "muon", "shampoo")
ADAPTIVE_OPTIMIZERS = {"adam", "shampoo"}


@dataclass(frozen=True)
class Config:
    eta: float
    beta1: float
    beta2: float = float("nan")


@dataclass
class EvalRow:
    eta: float
    beta1: float
    beta2: float
    risk: float
    risk_se: float
    chains: int


@dataclass
class Problem:
    left_basis: torch.Tensor
    right_basis: torch.Tensor
    left_eigs: torch.Tensor
    right_eigs: torch.Tensor
    feature_scale: torch.Tensor
    theta_star: torch.Tensor
    theta_star_tilde: torch.Tensor
    q_star: torch.Tensor
    q_star_tilde: torch.Tensor
    q_init: torch.Tensor
    logistic_p_star: torch.Tensor
    logistic_target_entropy: torch.Tensor
    sigma_epsilon: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/matrix_models"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--batches", default="2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260709)

    parser.add_argument("--broad-configs", type=int, default=128)
    parser.add_argument("--local-configs", type=int, default=160)
    parser.add_argument("--local-seeds", type=int, default=4)
    parser.add_argument("--search-chains", type=int, default=8)
    parser.add_argument("--refine-chains", type=int, default=24)
    parser.add_argument("--select-chains", type=int, default=96)
    parser.add_argument("--select-pool", type=int, default=12)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--validation-chains", type=int, default=1024)
    parser.add_argument("--report-chains", type=int, default=4096)
    parser.add_argument("--validation-chunk", type=int, default=256)
    parser.add_argument(
        "--search-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="dtype used for broad, local, and selection sweeps",
    )
    parser.add_argument("--skip-validation", action="store_true")

    parser.add_argument("--eta-min", type=float, default=1.0e-5)
    parser.add_argument("--eta-max", type=float, default=2.0)
    parser.add_argument("--beta1-gap-min", type=float, default=1.0e-4)
    parser.add_argument("--beta2-gap-min", type=float, default=1.0e-5)
    parser.add_argument("--local-eta-decades", type=float, default=0.8)
    parser.add_argument("--local-beta1-decades", type=float, default=1.2)
    parser.add_argument("--local-beta2-decades", type=float, default=1.5)

    parser.add_argument(
        "--sigma-epsilon",
        type=float,
        default=1.0,
        help="additive label-noise standard deviation for squared-loss models",
    )
    parser.add_argument(
        "--logistic-temperature",
        type=float,
        default=1.0,
        help="temperature dividing the logistic teacher logits",
    )
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--shampoo-damping", type=float, default=1.0e-4)
    parser.add_argument("--shampoo-abs-eps", type=float, default=1.0e-12)
    parser.add_argument("--max-parameter-norm", type=float, default=1.0e4)
    return parser.parse_args()


def parse_batches(raw: str) -> list[int]:
    batches = sorted({int(item) for item in raw.split(",") if item.strip()})
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("batches must be positive integers")
    return batches


def choose_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.arange(size, device=device, dtype=dtype)[:, None]
    cols = torch.arange(size, device=device, dtype=dtype)[None, :]
    basis = math.sqrt(2.0 / size) * torch.cos(math.pi * (cols + 0.5) * rows / size)
    basis[0] = 1.0 / math.sqrt(size)
    return basis.T


def make_problem(
    device: torch.device,
    dtype: torch.dtype,
    sigma_epsilon: float,
    logistic_temperature: float,
) -> Problem:
    left_basis = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ) / 2.0
    right_basis = dct_basis(4, device, dtype)
    left_eigs = torch.tensor([0.25, 0.5, 1.0, 2.0], device=device, dtype=dtype)
    right_eigs = torch.tensor([0.4, 0.7, 1.3, 2.5], device=device, dtype=dtype)
    feature_scale = torch.sqrt(
        16.0 * left_eigs[:, None] * right_eigs[None, :]
    )

    raw_target_tilde = torch.tensor(
        [
            [0.60, -0.25, 0.12, 0.05],
            [-0.30, 0.40, -0.20, 0.10],
            [0.15, -0.25, 0.35, -0.15],
            [0.05, 0.10, -0.20, 0.30],
        ],
        device=device,
        dtype=dtype,
    )
    score_variance = (
        left_eigs[:, None] * right_eigs[None, :] * raw_target_tilde.square()
    ).sum()
    theta_star_tilde = raw_target_tilde / torch.sqrt(score_variance)
    theta_star = left_basis @ theta_star_tilde @ right_basis.T
    q_star = theta_star / torch.linalg.vector_norm(theta_star)
    q_star_tilde = left_basis.T @ q_star @ right_basis

    raw_perp_tilde = torch.tensor(
        [
            [-0.10, 0.35, 0.20, -0.45],
            [0.30, 0.05, -0.40, 0.20],
            [-0.25, 0.20, 0.10, 0.30],
            [0.40, -0.15, 0.25, -0.05],
        ],
        device=device,
        dtype=dtype,
    )
    raw_perp = left_basis @ raw_perp_tilde @ right_basis.T
    raw_perp = raw_perp - (raw_perp * q_star).sum() * q_star
    raw_perp = raw_perp / torch.linalg.vector_norm(raw_perp)
    cosine = 0.25
    q_init = cosine * q_star + math.sqrt(1.0 - cosine * cosine) * raw_perp
    q_init = q_init / torch.linalg.vector_norm(q_init)

    signs = torch.tensor([-1.0, 1.0], device=device, dtype=dtype)[:, None, None]
    target_logits = (
        signs
        * feature_scale[None, :, :]
        * theta_star_tilde[None, :, :]
        / logistic_temperature
    )
    logistic_p_star = torch.sigmoid(target_logits)
    tiny = torch.finfo(dtype).eps
    p_clip = logistic_p_star.clamp(tiny, 1.0 - tiny)
    logistic_target_entropy = -(
        p_clip * torch.log(p_clip) + (1.0 - p_clip) * torch.log1p(-p_clip)
    ).mean()

    return Problem(
        left_basis=left_basis,
        right_basis=right_basis,
        left_eigs=left_eigs,
        right_eigs=right_eigs,
        feature_scale=feature_scale,
        theta_star=theta_star,
        theta_star_tilde=theta_star_tilde,
        q_star=q_star,
        q_star_tilde=q_star_tilde,
        q_init=q_init,
        logistic_p_star=logistic_p_star,
        logistic_target_entropy=logistic_target_entropy,
        sigma_epsilon=sigma_epsilon,
    )


def configs_to_tensors(
    configs: list[Config],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eta = torch.tensor([item.eta for item in configs], device=device, dtype=dtype)
    beta1 = torch.tensor([item.beta1 for item in configs], device=device, dtype=dtype)
    beta2 = torch.tensor(
        [0.0 if math.isnan(item.beta2) else item.beta2 for item in configs],
        device=device,
        dtype=dtype,
    )
    return eta, beta1, beta2


def dedupe_configs(configs: Iterable[Config]) -> list[Config]:
    seen: set[tuple[int, int, int]] = set()
    out: list[Config] = []
    for config in configs:
        key = (
            round(math.log10(config.eta) * 1.0e9),
            round(config.beta1 * 1.0e9),
            -1 if math.isnan(config.beta2) else round(config.beta2 * 1.0e9),
        )
        if key not in seen:
            seen.add(key)
            out.append(config)
    return out


def unit_to_config(
    values: torch.Tensor,
    optimizer: str,
    args: argparse.Namespace,
) -> Config:
    log_eta = math.log10(args.eta_min) + float(values[0]) * (
        math.log10(args.eta_max) - math.log10(args.eta_min)
    )
    log_gap1 = math.log10(args.beta1_gap_min) * float(values[1])
    beta1 = 1.0 - 10.0**log_gap1
    if optimizer not in ADAPTIVE_OPTIMIZERS:
        return Config(eta=10.0**log_eta, beta1=beta1)
    log_gap2 = math.log10(args.beta2_gap_min) * float(values[2])
    beta2 = 1.0 - 10.0**log_gap2
    return Config(eta=10.0**log_eta, beta1=beta1, beta2=beta2)


def broad_configs(
    optimizer: str,
    args: argparse.Namespace,
    seed: int,
) -> list[Config]:
    dim = 3 if optimizer in ADAPTIVE_OPTIMIZERS else 2
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    draws = engine.draw(args.broad_configs)
    configs = [unit_to_config(row, optimizer, args) for row in draws]

    eta_anchors = [1.0e-4, 1.0e-3, 1.0e-2, 0.05, 0.2, 0.8]
    eta_anchors = [
        min(max(value, args.eta_min), args.eta_max) for value in eta_anchors
    ]
    beta1_anchors = [0.0, 0.5, 0.9, 0.99]
    if optimizer not in ADAPTIVE_OPTIMIZERS:
        configs.extend(
            Config(eta=eta, beta1=beta1)
            for eta in eta_anchors
            for beta1 in beta1_anchors
        )
    else:
        beta2_anchors = [0.0, 0.9, 0.99, 0.999]
        configs.extend(
            Config(eta=eta, beta1=beta1, beta2=beta2)
            for eta in eta_anchors
            for beta1 in beta1_anchors
            for beta2 in beta2_anchors
        )
    return dedupe_configs(configs)


def clamp_log(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def local_configs(
    seeds: list[Config],
    optimizer: str,
    args: argparse.Namespace,
    seed: int,
) -> list[Config]:
    if not seeds:
        return []
    dim = 3 if optimizer in ADAPTIVE_OPTIMIZERS else 2
    per_seed = max(1, math.ceil(args.local_configs / len(seeds)))
    configs: list[Config] = list(seeds)
    log_eta_lo = math.log10(args.eta_min)
    log_eta_hi = math.log10(args.eta_max)
    log_gap1_lo = math.log10(args.beta1_gap_min)
    log_gap2_lo = math.log10(args.beta2_gap_min)

    for seed_idx, center in enumerate(seeds):
        engine = torch.quasirandom.SobolEngine(
            dim, scramble=True, seed=seed + 7919 * seed_idx
        )
        draws = engine.draw(per_seed)
        center_log_eta = math.log10(center.eta)
        center_log_gap1 = math.log10(max(1.0 - center.beta1, args.beta1_gap_min))
        center_log_gap2 = (
            math.log10(max(1.0 - center.beta2, args.beta2_gap_min))
            if optimizer in ADAPTIVE_OPTIMIZERS
            else 0.0
        )
        for row in draws:
            log_eta = clamp_log(
                center_log_eta
                + (2.0 * float(row[0]) - 1.0) * args.local_eta_decades,
                log_eta_lo,
                log_eta_hi,
            )
            log_gap1 = clamp_log(
                center_log_gap1
                + (2.0 * float(row[1]) - 1.0) * args.local_beta1_decades,
                log_gap1_lo,
                0.0,
            )
            beta1 = 1.0 - 10.0**log_gap1
            if optimizer not in ADAPTIVE_OPTIMIZERS:
                configs.append(Config(eta=10.0**log_eta, beta1=beta1))
                continue
            log_gap2 = clamp_log(
                center_log_gap2
                + (2.0 * float(row[2]) - 1.0) * args.local_beta2_decades,
                log_gap2_lo,
                0.0,
            )
            beta2 = 1.0 - 10.0**log_gap2
            configs.append(
                Config(eta=10.0**log_eta, beta1=beta1, beta2=beta2)
            )
    return dedupe_configs(configs)


def uniform_multinomial(
    total_count: int,
    categories: int,
    chains: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    remaining = torch.full(
        (chains,), float(total_count), device=device, dtype=dtype
    )
    counts = torch.empty((chains, categories), device=device, dtype=dtype)
    for category in range(categories - 1):
        probability = torch.full_like(remaining, 1.0 / (categories - category))
        draw = torch.binomial(remaining, probability, generator=generator)
        counts[:, category] = draw
        remaining = remaining - draw
    counts[:, -1] = remaining
    return counts


def to_eigenbasis(theta: torch.Tensor, problem: Problem) -> torch.Tensor:
    return problem.left_basis.T @ theta @ problem.right_basis


def from_eigenbasis(theta: torch.Tensor, problem: Problem) -> torch.Tensor:
    return problem.left_basis @ theta @ problem.right_basis.T


def sample_gradient(
    model: str,
    theta: torch.Tensor,
    batch: int,
    problem: Problem,
    generator: torch.Generator,
) -> torch.Tensor:
    configs, chains, rows, cols = theta.shape
    theta_tilde = to_eigenbasis(theta, problem)
    dimensions = rows * cols

    if model in {"linear", "normalized_linear"}:
        counts = uniform_multinomial(
            batch, dimensions, chains, theta.device, theta.dtype, generator
        ).reshape(chains, rows, cols)
        noise = torch.randn(
            (chains, rows, cols),
            device=theta.device,
            dtype=theta.dtype,
            generator=generator,
        )
        if model == "linear":
            target = problem.theta_star_tilde
            state = theta_tilde
            radius = None
            q = None
        else:
            radius = torch.linalg.vector_norm(theta, dim=(-2, -1), keepdim=True)
            safe_radius = torch.clamp(radius, min=1.0e-12)
            q = theta / safe_radius
            state = to_eigenbasis(q, problem)
            target = problem.q_star_tilde

        signal = (
            counts[None, :, :, :]
            * problem.feature_scale.square()[None, None, :, :]
            * (state - target)
            / float(batch)
        )
        label_noise = (
            problem.sigma_epsilon
            * torch.sqrt(counts)[None, :, :, :]
            * problem.feature_scale[None, None, :, :]
            * noise[None, :, :, :]
            / float(batch)
        )
        gradient = from_eigenbasis(signal - label_noise, problem)
        if model == "normalized_linear":
            assert radius is not None and q is not None
            radial = (gradient * q).sum(dim=(-2, -1), keepdim=True)
            gradient = (gradient - radial * q) / torch.clamp(radius, min=1.0e-12)
        return gradient

    if model == "logistic":
        categories = 2 * dimensions
        counts = uniform_multinomial(
            batch, categories, chains, theta.device, theta.dtype, generator
        ).reshape(chains, 2, rows, cols)
        p_star = problem.logistic_p_star[None, :, :, :].expand(
            chains, 2, rows, cols
        )
        positive = torch.binomial(counts, p_star, generator=generator)
        signs = torch.tensor(
            [-1.0, 1.0], device=theta.device, dtype=theta.dtype
        ).view(1, 1, 2, 1, 1)
        logits = (
            signs
            * problem.feature_scale.view(1, 1, 1, rows, cols)
            * theta_tilde[:, :, None, :, :]
        )
        probabilities = torch.sigmoid(logits)
        residual_sum = (
            counts[None, :, :, :, :] * probabilities
            - positive[None, :, :, :, :]
        )
        gradient_tilde = (
            residual_sum
            * signs
            * problem.feature_scale.view(1, 1, 1, rows, cols)
        ).sum(dim=2) / float(batch)
        return from_eigenbasis(gradient_tilde, problem)

    raise ValueError(model)


def population_excess_risk(
    model: str,
    theta: torch.Tensor,
    problem: Problem,
) -> torch.Tensor:
    if model == "linear":
        delta = to_eigenbasis(theta, problem) - problem.theta_star_tilde
        return 0.5 * (
            problem.left_eigs[None, None, :, None]
            * problem.right_eigs[None, None, None, :]
            * delta.square()
        ).sum(dim=(-2, -1))

    if model == "normalized_linear":
        radius = torch.linalg.vector_norm(theta, dim=(-2, -1), keepdim=True)
        q = theta / torch.clamp(radius, min=1.0e-12)
        delta = to_eigenbasis(q, problem) - problem.q_star_tilde
        return 0.5 * (
            problem.left_eigs[None, None, :, None]
            * problem.right_eigs[None, None, None, :]
            * delta.square()
        ).sum(dim=(-2, -1))

    if model == "logistic":
        theta_tilde = to_eigenbasis(theta, problem)
        signs = torch.tensor(
            [-1.0, 1.0], device=theta.device, dtype=theta.dtype
        ).view(1, 1, 2, 1, 1)
        logits = (
            signs
            * problem.feature_scale.view(1, 1, 1, 4, 4)
            * theta_tilde[:, :, None, :, :]
        )
        p_star = problem.logistic_p_star.view(1, 1, 2, 4, 4)
        cross_entropy = (
            p_star * F.softplus(-logits)
            + (1.0 - p_star) * F.softplus(logits)
        ).mean(dim=(-3, -2, -1))
        return torch.clamp(
            cross_entropy - problem.logistic_target_entropy, min=0.0
        )

    raise ValueError(model)


def inverse_fourth_root(
    matrix: torch.Tensor,
    relative_damping: float,
    absolute_epsilon: float,
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    dimension = matrix.shape[-1]
    trace_scale = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / dimension
    damping = relative_damping * trace_scale + absolute_epsilon
    inverse_root = torch.pow(
        torch.clamp(eigenvalues + damping[..., None], min=absolute_epsilon),
        -0.25,
    )
    return (
        eigenvectors * inverse_root.unsqueeze(-2)
    ) @ eigenvectors.transpose(-2, -1)


def optimizer_direction(
    optimizer: str,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor | None,
    left_gram: torch.Tensor | None,
    right_gram: torch.Tensor | None,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    step: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    beta1_view = beta1[:, None, None, None]
    momentum_hat = momentum / torch.clamp(
        1.0 - torch.pow(beta1, step + 1), min=1.0e-12
    )[:, None, None, None]

    if optimizer == "sgd_hb":
        return momentum, second_moment, left_gram, right_gram
    if optimizer == "signsgd_hb":
        return torch.sign(momentum), second_moment, left_gram, right_gram
    if optimizer == "muon":
        safe_momentum = torch.nan_to_num(momentum)
        left, _, right_h = torch.linalg.svd(safe_momentum, full_matrices=False)
        direction = left @ right_h
        nonzero = torch.linalg.vector_norm(
            safe_momentum, dim=(-2, -1), keepdim=True
        ) > 1.0e-20
        return (
            torch.where(nonzero, direction, torch.zeros_like(direction)),
            second_moment,
            left_gram,
            right_gram,
        )
    if optimizer == "adam":
        assert second_moment is not None
        beta2_view = beta2[:, None, None, None]
        second_moment = (
            beta2_view * second_moment
            + (1.0 - beta2_view) * gradient.square()
        )
        second_hat = second_moment / torch.clamp(
            1.0 - torch.pow(beta2, step + 1), min=1.0e-12
        )[:, None, None, None]
        direction = momentum_hat / (torch.sqrt(second_hat) + args.adam_eps)
        return direction, second_moment, left_gram, right_gram
    if optimizer == "shampoo":
        assert left_gram is not None and right_gram is not None
        beta2_left = beta2[:, None, None, None]
        left_gram = (
            beta2_left * left_gram
            + (1.0 - beta2_left) * (gradient @ gradient.transpose(-2, -1))
        )
        right_gram = (
            beta2_left * right_gram
            + (1.0 - beta2_left) * (gradient.transpose(-2, -1) @ gradient)
        )
        correction = torch.clamp(
            1.0 - torch.pow(beta2, step + 1), min=1.0e-12
        )[:, None, None, None]
        left_hat = left_gram / correction
        right_hat = right_gram / correction
        left_inverse = inverse_fourth_root(
            left_hat, args.shampoo_damping, args.shampoo_abs_eps
        )
        right_inverse = inverse_fourth_root(
            right_hat, args.shampoo_damping, args.shampoo_abs_eps
        )
        direction = left_inverse @ momentum_hat @ right_inverse
        return direction, second_moment, left_gram, right_gram
    raise ValueError(optimizer)


def simulate_losses(
    model: str,
    optimizer: str,
    batch: int,
    steps: int,
    configs: list[Config],
    chains: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> torch.Tensor:
    if not configs:
        return torch.empty((0, chains), dtype=torch.float64)
    problem = make_problem(
        device,
        dtype,
        args.sigma_epsilon,
        args.logistic_temperature,
    )
    eta, beta1, beta2 = configs_to_tensors(configs, device, dtype)
    num_configs = len(configs)
    if model == "normalized_linear":
        initial = problem.q_init
    else:
        initial = torch.zeros((4, 4), device=device, dtype=dtype)
    theta = initial.view(1, 1, 4, 4).expand(num_configs, chains, 4, 4).clone()
    momentum = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta) if optimizer == "adam" else None
    left_gram = (
        torch.zeros((num_configs, chains, 4, 4), device=device, dtype=dtype)
        if optimizer == "shampoo"
        else None
    )
    right_gram = (
        torch.zeros((num_configs, chains, 4, 4), device=device, dtype=dtype)
        if optimizer == "shampoo"
        else None
    )
    valid = torch.ones((num_configs, chains), device=device, dtype=torch.bool)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    beta1_view = beta1[:, None, None, None]
    eta_view = eta[:, None, None, None]
    safe_initial = initial.view(1, 1, 4, 4)
    for step in range(steps):
        gradient = sample_gradient(model, theta, batch, problem, generator)
        momentum = (
            beta1_view * momentum + (1.0 - beta1_view) * gradient
        )
        direction, second_moment, left_gram, right_gram = optimizer_direction(
            optimizer,
            gradient,
            momentum,
            second_moment,
            left_gram,
            right_gram,
            beta1,
            beta2,
            step,
            args,
        )
        theta = theta - eta_view * direction

        parameter_norm = torch.linalg.vector_norm(theta, dim=(-2, -1))
        step_valid = torch.isfinite(theta).all(dim=(-2, -1))
        step_valid &= parameter_norm < args.max_parameter_norm
        if model == "normalized_linear":
            step_valid &= parameter_norm > 1.0e-10
        valid &= step_valid
        safe_mask = step_valid[:, :, None, None]
        theta = torch.where(safe_mask, theta, safe_initial)
        momentum = torch.where(safe_mask, momentum, torch.zeros_like(momentum))
        if second_moment is not None:
            second_moment = torch.where(
                safe_mask, second_moment, torch.zeros_like(second_moment)
            )
        if left_gram is not None:
            left_gram = torch.where(
                safe_mask, left_gram, torch.zeros_like(left_gram)
            )
        if right_gram is not None:
            right_gram = torch.where(
                safe_mask, right_gram, torch.zeros_like(right_gram)
            )

    losses = population_excess_risk(model, theta, problem)
    losses = torch.where(valid, losses, torch.full_like(losses, 1.0e12))
    return losses.detach().to(device="cpu", dtype=torch.float64)


def evaluate_configs(
    model: str,
    optimizer: str,
    batch: int,
    steps: int,
    configs: list[Config],
    chains: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[EvalRow]:
    losses = simulate_losses(
        model,
        optimizer,
        batch,
        steps,
        configs,
        chains,
        seed,
        device,
        dtype,
        args,
    )
    means = losses.mean(dim=1)
    if chains > 1:
        standard_errors = losses.std(dim=1, unbiased=True) / math.sqrt(chains)
    else:
        standard_errors = torch.zeros_like(means)
    rows = [
        EvalRow(
            eta=config.eta,
            beta1=config.beta1,
            beta2=config.beta2,
            risk=float(mean),
            risk_se=float(standard_error),
            chains=chains,
        )
        for config, mean, standard_error in zip(
            configs, means, standard_errors, strict=True
        )
    ]
    return sorted(rows, key=lambda item: item.risk)


def eval_row_to_config(row: EvalRow) -> Config:
    return Config(eta=row.eta, beta1=row.beta1, beta2=row.beta2)


def validate_in_chunks(
    model: str,
    optimizer: str,
    batch: int,
    steps: int,
    configs: list[Config],
    total_chains: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[EvalRow]:
    total_sum = torch.zeros(len(configs), dtype=torch.float64)
    total_sumsq = torch.zeros(len(configs), dtype=torch.float64)
    total_count = 0
    chunk_id = 0
    while total_count < total_chains:
        chains = min(chunk_size, total_chains - total_count)
        losses = simulate_losses(
            model,
            optimizer,
            batch,
            steps,
            configs,
            chains,
            seed + 1000003 * chunk_id,
            device,
            dtype,
            args,
        )
        total_sum += losses.sum(dim=1)
        total_sumsq += losses.square().sum(dim=1)
        total_count += chains
        chunk_id += 1
    means = total_sum / total_count
    variances = torch.clamp(total_sumsq / total_count - means.square(), min=0.0)
    standard_errors = torch.sqrt(variances / total_count)
    rows = [
        EvalRow(
            eta=config.eta,
            beta1=config.beta1,
            beta2=config.beta2,
            risk=float(mean),
            risk_se=float(standard_error),
            chains=total_count,
        )
        for config, mean, standard_error in zip(
            configs, means, standard_errors, strict=True
        )
    ]
    return sorted(rows, key=lambda item: item.risk)


def task_seed(
    base_seed: int,
    model: str,
    optimizer: str,
    batch: int,
    stage: int,
) -> int:
    return (
        base_seed
        + 10000019 * MODELS.index(model)
        + 100003 * OPTIMIZERS.index(optimizer)
        + 1009 * batch
        + 9176 * stage
    )


def top_unique_configs(rows: list[EvalRow], count: int) -> list[Config]:
    return dedupe_configs(eval_row_to_config(row) for row in rows[:count])


def row_dict(row: EvalRow) -> dict[str, float | int]:
    return asdict(row)


def run_task(
    model: str,
    optimizer: str,
    batch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    if args.T % batch != 0:
        raise ValueError(f"T={args.T} must be divisible by batch={batch}")
    steps = args.T // batch
    start_time = time.time()
    search_dtype = (
        torch.float64 if args.search_dtype == "float64" else torch.float32
    )
    validation_dtype = torch.float64

    broad = broad_configs(
        optimizer, args, task_seed(args.seed, model, optimizer, batch, 0)
    )
    broad_rows = evaluate_configs(
        model,
        optimizer,
        batch,
        steps,
        broad,
        args.search_chains,
        task_seed(args.seed, model, optimizer, batch, 1),
        device,
        search_dtype,
        args,
    )
    local_seed_configs = top_unique_configs(broad_rows, args.local_seeds)
    local = local_configs(
        local_seed_configs,
        optimizer,
        args,
        task_seed(args.seed, model, optimizer, batch, 2),
    )
    local_rows = evaluate_configs(
        model,
        optimizer,
        batch,
        steps,
        local,
        args.refine_chains,
        task_seed(args.seed, model, optimizer, batch, 3),
        device,
        search_dtype,
        args,
    )

    select_pool = dedupe_configs(
        [
            *top_unique_configs(broad_rows, args.select_pool // 2),
            *top_unique_configs(local_rows, args.select_pool),
        ]
    )
    select_rows = evaluate_configs(
        model,
        optimizer,
        batch,
        steps,
        select_pool,
        args.select_chains,
        task_seed(args.seed, model, optimizer, batch, 4),
        device,
        search_dtype,
        args,
    )
    finalists = top_unique_configs(select_rows, args.finalists)

    if args.skip_validation:
        validation_rows = select_rows[: args.finalists]
        winner = validation_rows[0]
        report = winner
        validation_method = "search_only_float32"
    else:
        validation_rows = validate_in_chunks(
            model,
            optimizer,
            batch,
            steps,
            finalists,
            args.validation_chains,
            args.validation_chunk,
            task_seed(args.seed, model, optimizer, batch, 5),
            device,
            validation_dtype,
            args,
        )
        winner_config = eval_row_to_config(validation_rows[0])
        report = validate_in_chunks(
            model,
            optimizer,
            batch,
            steps,
            [winner_config],
            args.report_chains,
            args.validation_chunk,
            task_seed(args.seed, model, optimizer, batch, 6),
            device,
            validation_dtype,
            args,
        )[0]
        winner = validation_rows[0]
        validation_method = "independent_exact_minibatch_mc_float64"

    elapsed = time.time() - start_time
    return {
        "model": model,
        "optimizer": optimizer,
        "batch": batch,
        "steps": steps,
        "T": args.T,
        "winner": row_dict(winner),
        "report": row_dict(report),
        "validation_method": validation_method,
        "broad_top": [row_dict(row) for row in broad_rows[:20]],
        "local_top": [row_dict(row) for row in local_rows[:20]],
        "selection": [row_dict(row) for row in select_rows[:20]],
        "validation": [row_dict(row) for row in validation_rows],
        "counts": {
            "broad_configs": len(broad),
            "local_configs": len(local),
            "select_pool": len(select_pool),
            "finalists": len(finalists),
        },
        "elapsed_seconds": elapsed,
        "device": str(device),
    }


def all_tasks(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    batches = parse_batches(args.batches)
    tasks = [
        (model, optimizer, batch)
        for batch in batches
        for model in args.models
        for optimizer in args.optimizers
    ]
    cost = {"sgd_hb": 1, "signsgd_hb": 1, "adam": 2, "muon": 3, "shampoo": 4}
    tasks.sort(key=lambda item: (-args.T // item[2], -cost[item[1]], item[0], item[1]))
    return [
        task
        for index, task in enumerate(tasks)
        if index % args.num_shards == args.shard_index
    ]


def task_path(outdir: Path, model: str, optimizer: str, batch: int) -> Path:
    return outdir / f"{model}__{optimizer}__B{batch:04d}.json"


def main() -> None:
    args = parse_args()
    if args.sigma_epsilon <= 0.0:
        raise ValueError("sigma-epsilon must be positive")
    if args.logistic_temperature <= 0.0:
        raise ValueError("logistic-temperature must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0, num-shards)")
    device = choose_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    tasks = all_tasks(args)
    print(
        json.dumps(
            {
                "device": str(device),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "tasks": len(tasks),
            }
        ),
        flush=True,
    )
    completed = 0
    for model, optimizer, batch in tasks:
        destination = task_path(args.outdir, model, optimizer, batch)
        print(
            f"START model={model} optimizer={optimizer} batch={batch}",
            flush=True,
        )
        result = run_task(model, optimizer, batch, args, device)
        payload = {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            **result,
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        completed += 1
        report = result["report"]
        assert isinstance(report, dict)
        print(
            "DONE "
            f"model={model} optimizer={optimizer} batch={batch} "
            f"risk={report['risk']:.8g} se={report['risk_se']:.3g} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    summary = {
        "completed": completed,
        "tasks": len(tasks),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device": str(device),
    }
    (args.outdir / f"shard_{args.shard_index:03d}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
