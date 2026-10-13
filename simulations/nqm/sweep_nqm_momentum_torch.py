#!/usr/bin/env python3
"""GPU finite-budget NQM sweep with first-moment momentum.

This script estimates tuned finite-budget risk for four optimizer families on
the three-coordinate noisy quadratic used in the paper.  Every optimizer gets a
first-moment/heavyball parameter beta1 and a learning rate eta.  The adaptive
optimizer is Adam-style: momentum is applied to the raw noisy gradient before
division by the square-root second moment.  SignSGD also applies momentum before
taking the sign.

The simulation is Monte Carlo but vectorized over hyperparameter grids on a
single device (--device, default cuda; CPU when CUDA is unavailable).  Use
--shard-index/--num-shards to split batch sizes across multiple processes/GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import torch


OPTIMIZERS = ("sgd_hb", "newton_hb", "signsgd_hb", "adam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/nqm_momentum_sweep"))
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=49)
    parser.add_argument("--steps-min", type=int, default=4)
    parser.add_argument("--steps-max", type=int, default=None)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--chains", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-configs-per-chunk", type=int, default=4096)
    parser.add_argument("--divergence-threshold", type=float, default=1.0e12)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--no-adam-bias-correction", action="store_true")

    parser.add_argument("--eta-points", type=int, default=80)
    parser.add_argument("--beta1-points", type=int, default=25)
    parser.add_argument("--beta2-points", type=int, default=21)
    parser.add_argument("--beta-gap-min", type=float, default=1.0e-3)
    parser.add_argument("--eta-min", type=float, default=1.0e-4)
    parser.add_argument("--sgd-eta-max", type=float, default=1.0)
    parser.add_argument("--newton-eta-max", type=float, default=2.0)
    parser.add_argument("--sign-eta-min", type=float, default=5.0e-4)
    parser.add_argument("--sign-eta-max", type=float, default=1.2)
    parser.add_argument("--adam-eta-max", type=float, default=2.0)
    parser.add_argument("--alpha-values", default="0.0")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def choose_device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def make_steps(args: argparse.Namespace) -> list[int]:
    steps_max = args.steps_max if args.steps_max is not None else args.T // 2
    raw = torch.logspace(
        math.log10(args.steps_min),
        math.log10(steps_max),
        args.num_points,
        dtype=torch.float64,
    )
    steps = sorted({int(round(float(x))) for x in raw if int(round(float(x))) >= 1})
    return [step for idx, step in enumerate(steps) if idx % args.num_shards == args.shard_index]


def log_grid(lo: float, hi: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError(f"invalid log grid [{lo}, {hi}]")
    return torch.logspace(math.log10(lo), math.log10(hi), n, device=device, dtype=dtype)


def beta_grid(n: int, gap_min: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = 1.0 - torch.logspace(0.0, math.log10(gap_min), n, device=device, dtype=dtype)
    anchors = torch.tensor([0.0, 0.5, 0.9, 0.99, 0.999], device=device, dtype=dtype)
    grid = torch.cat([base, anchors])
    grid = torch.clamp(grid, 0.0, 1.0 - gap_min)
    return torch.unique(grid, sorted=True)


def alpha_grid(values: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    parsed = [float(item) for item in values.split(",") if item.strip()]
    if not parsed:
        parsed = [0.0]
    grid = torch.tensor(parsed, device=device, dtype=dtype)
    grid = torch.clamp(grid, 0.0, 1.0)
    return torch.unique(grid, sorted=True)


def optimizer_eta_grid(
    optimizer: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if optimizer == "signsgd_hb":
        return log_grid(args.sign_eta_min, args.sign_eta_max, args.eta_points, device, dtype)
    hi = {
        "sgd_hb": args.sgd_eta_max,
        "newton_hb": args.newton_eta_max,
        "adam": args.adam_eta_max,
    }[optimizer]
    return log_grid(args.eta_min, hi, args.eta_points, device, dtype)


def make_configs(
    optimizer: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    eta = optimizer_eta_grid(optimizer, args, device, dtype)
    beta1 = beta_grid(args.beta1_points, args.beta_gap_min, device, dtype)
    alpha = alpha_grid(args.alpha_values, device, dtype)
    if optimizer != "adam":
        eta_mesh, beta1_mesh, alpha_mesh = torch.meshgrid(eta, beta1, alpha, indexing="ij")
        return eta_mesh.reshape(-1), beta1_mesh.reshape(-1), None, alpha_mesh.reshape(-1)

    beta2 = beta_grid(args.beta2_points, args.beta_gap_min, device, dtype)
    eta_mesh, beta1_mesh, beta2_mesh, alpha_mesh = torch.meshgrid(eta, beta1, beta2, alpha, indexing="ij")
    return eta_mesh.reshape(-1), beta1_mesh.reshape(-1), beta2_mesh.reshape(-1), alpha_mesh.reshape(-1)


def sign_no_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def risk_for_chunk(
    optimizer: str,
    eta: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor | None,
    alpha: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    w0: torch.Tensor,
    batch: float,
    steps: int,
    args: argparse.Namespace,
    generator_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    configs = eta.numel()
    chains = args.chains
    d = h.numel()
    h_view = h.view(1, 1, d)
    c_view = c.view(1, 1, d)
    eta_view = eta.view(configs, 1, 1)
    beta1_view = beta1.view(configs, 1, 1)
    rho_view = (1.0 - alpha).view(configs, 1, 1)
    w = w0.view(1, 1, d).expand(configs, chains, d).clone()
    m = torch.zeros_like(w)
    v = torch.zeros_like(w) if optimizer == "adam" else None
    beta2_view = beta2.view(configs, 1, 1) if beta2 is not None else None
    sigma = torch.sqrt(c_view / batch)

    generator = torch.Generator(device=device)
    generator.manual_seed(generator_seed)
    beta1_power = torch.ones_like(beta1_view)
    beta2_power = torch.ones_like(beta2_view) if beta2_view is not None else None
    bias_correct = optimizer == "adam" and not args.no_adam_bias_correction

    for step in range(1, steps + 1):
        noise = torch.randn((chains, d), device=device, dtype=dtype, generator=generator)
        g = h_view * w + sigma * noise.view(1, chains, d)

        if optimizer == "sgd_hb":
            m = beta1_view * m + (1.0 - beta1_view) * g
            w = rho_view * w - eta_view * m
        elif optimizer == "newton_hb":
            newton_g = g / h_view
            m = beta1_view * m + (1.0 - beta1_view) * newton_g
            w = rho_view * w - eta_view * m
        elif optimizer == "signsgd_hb":
            m = beta1_view * m + (1.0 - beta1_view) * g
            w = rho_view * w - eta_view * sign_no_zero(m)
        elif optimizer == "adam":
            assert v is not None and beta2_view is not None and beta2_power is not None
            m = beta1_view * m + (1.0 - beta1_view) * g
            v = beta2_view * v + (1.0 - beta2_view) * g.square()
            if bias_correct:
                beta1_power = beta1_power * beta1_view
                beta2_power = beta2_power * beta2_view
                m_hat = m / torch.clamp(1.0 - beta1_power, min=args.eps)
                v_hat = v / torch.clamp(1.0 - beta2_power, min=args.eps)
            else:
                m_hat = m
                v_hat = v
            w = rho_view * w - eta_view * m_hat / (torch.sqrt(v_hat) + args.eps)
        else:
            raise ValueError(optimizer)

        if step % 128 == 0 or step == steps:
            bad = (~torch.isfinite(w)).any(dim=(1, 2))
            if bad.any():
                w[bad] = args.divergence_threshold
                if m is not None:
                    m[bad] = 0.0
                if v is not None:
                    v[bad] = args.divergence_threshold

    risk = 0.5 * (h_view * w.square()).sum(dim=2).mean(dim=1)
    risk = torch.where(torch.isfinite(risk), risk, torch.full_like(risk, float("inf")))
    risk = torch.where(risk > args.divergence_threshold, torch.full_like(risk, float("inf")), risk)
    return risk


def evaluate_optimizer(
    optimizer: str,
    batch: float,
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int | str]:
    h = torch.tensor([0.2, 1.0, 5.0], device=device, dtype=dtype)
    c = args.noise_scale * torch.tensor([10.0, 1.0, 0.1], device=device, dtype=dtype)
    w0 = torch.tensor([0.3, 1.0, 3.0], device=device, dtype=dtype)
    eta_all, beta1_all, beta2_all, alpha_all = make_configs(optimizer, args, device, dtype)
    best = {
        "optimizer": optimizer,
        "batch": float(batch),
        "steps": int(steps),
        "eta": float("nan"),
        "beta1": float("nan"),
        "beta2": float("nan"),
        "alpha": float("nan"),
        "risk": float("inf"),
        "num_configs": int(eta_all.numel()),
    }

    optimizer_offset = {name: 10000 * idx for idx, name in enumerate(OPTIMIZERS)}[optimizer]
    base_seed = args.seed + 104729 * int(steps) + optimizer_offset
    chunk_size = args.max_configs_per_chunk
    for start in range(0, eta_all.numel(), chunk_size):
        end = min(start + chunk_size, eta_all.numel())
        beta2 = beta2_all[start:end] if beta2_all is not None else None
        alpha = alpha_all[start:end]
        risks = risk_for_chunk(
            optimizer,
            eta_all[start:end],
            beta1_all[start:end],
            beta2,
            alpha,
            h,
            c,
            w0,
            batch,
            steps,
            args,
            base_seed,
            device,
            dtype,
        )
        idx = int(torch.argmin(risks).item())
        risk = float(risks[idx].detach().cpu())
        if risk < float(best["risk"]):
            best["risk"] = risk
            best["eta"] = float(eta_all[start + idx].detach().cpu())
            best["beta1"] = float(beta1_all[start + idx].detach().cpu())
            best["beta2"] = (
                float(beta2_all[start + idx].detach().cpu()) if beta2_all is not None else float("nan")
            )
            best["alpha"] = float(alpha_all[start + idx].detach().cpu())

    return best


def write_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["optimizer", "batch", "steps", "eta", "beta1", "beta2", "alpha", "risk", "num_configs"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    device = choose_device(args.device)
    dtype = torch_dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(0)

    args.outdir.mkdir(parents=True, exist_ok=True)
    steps_list = make_steps(args)
    rows: list[dict[str, float | int | str]] = []
    for steps in steps_list:
        batch = args.T / float(steps)
        for optimizer in args.optimizers:
            rows.append(evaluate_optimizer(optimizer, batch, steps, args, device, dtype))
            print(json.dumps(rows[-1]), flush=True)

    shard_csv = args.outdir / f"best_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.csv"
    write_csv(shard_csv, rows)
    summary = {
        "out": str(shard_csv),
        "device": str(device),
        "rows": len(rows),
        "steps": steps_list,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.outdir / f"summary_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
