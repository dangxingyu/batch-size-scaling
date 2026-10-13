#!/usr/bin/env python3
"""Deterministic GPU finite-volume sweep for momentum NQM optimizers.

Deterministic counterpart of the Monte Carlo sweep in sweep_nqm_momentum_torch.py
(same optimizers, hyperparameter ranges and output CSV):

* SGD-M and Newton-M use exact deterministic second-moment recursions.
* SignSGD-M uses a finite-volume grid over (theta, m), with momentum before
  sign.
* Adam uses a finite-volume grid over (theta, m, v), with m on raw gradients,
  v on raw gradient squares, and the update m / sqrt(v).

The finite-volume transport uses deterministic three-point Gaussian quadrature
and nearest-cell conservative deposition.  The default cell and hyperparameter
grids are coarse; increase them for higher accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


OPTIMIZERS = ("sgd_hb", "newton_hb", "signsgd_hb", "adam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/nqm_momentum_fv"))
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=15)
    parser.add_argument("--steps-min", type=int, default=4)
    parser.add_argument("--steps-max", type=int, default=None)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--chunk-configs", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--no-adam-bias-correction", action="store_true")

    parser.add_argument("--eta-points", type=int, default=12)
    parser.add_argument("--beta1-points", type=int, default=5)
    parser.add_argument("--beta2-points", type=int, default=5)
    parser.add_argument("--beta-gap-min", type=float, default=1.0e-3)
    parser.add_argument("--eta-min", type=float, default=1.0e-4)
    parser.add_argument("--sgd-eta-max", type=float, default=1.0)
    parser.add_argument("--newton-eta-max", type=float, default=2.0)
    parser.add_argument("--sign-eta-min", type=float, default=5.0e-4)
    parser.add_argument("--sign-eta-max", type=float, default=1.2)
    parser.add_argument("--adam-eta-max", type=float, default=2.0)
    parser.add_argument("--alpha-values", default="0.0")
    parser.add_argument("--noise-scale", type=float, default=1.0)

    parser.add_argument("--sign-w-cells", type=int, default=41)
    parser.add_argument("--sign-m-cells", type=int, default=31)
    parser.add_argument("--adam-w-cells", type=int, default=25)
    parser.add_argument("--adam-m-cells", type=int, default=13)
    parser.add_argument("--adam-v-cells", type=int, default=13)
    parser.add_argument("--w-bound", type=float, default=6.0)
    parser.add_argument("--m-bound", type=float, default=18.0)
    parser.add_argument("--v-bound", type=float, default=324.0)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def choose_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def make_steps(args: argparse.Namespace) -> list[int]:
    steps_max = args.steps_max if args.steps_max is not None else args.T // 2
    raw = torch.logspace(
        math.log10(args.steps_min),
        math.log10(steps_max),
        args.num_points,
        dtype=torch.float64,
    )
    steps = sorted({max(1, int(round(float(x)))) for x in raw})
    return [step for idx, step in enumerate(steps) if idx % args.num_shards == args.shard_index]


def log_grid(lo: float, hi: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.logspace(math.log10(lo), math.log10(hi), n, device=device, dtype=dtype)


def beta_grid(n: int, gap_min: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    grid = 1.0 - torch.logspace(0.0, math.log10(gap_min), n, device=device, dtype=dtype)
    anchors = torch.tensor([0.0, 0.5, 0.9, 0.99, 0.999], device=device, dtype=dtype)
    return torch.unique(torch.clamp(torch.cat([grid, anchors]), 0.0, 1.0 - gap_min), sorted=True)


def alpha_grid(values: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    parsed = [float(item) for item in values.split(",") if item.strip()]
    if not parsed:
        parsed = [0.0]
    grid = torch.tensor(parsed, device=device, dtype=dtype)
    grid = torch.clamp(grid, 0.0, 1.0)
    return torch.unique(grid, sorted=True)


def make_configs(
    optimizer: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    eta_hi = {
        "sgd_hb": args.sgd_eta_max,
        "newton_hb": args.newton_eta_max,
        "signsgd_hb": args.sign_eta_max,
        "adam": args.adam_eta_max,
    }[optimizer]
    eta_lo = args.sign_eta_min if optimizer == "signsgd_hb" else args.eta_min
    eta = log_grid(eta_lo, eta_hi, args.eta_points, device, dtype)
    beta1 = beta_grid(args.beta1_points, args.beta_gap_min, device, dtype)
    alpha = alpha_grid(args.alpha_values, device, dtype)
    if optimizer != "adam":
        eta_m, b1_m, alpha_m = torch.meshgrid(eta, beta1, alpha, indexing="ij")
        return eta_m.reshape(-1), b1_m.reshape(-1), None, alpha_m.reshape(-1)
    beta2 = beta_grid(args.beta2_points, args.beta_gap_min, device, dtype)
    eta_m, b1_m, b2_m, alpha_m = torch.meshgrid(eta, beta1, beta2, alpha, indexing="ij")
    return eta_m.reshape(-1), b1_m.reshape(-1), b2_m.reshape(-1), alpha_m.reshape(-1)


def sign_no_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def linear_hb_risk(
    optimizer: str,
    eta: torch.Tensor,
    beta1: torch.Tensor,
    alpha: torch.Tensor,
    batch: float,
    steps: int,
    noise_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h_vals = torch.tensor([0.2, 1.0, 5.0], device=device, dtype=dtype)
    c_vals = noise_scale * torch.tensor([10.0, 1.0, 0.1], device=device, dtype=dtype)
    w0_vals = torch.tensor([0.3, 1.0, 3.0], device=device, dtype=dtype)
    eta = eta[:, None]
    beta1 = beta1[:, None]
    rho = (1.0 - alpha)[:, None]
    total = torch.zeros((eta.shape[0],), device=device, dtype=dtype)

    for h, c, w0 in zip(h_vals, c_vals, w0_vals, strict=True):
        if optimizer == "sgd_hb":
            a = h
            q = c / batch
        elif optimizer == "newton_hb":
            a = torch.ones((), device=device, dtype=dtype)
            q = c / (batch * h * h)
        else:
            raise ValueError(optimizer)
        mu_w = torch.full((eta.shape[0], 1), w0, device=device, dtype=dtype)
        mu_m = torch.zeros_like(mu_w)
        ww = mu_w.square()
        mm = torch.zeros_like(mu_w)
        wm = torch.zeros_like(mu_w)
        one_minus = 1.0 - beta1
        for _ in range(steps):
            mu_m_next = beta1 * mu_m + one_minus * a * mu_w
            mm_next = (
                beta1.square() * mm
                + one_minus.square() * (a.square() * ww + q)
                + 2.0 * beta1 * one_minus * a * wm
            )
            wm_pre_update = beta1 * wm + one_minus * a * ww
            mu_w_next = rho * mu_w - eta * mu_m_next
            ww_next = rho.square() * ww - 2.0 * rho * eta * wm_pre_update + eta.square() * mm_next
            wm_next = rho * wm_pre_update - eta * mm_next
            mu_w, mu_m, ww, mm, wm = mu_w_next, mu_m_next, ww_next, mm_next, wm_next
        total += 0.5 * h * ww[:, 0]
    return total


def grid_index(x: torch.Tensor, lo: float, hi: float, cells: int) -> torch.Tensor:
    pos = (x - lo) * ((cells - 1) / (hi - lo))
    return torch.clamp(torch.round(pos), 0, cells - 1).to(torch.long)


def finite_volume_coord_risk(
    optimizer: str,
    eta: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor | None,
    alpha: torch.Tensor,
    h: float,
    c: float,
    w0: float,
    batch: float,
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cnum = eta.numel()
    z_nodes = torch.tensor([-math.sqrt(3.0), 0.0, math.sqrt(3.0)], device=device, dtype=dtype)
    z_wts = torch.tensor([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], device=device, dtype=dtype)
    sigma = math.sqrt(c / batch)
    eta_v = eta[:, None]
    beta1_v = beta1[:, None]
    rho_v = (1.0 - alpha)[:, None]
    h_t = torch.tensor(h, device=device, dtype=dtype)
    sigma_t = torch.tensor(sigma, device=device, dtype=dtype)

    if optimizer == "signsgd_hb":
        nw, nm = args.sign_w_cells, args.sign_m_cells
        w_grid = torch.linspace(-args.w_bound, args.w_bound, nw, device=device, dtype=dtype)
        m_grid = torch.linspace(-args.m_bound, args.m_bound, nm, device=device, dtype=dtype)
        ww, mm = torch.meshgrid(w_grid, m_grid, indexing="ij")
        w_flat, m_flat = ww.reshape(-1), mm.reshape(-1)
        ncells = w_flat.numel()
        mass = torch.zeros((cnum, ncells), device=device, dtype=dtype)
        w0_idx = int(grid_index(torch.tensor(w0, device=device, dtype=dtype), -args.w_bound, args.w_bound, nw))
        m0_idx = int(grid_index(torch.tensor(0.0, device=device, dtype=dtype), -args.m_bound, args.m_bound, nm))
        mass[:, w0_idx * nm + m0_idx] = 1.0
        for _ in range(steps):
            next_mass = torch.zeros_like(mass)
            for z, wt in zip(z_nodes, z_wts, strict=True):
                g = h_t * w_flat + sigma_t * z
                m_next = beta1_v * m_flat[None, :] + (1.0 - beta1_v) * g[None, :]
                w_next = rho_v * w_flat[None, :] - eta_v * sign_no_zero(m_next)
                wi = grid_index(w_next, -args.w_bound, args.w_bound, nw)
                mi = grid_index(m_next, -args.m_bound, args.m_bound, nm)
                idx = wi * nm + mi
                next_mass.scatter_add_(1, idx, mass * wt)
            mass = next_mass / torch.clamp(next_mass.sum(dim=1, keepdim=True), min=1.0e-30)
        ew2 = (mass * w_flat.square()[None, :]).sum(dim=1)
        return 0.5 * h_t * ew2

    if optimizer == "adam":
        assert beta2 is not None
        nw, nm, nv = args.adam_w_cells, args.adam_m_cells, args.adam_v_cells
        w_grid = torch.linspace(-args.w_bound, args.w_bound, nw, device=device, dtype=dtype)
        m_grid = torch.linspace(-args.m_bound, args.m_bound, nm, device=device, dtype=dtype)
        v_grid = torch.linspace(0.0, args.v_bound, nv, device=device, dtype=dtype)
        ww, mm, vv = torch.meshgrid(w_grid, m_grid, v_grid, indexing="ij")
        w_flat, m_flat, v_flat = ww.reshape(-1), mm.reshape(-1), vv.reshape(-1)
        ncells = w_flat.numel()
        mass = torch.zeros((cnum, ncells), device=device, dtype=dtype)
        w0_idx = int(grid_index(torch.tensor(w0, device=device, dtype=dtype), -args.w_bound, args.w_bound, nw))
        m0_idx = int(grid_index(torch.tensor(0.0, device=device, dtype=dtype), -args.m_bound, args.m_bound, nm))
        v0_idx = 0
        mass[:, (w0_idx * nm + m0_idx) * nv + v0_idx] = 1.0
        beta2_v = beta2[:, None]
        beta1_pow = torch.ones((cnum, 1), device=device, dtype=dtype)
        beta2_pow = torch.ones((cnum, 1), device=device, dtype=dtype)
        for _ in range(steps):
            next_mass = torch.zeros_like(mass)
            beta1_pow = beta1_pow * beta1_v
            beta2_pow = beta2_pow * beta2_v
            for z, wt in zip(z_nodes, z_wts, strict=True):
                g = h_t * w_flat + sigma_t * z
                m_next = beta1_v * m_flat[None, :] + (1.0 - beta1_v) * g[None, :]
                v_next = beta2_v * v_flat[None, :] + (1.0 - beta2_v) * g.square()[None, :]
                if args.no_adam_bias_correction:
                    m_hat, v_hat = m_next, v_next
                else:
                    m_hat = m_next / torch.clamp(1.0 - beta1_pow, min=args.eps)
                    v_hat = v_next / torch.clamp(1.0 - beta2_pow, min=args.eps)
                w_next = rho_v * w_flat[None, :] - eta_v * m_hat / (torch.sqrt(torch.clamp(v_hat, min=0.0)) + args.eps)
                wi = grid_index(w_next, -args.w_bound, args.w_bound, nw)
                mi = grid_index(m_next, -args.m_bound, args.m_bound, nm)
                vi = grid_index(v_next, 0.0, args.v_bound, nv)
                idx = (wi * nm + mi) * nv + vi
                next_mass.scatter_add_(1, idx, mass * wt)
            mass = next_mass / torch.clamp(next_mass.sum(dim=1, keepdim=True), min=1.0e-30)
        ew2 = (mass * w_flat.square()[None, :]).sum(dim=1)
        return 0.5 * h_t * ew2

    raise ValueError(optimizer)


def finite_volume_risk(
    optimizer: str,
    eta: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor | None,
    alpha: torch.Tensor,
    batch: float,
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    total = torch.zeros((eta.numel(),), device=device, dtype=dtype)
    for h, c, w0 in [
        (0.2, 10.0 * args.noise_scale, 0.3),
        (1.0, 1.0 * args.noise_scale, 1.0),
        (5.0, 0.1 * args.noise_scale, 3.0),
    ]:
        total += finite_volume_coord_risk(
            optimizer, eta, beta1, beta2, alpha, h, c, w0, batch, steps, args, device, dtype
        )
    return total


def evaluate_optimizer(
    optimizer: str,
    batch: float,
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int | str]:
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
    for start in range(0, eta_all.numel(), args.chunk_configs):
        end = min(start + args.chunk_configs, eta_all.numel())
        eta = eta_all[start:end]
        beta1 = beta1_all[start:end]
        beta2 = beta2_all[start:end] if beta2_all is not None else None
        alpha = alpha_all[start:end]
        if optimizer in ("sgd_hb", "newton_hb"):
            risks = linear_hb_risk(optimizer, eta, beta1, alpha, batch, steps, args.noise_scale, device, dtype)
        else:
            risks = finite_volume_risk(optimizer, eta, beta1, beta2, alpha, batch, steps, args, device, dtype)
        finite = torch.isfinite(risks)
        if not bool(finite.any()):
            continue
        masked_risks = torch.where(finite, risks, torch.full_like(risks, float("inf")))
        idx = int(torch.argmin(masked_risks).item())
        risk = float(risks[idx].detach().cpu())
        if risk < float(best["risk"]):
            best["risk"] = risk
            best["eta"] = float(eta[idx].detach().cpu())
            best["beta1"] = float(beta1[idx].detach().cpu())
            best["beta2"] = float(beta2[idx].detach().cpu()) if beta2 is not None else float("nan")
            best["alpha"] = float(alpha[idx].detach().cpu())
    return best


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["optimizer", "batch", "steps", "eta", "beta1", "beta2", "alpha", "risk", "num_configs"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("bad shard index")
    device = choose_device(args.device)
    dtype = torch_dtype(args.dtype)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for steps in make_steps(args):
        batch = args.T / float(steps)
        for optimizer in args.optimizers:
            row = evaluate_optimizer(optimizer, batch, steps, args, device, dtype)
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = args.outdir / f"best_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.csv"
    write_csv(out, rows)
    summary = {
        "out": str(out),
        "rows": len(rows),
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.outdir / f"summary_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
