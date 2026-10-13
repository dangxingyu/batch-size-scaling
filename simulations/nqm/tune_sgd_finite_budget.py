#!/usr/bin/env python3
"""Exact finite-budget SGD learning-rate tuning on the noisy quadratic model.

The simulation uses the closed-form finite-S risk from the theory section,
with integer S and B = T / S.  It is deterministic: no Monte Carlo sampling is
needed for SGD on a diagonal noisy quadratic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/nqm_sgd_tuned_eta"))
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=90)
    parser.add_argument("--grid-size", type=int, default=5000)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    return parser.parse_args()


def finite_sum_factor(x: np.ndarray, steps: int) -> np.ndarray:
    r = (1.0 - x) ** 2
    denom = 1.0 - r
    out = np.empty_like(x)
    near = np.abs(denom) < 1.0e-10
    out[near] = steps
    out[~near] = (1.0 - r[~near] ** steps) / denom[~near]
    return out


def risk_grid(
    eta: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> np.ndarray:
    batch = horizon / steps
    total = np.zeros_like(eta, dtype=float)
    for hi, ci, wi in zip(h, c, w0, strict=True):
        x = eta * hi
        bias = (1.0 - x) ** (2 * steps)
        variance = x * x * finite_sum_factor(x, steps)
        total += hi * wi * wi * bias + (ci / (batch * hi)) * variance
    return 0.5 * total


def risk_scalar(
    eta: float,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> float:
    return float(risk_grid(np.array([eta]), h, c, w0, horizon, steps)[0])


def golden_refine(
    lo: float,
    hi: float,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> tuple[float, float]:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = risk_scalar(x1, h, c, w0, horizon, steps)
    f2 = risk_scalar(x2, h, c, w0, horizon, steps)
    for _ in range(90):
        if f1 > f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = risk_scalar(x2, h, c, w0, horizon, steps)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = risk_scalar(x1, h, c, w0, horizon, steps)
    eta = 0.5 * (lo + hi)
    return eta, risk_scalar(eta, h, c, w0, horizon, steps)


def tune_eta(
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
    grid_size: int,
) -> tuple[float, float]:
    eta_max = 2.0 / float(np.max(h))
    eta_grid = np.linspace(1.0e-12, eta_max - 1.0e-12, grid_size)
    values = risk_grid(eta_grid, h, c, w0, horizon, steps)
    idx = int(np.argmin(values))
    lo = eta_grid[max(0, idx - 3)]
    hi = eta_grid[min(grid_size - 1, idx + 3)]
    return golden_refine(lo, hi, h, c, w0, horizon, steps)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # A three-coordinate example with separated curvatures.  The exact optimum
    # bends sharply as the update budget S = T / B becomes small.
    h = np.array([0.2, 1.0, 5.0], dtype=float)
    c = args.noise_scale * np.array([10.0, 1.0, 0.1], dtype=float)
    w0 = np.array([0.3, 1.0, 3.0], dtype=float)

    steps = np.unique(np.round(np.geomspace(4, args.T // 2, args.num_points)).astype(int))
    records = []
    for step in steps:
        eta, value = tune_eta(h, c, w0, args.T, int(step), args.grid_size)
        records.append((args.T / int(step), int(step), eta, value))
    data = np.array(records, dtype=float)

    # Sort by increasing batch.
    order = np.argsort(data[:, 0])
    batches = data[order, 0]
    steps = data[order, 1]
    etas = data[order, 2]
    risks = data[order, 3]

    log_batches = np.log(batches)
    log_etas = np.log(etas)
    slope = np.gradient(log_etas, log_batches)
    anchor_batch = float(np.min(batches))
    anchor_eta = float(etas[np.argmin(batches)])
    theory_power1 = anchor_eta * batches / anchor_batch

    h_star = float(np.min(h[w0 != 0.0]))
    active = h == h_star
    a_star = float(np.sum(h[active] ** 2 * w0[active] ** 2))
    c_total = float(np.sum(c))
    tau = math.log(4.0 * args.T * a_star / c_total) / (2.0 * h_star)
    asymptotic = batches * tau / args.T
    asymptotic_mask = steps >= 32

    summary = {
        "T": args.T,
        "h": h.tolist(),
        "c": c.tolist(),
        "noise_scale": args.noise_scale,
        "w0": w0.tolist(),
        "theory_power": 1.0,
        "theory_anchor_batch": anchor_batch,
        "theory_anchor_eta": anchor_eta,
        "log_rmse_theory_power1_from_anchor": float(np.sqrt(np.mean((log_etas - np.log(theory_power1)) ** 2))),
        "slope_min": float(np.min(slope)),
        "slope_max": float(np.max(slope)),
        "eta_min": float(np.min(etas)),
        "eta_max": float(np.max(etas)),
        "batch_min": float(np.min(batches)),
        "batch_max": float(np.max(batches)),
    }
    (args.outdir / "sgd_finite_budget_tuning.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savetxt(
        args.outdir / "sgd_finite_budget_tuning.csv",
        np.column_stack([batches, steps, etas, risks, theory_power1, asymptotic, slope]),
        delimiter=",",
        header="batch,steps,eta_star,risk_star,theory_power1_from_B2,large_T_asymptotic,local_log_slope",
        comments="",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
