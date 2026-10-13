#!/usr/bin/env python3
"""Finite-budget learning-rate tuning for Newton, SignSGD, and RMSProp on the noisy quadratic model.

The Newton curve uses the exact closed-form finite-S risk.  The SignSGD curve
uses a deterministic finite-volume Markov recursion for the no-momentum,
lambda-zero SignSGD specialization.  The RMSProp curves use a deterministic
mean-Fisher recursion, tracking the second moment of the iterate and the mean
empirical Fisher accumulator.  The main RMSProp curve tunes both beta2 and eta
by a deterministic two-dimensional grid sweep.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/nqm_other_optimizer_tuning"))
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=4000)
    parser.add_argument("--sign-grid-size", type=int, default=401)
    parser.add_argument("--sign-eta-points", type=int, default=65)
    parser.add_argument("--rmsprop-eta-points", type=int, default=140)
    parser.add_argument("--rmsprop-beta2-points", type=int, default=41)
    parser.add_argument("--rmsprop-beta2", type=float, default=0.9)
    parser.add_argument("--rmsprop-base-batch", type=float, default=2.0)
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


def newton_risk_grid(
    eta: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> np.ndarray:
    batch = horizon / steps
    a0 = float(np.sum(h * w0 * w0))
    cn = float(np.sum(c / h))
    bias = (1.0 - eta) ** (2 * steps)
    variance = eta * eta * finite_sum_factor(eta, steps)
    return 0.5 * (a0 * bias + (cn / batch) * variance)


def newton_risk_scalar(
    eta: float,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> float:
    return float(newton_risk_grid(np.array([eta]), h, c, w0, horizon, steps)[0])


def golden_refine_newton(
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
    f1 = newton_risk_scalar(x1, h, c, w0, horizon, steps)
    f2 = newton_risk_scalar(x2, h, c, w0, horizon, steps)
    for _ in range(90):
        if f1 > f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = newton_risk_scalar(x2, h, c, w0, horizon, steps)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = newton_risk_scalar(x1, h, c, w0, horizon, steps)
    eta = 0.5 * (lo + hi)
    return eta, newton_risk_scalar(eta, h, c, w0, horizon, steps)


def tune_newton(
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
    grid_size: int,
) -> tuple[float, float]:
    eta_grid = np.linspace(1.0e-12, 2.0 - 1.0e-12, grid_size)
    values = newton_risk_grid(eta_grid, h, c, w0, horizon, steps)
    idx = int(np.argmin(values))
    lo = eta_grid[max(0, idx - 3)]
    hi = eta_grid[min(grid_size - 1, idx + 3)]
    return golden_refine_newton(lo, hi, h, c, w0, horizon, steps)


def normal_cdf_approx(x: np.ndarray) -> np.ndarray:
    # Abramowitz-Stegun erf approximation, vectorized and accurate enough for
    # evaluating a deterministic finite-volume Markov recursion.
    z = x / math.sqrt(2.0)
    sign = np.where(z < 0.0, -1.0, 1.0)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    erf = sign * (
        1.0
        - (
            (
                (
                    (1.061405429 * t - 1.453152027) * t
                    + 1.421413741
                )
                * t
                - 0.284496736
            )
            * t
            + 0.254829592
        )
        * t
        * np.exp(-a * a)
    )
    return np.clip(0.5 * (1.0 + erf), 0.0, 1.0)


def deposit_linear(
    out: np.ndarray,
    grid_lo: float,
    grid_dx: float,
    targets: np.ndarray,
    masses: np.ndarray,
) -> None:
    n = out.shape[0]
    pos = (targets - grid_lo) / grid_dx
    pos = np.clip(pos, 0.0, n - 1.0)
    idx = np.floor(pos).astype(np.int64)
    frac = pos - idx
    np.add.at(out, idx, masses * (1.0 - frac))
    upper = np.minimum(idx + 1, n - 1)
    np.add.at(out, upper, masses * frac)


def signsgd_coordinate_moment(
    eta: float,
    h: float,
    c: float,
    w_initial: float,
    batch: float,
    steps: int,
    grid_size: int,
) -> float:
    sigma = math.sqrt(c / batch)
    bound = max(abs(w_initial) + eta * steps, 1.0) + eta + 0.75
    grid = np.linspace(-bound, bound, grid_size)
    dx = float(grid[1] - grid[0])
    dist = np.zeros(grid_size, dtype=np.float64)
    deposit_linear(dist, float(grid[0]), dx, np.array([w_initial]), np.array([1.0]))
    prob_plus = normal_cdf_approx(h * grid / sigma)

    for _ in range(steps):
        plus = grid - eta
        minus = grid + eta
        next_dist = np.zeros_like(dist)
        deposit_linear(next_dist, float(grid[0]), dx, plus, dist * prob_plus)
        deposit_linear(next_dist, float(grid[0]), dx, minus, dist * (1.0 - prob_plus))
        total = float(next_dist.sum())
        if total <= 0.0:
            return float("inf")
        dist = next_dist / total
    return float(np.dot(dist, grid * grid))


def signsgd_risk(
    eta: float,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
    grid_size: int,
) -> float:
    batch = horizon / steps
    total = 0.0
    for hi, ci, wi in zip(h, c, w0, strict=True):
        moment = signsgd_coordinate_moment(eta, float(hi), float(ci), float(wi), batch, steps, grid_size)
        total += float(hi) * moment
    return 0.5 * total


def tune_signsgd(
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
    grid_size: int,
    eta_points: int,
) -> tuple[float, float]:
    eta_grid = np.geomspace(5.0e-4, 1.15, eta_points)
    values = np.array(
        [
            signsgd_risk(float(eta), h, c, w0, horizon, steps, grid_size)
            for eta in eta_grid
        ]
    )
    idx = int(np.argmin(values))
    return float(eta_grid[idx]), float(values[idx])


def rmsprop_mean_fisher_risk(
    eta: float,
    beta2: float,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> tuple[float, float]:
    batch = horizon / steps
    second = w0 * w0
    fisher = h * h * second + c / batch
    eps = 1.0e-10
    for _ in range(steps):
        fisher = beta2 * fisher + (1.0 - beta2) * (h * h * second + c / batch)
        denom = np.sqrt(np.maximum(fisher, eps))
        contraction = 1.0 - eta * h / denom
        noise_var = eta * eta * (c / batch) / np.maximum(fisher, eps)
        second = contraction * contraction * second + noise_var
        if not np.all(np.isfinite(second)) or float(np.max(second)) > 1.0e12:
            return float("inf"), float("inf")
    risk = 0.5 * float(np.sum(h * second))
    mean_fisher = float(np.mean(fisher))
    return risk, mean_fisher


def tune_rmsprop(
    beta2: float,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
    eta_points: int,
) -> tuple[float, float, float]:
    eta_grid = np.geomspace(1.0e-4, 2.0, eta_points)
    risks = []
    fishers = []
    for eta in eta_grid:
        risk, fisher = rmsprop_mean_fisher_risk(float(eta), beta2, h, c, w0, horizon, steps)
        risks.append(risk)
        fishers.append(fisher)
    risks_arr = np.array(risks)
    idx = int(np.argmin(risks_arr))
    return float(eta_grid[idx]), float(risks_arr[idx]), float(fishers[idx])


def rmsprop_beta2_grid(num_points: int) -> np.ndarray:
    grid = 1.0 - np.geomspace(1.0, 1.0e-4, num_points)
    anchors = np.array([0.0, 0.5, 0.9, 0.99, 0.999], dtype=float)
    return np.unique(np.clip(np.concatenate([grid, anchors]), 0.0, 0.9999))


def rmsprop_mean_fisher_grid(
    eta_grid: np.ndarray,
    beta2_grid: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch = horizon / steps
    eta_flat = np.tile(eta_grid, beta2_grid.size)
    beta2_flat = np.repeat(beta2_grid, eta_grid.size)
    second = np.broadcast_to(w0 * w0, (eta_flat.size, w0.size)).copy()
    fisher = h * h * second + c / batch
    eta_col = eta_flat[:, None]
    beta_col = beta2_flat[:, None]
    eps = 1.0e-10
    invalid = np.zeros(eta_flat.size, dtype=bool)

    for _ in range(steps):
        fisher = beta_col * fisher + (1.0 - beta_col) * (h * h * second + c / batch)
        denom_sq = np.maximum(fisher, eps)
        contraction = 1.0 - eta_col * h / np.sqrt(denom_sq)
        noise_var = eta_col * eta_col * (c / batch) / denom_sq
        second = contraction * contraction * second + noise_var
        bad = (~np.all(np.isfinite(second), axis=1)) | (np.max(second, axis=1) > 1.0e12)
        if np.any(bad):
            invalid |= bad
            second[bad, :] = 1.0e12
            fisher[bad, :] = 1.0e12

    risks = 0.5 * np.sum(h * second, axis=1)
    risks[invalid] = np.inf
    mean_fisher = np.mean(fisher, axis=1)
    mean_fisher[invalid] = np.inf
    return risks.reshape(beta2_grid.size, eta_grid.size), mean_fisher.reshape(beta2_grid.size, eta_grid.size)


def tune_rmsprop_2d(
    h: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    horizon: int,
    steps: int,
    eta_points: int,
    beta2_points: int,
) -> tuple[float, float, float, float]:
    eta_grid = np.geomspace(1.0e-4, 2.0, eta_points)
    beta2_grid = rmsprop_beta2_grid(beta2_points)
    risks, fishers = rmsprop_mean_fisher_grid(eta_grid, beta2_grid, h, c, w0, horizon, steps)
    flat_idx = int(np.nanargmin(risks))
    beta_idx, eta_idx = np.unravel_index(flat_idx, risks.shape)
    return (
        float(eta_grid[eta_idx]),
        float(beta2_grid[beta_idx]),
        float(risks[beta_idx, eta_idx]),
        float(fishers[beta_idx, eta_idx]),
    )


def make_records(
    name: str,
    hyper_label: str,
    batches: np.ndarray,
    steps: np.ndarray,
    hyper: np.ndarray,
    risks: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    order = np.argsort(batches)
    batches = batches[order]
    steps = steps[order]
    hyper = hyper[order]
    risks = risks[order]
    log_batches = np.log(batches)
    log_hyper = np.log(np.maximum(hyper, 1.0e-300))
    slope = np.gradient(log_hyper, log_batches)
    anchor = float(hyper[np.argmin(batches)])
    linear = anchor * batches / float(np.min(batches))
    out: dict[str, object] = {
        "name": name,
        "hyper_label": hyper_label,
        "batches": batches,
        "steps": steps,
        "hyper": hyper,
        "risk": risks,
        "slope": slope,
        "linear": linear,
        "slope_min": float(np.min(slope)),
        "slope_max": float(np.max(slope)),
        "anchor_batch": float(np.min(batches)),
        "anchor_hyper": anchor,
    }
    if extra:
        for key, value in extra.items():
            out[key] = value[order]
    return out


def record_slug(record: dict[str, object]) -> str:
    raw = str(record["name"]).lower()
    return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    h = np.array([0.2, 1.0, 5.0], dtype=float)
    c = args.noise_scale * np.array([10.0, 1.0, 0.1], dtype=float)
    w0 = np.array([0.3, 1.0, 3.0], dtype=float)

    steps = np.unique(np.round(np.geomspace(4, args.T // 2, args.num_points)).astype(int))
    batches = args.T / steps.astype(float)

    newton_rows = []
    sign_rows = []
    rms_constant_rows = []
    rms_scaled_rows = []
    rms_2d_rows = []
    for step in steps:
        step_int = int(step)
        batch = args.T / step_int

        eta_newton, risk_newton = tune_newton(h, c, w0, args.T, step_int, args.grid_size)
        newton_rows.append((batch, step_int, eta_newton, risk_newton))

        eta_sign, risk_sign = tune_signsgd(
            h,
            c,
            w0,
            args.T,
            step_int,
            args.sign_grid_size,
            args.sign_eta_points,
        )
        sign_rows.append((batch, step_int, eta_sign, risk_sign))

        eta_rms, risk_rms, fisher_rms = tune_rmsprop(
            args.rmsprop_beta2,
            h,
            c,
            w0,
            args.T,
            step_int,
            args.rmsprop_eta_points,
        )
        rms_constant_rows.append((batch, step_int, eta_rms, risk_rms, fisher_rms, args.rmsprop_beta2))

        scaled_beta2 = args.rmsprop_beta2 ** (batch / args.rmsprop_base_batch)
        eta_rms_scaled, risk_rms_scaled, fisher_rms_scaled = tune_rmsprop(
            scaled_beta2,
            h,
            c,
            w0,
            args.T,
            step_int,
            args.rmsprop_eta_points,
        )
        rms_scaled_rows.append((batch, step_int, eta_rms_scaled, risk_rms_scaled, fisher_rms_scaled, scaled_beta2))

        eta_rms_2d, beta2_rms_2d, risk_rms_2d, fisher_rms_2d = tune_rmsprop_2d(
            h,
            c,
            w0,
            args.T,
            step_int,
            args.rmsprop_eta_points,
            args.rmsprop_beta2_points,
        )
        rms_2d_rows.append((batch, step_int, eta_rms_2d, risk_rms_2d, fisher_rms_2d, beta2_rms_2d))

    newton_data = np.array(newton_rows, dtype=float)
    sign_data = np.array(sign_rows, dtype=float)
    rms_constant_data = np.array(rms_constant_rows, dtype=float)
    rms_scaled_data = np.array(rms_scaled_rows, dtype=float)
    rms_2d_data = np.array(rms_2d_rows, dtype=float)

    newton_record = make_records("Newton", "eta*(B)", newton_data[:, 0], newton_data[:, 1], newton_data[:, 2], newton_data[:, 3])
    sign_record = make_records("SignSGD", "eta*(B)", sign_data[:, 0], sign_data[:, 1], sign_data[:, 2], sign_data[:, 3])
    rms_constant_record = make_records(
        "RMSProp constant beta2",
        "eta*(B)",
        rms_constant_data[:, 0],
        rms_constant_data[:, 1],
        rms_constant_data[:, 2],
        rms_constant_data[:, 3],
        {"mean_fisher": rms_constant_data[:, 4], "beta2": rms_constant_data[:, 5]},
    )
    rms_scaled_record = make_records(
        "RMSProp fixed beta2 root",
        "eta*(B)",
        rms_scaled_data[:, 0],
        rms_scaled_data[:, 1],
        rms_scaled_data[:, 2],
        rms_scaled_data[:, 3],
        {"mean_fisher": rms_scaled_data[:, 4], "beta2": rms_scaled_data[:, 5]},
    )
    rms_2d_record = make_records(
        "RMSProp 2D tuned",
        "eta*(B)",
        rms_2d_data[:, 0],
        rms_2d_data[:, 1],
        rms_2d_data[:, 2],
        rms_2d_data[:, 3],
        {"mean_fisher": rms_2d_data[:, 4], "beta2": rms_2d_data[:, 5]},
    )
    records = [newton_record, sign_record, rms_constant_record, rms_scaled_record, rms_2d_record]

    summary = {
        "T": args.T,
        "h": h.tolist(),
        "c": c.tolist(),
        "noise_scale": args.noise_scale,
        "w0": w0.tolist(),
        "rmsprop_beta2": args.rmsprop_beta2,
        "rmsprop_base_batch": args.rmsprop_base_batch,
        "rmsprop_scaled_beta2_rule": "beta2(B)=beta2_base**(B/base_batch)",
        "rmsprop_2d_sweep": {
            "eta_grid_min": 1.0e-4,
            "eta_grid_max": 2.0,
            "eta_points": args.rmsprop_eta_points,
            "beta2_grid_min": float(np.min(rmsprop_beta2_grid(args.rmsprop_beta2_points))),
            "beta2_grid_max": float(np.max(rmsprop_beta2_grid(args.rmsprop_beta2_points))),
            "beta2_points": int(rmsprop_beta2_grid(args.rmsprop_beta2_points).size),
        },
        "signsgd": {
            "alpha": 1.0,
            "lambda": 0.0,
            "eta_grid_min": 5.0e-4,
            "eta_grid_max": 1.15,
            "finite_volume_grid_size": args.sign_grid_size,
        },
        "records": {
            record["name"]: {
                "anchor_batch": record["anchor_batch"],
                "anchor_hyper": record["anchor_hyper"],
                "slope_min": record["slope_min"],
                "slope_max": record["slope_max"],
                "hyper_min": float(np.min(record["hyper"])),
                "hyper_max": float(np.max(record["hyper"])),
                "risk_min": float(np.min(record["risk"])),
                "risk_max": float(np.max(record["risk"])),
            }
            for record in records
        },
    }
    (args.outdir / "nqm_other_optimizer_tuning.json").write_text(json.dumps(summary, indent=2) + "\n")

    for record in records:
        cols = [record["batches"], record["steps"], record["hyper"], record["risk"], record["linear"], record["slope"]]
        header = "batch,steps,tuned_step,risk,linear_from_B2,local_log_slope"
        if "mean_fisher" in record:
            cols.append(record["mean_fisher"])
            header += ",mean_fisher"
        if "beta2" in record:
            cols.append(record["beta2"])
            header += ",beta2"
        np.savetxt(
            args.outdir / f"{record_slug(record)}_tuning.csv",
            np.column_stack(cols),
            delimiter=",",
            header=header,
            comments="",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
