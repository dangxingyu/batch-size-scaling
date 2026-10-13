#!/usr/bin/env python3
"""High-precision validation for fixed NQM momentum hyperparameters.

This script is intentionally not a hyperparameter search.  It takes a CSV of
candidate/winning hyperparameters and re-evaluates each row with independent
Monte Carlo samples.  Linear optimizers (SGD-M and Newton-M) are evaluated by
exact second-moment recurrences in float64; nonlinear optimizers are evaluated
by high-chain Monte Carlo and reported with a standard error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


LINEAR_OPTIMIZERS = {"sgd_hb", "newton_hb"}
NONLINEAR_OPTIMIZERS = {"signsgd_hb", "adam"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=131072)
    parser.add_argument("--chain-chunk", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--no-adam-bias-correction", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def choose_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float:
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def exact_linear_risk(
    optimizer: str,
    eta: float,
    beta1: float,
    alpha: float,
    batch: float,
    steps: int,
    noise_scale: float,
    device: torch.device,
) -> float:
    dtype = torch.float64
    h_vals = torch.tensor([0.2, 1.0, 5.0], device=device, dtype=dtype)
    c_vals = noise_scale * torch.tensor([10.0, 1.0, 0.1], device=device, dtype=dtype)
    w0_vals = torch.tensor([0.3, 1.0, 3.0], device=device, dtype=dtype)
    eta_t = torch.tensor([[eta]], device=device, dtype=dtype)
    beta_t = torch.tensor([[beta1]], device=device, dtype=dtype)
    rho_t = torch.tensor([[1.0 - alpha]], device=device, dtype=dtype)
    total = torch.zeros((), device=device, dtype=dtype)

    for h, c, w0 in zip(h_vals, c_vals, w0_vals, strict=True):
        if optimizer == "sgd_hb":
            a = h
            q = c / batch
        elif optimizer == "newton_hb":
            a = torch.ones((), device=device, dtype=dtype)
            q = c / (batch * h * h)
        else:
            raise ValueError(optimizer)

        mu_w = torch.full((1, 1), w0, device=device, dtype=dtype)
        mu_m = torch.zeros_like(mu_w)
        ww = mu_w.square()
        mm = torch.zeros_like(mu_w)
        wm = torch.zeros_like(mu_w)
        one_minus = 1.0 - beta_t
        for _ in range(steps):
            mu_m_next = beta_t * mu_m + one_minus * a * mu_w
            mm_next = (
                beta_t.square() * mm
                + one_minus.square() * (a.square() * ww + q)
                + 2.0 * beta_t * one_minus * a * wm
            )
            wm_pre_update = beta_t * wm + one_minus * a * ww
            mu_w_next = rho_t * mu_w - eta_t * mu_m_next
            ww_next = rho_t.square() * ww - 2.0 * rho_t * eta_t * wm_pre_update + eta_t.square() * mm_next
            wm_next = rho_t * wm_pre_update - eta_t * mm_next
            mu_w, mu_m, ww, mm, wm = mu_w_next, mu_m_next, ww_next, mm_next, wm_next
        total = total + 0.5 * h * ww[0, 0]
    return float(total.detach().cpu())


def sign_no_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def nonlinear_samples(
    optimizer: str,
    eta: float,
    beta1: float,
    beta2: float,
    alpha: float,
    batch: float,
    steps: int,
    chains: int,
    chain_chunk: int,
    repeats: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    h = torch.tensor([0.2, 1.0, 5.0], device=device, dtype=dtype).view(1, 3)
    c = args.noise_scale * torch.tensor([10.0, 1.0, 0.1], device=device, dtype=dtype).view(1, 3)
    w0 = torch.tensor([0.3, 1.0, 3.0], device=device, dtype=dtype).view(1, 3)
    eta_t = torch.tensor(eta, device=device, dtype=dtype)
    beta1_t = torch.tensor(beta1, device=device, dtype=dtype)
    beta2_t = torch.tensor(beta2, device=device, dtype=dtype) if optimizer == "adam" else None
    rho_t = torch.tensor(1.0 - alpha, device=device, dtype=dtype)
    sigma = torch.sqrt(c / batch)

    total_n = 0
    total_sum = 0.0
    total_sumsq = 0.0

    for repeat in range(repeats):
        remaining = chains
        chunk_id = 0
        while remaining > 0:
            n = min(chain_chunk, remaining)
            remaining -= n
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 1000003 * repeat + 9176 * chunk_id)
            chunk_id += 1

            w = w0.expand(n, 3).clone()
            m = torch.zeros_like(w)
            v = torch.zeros_like(w) if optimizer == "adam" else None
            beta1_power = torch.ones((), device=device, dtype=dtype)
            beta2_power = torch.ones((), device=device, dtype=dtype)
            for _ in range(steps):
                noise = torch.randn((n, 3), device=device, dtype=dtype, generator=generator)
                g = h * w + sigma * noise
                if optimizer == "signsgd_hb":
                    m = beta1_t * m + (1.0 - beta1_t) * g
                    w = rho_t * w - eta_t * sign_no_zero(m)
                elif optimizer == "adam":
                    assert v is not None and beta2_t is not None
                    m = beta1_t * m + (1.0 - beta1_t) * g
                    v = beta2_t * v + (1.0 - beta2_t) * g.square()
                    if args.no_adam_bias_correction:
                        m_hat, v_hat = m, v
                    else:
                        beta1_power = beta1_power * beta1_t
                        beta2_power = beta2_power * beta2_t
                        m_hat = m / torch.clamp(1.0 - beta1_power, min=args.eps)
                        v_hat = v / torch.clamp(1.0 - beta2_power, min=args.eps)
                    w = rho_t * w - eta_t * m_hat / (torch.sqrt(v_hat) + args.eps)
                else:
                    raise ValueError(optimizer)

            losses = 0.5 * (h * w.square()).sum(dim=1)
            losses = losses.detach().to(torch.float64)
            total_n += n
            total_sum += float(losses.sum().cpu())
            total_sumsq += float(losses.square().sum().cpu())

    mean = total_sum / total_n
    variance = max(total_sumsq / total_n - mean * mean, 0.0)
    se = math.sqrt(variance / total_n)
    return mean, se, total_n


def validate_row(
    row: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int | str]:
    optimizer = row["optimizer"]
    batch = parse_float(row["batch"])
    steps = int(round(parse_float(row["steps"])))
    eta = parse_float(row["eta"])
    beta1 = parse_float(row["beta1"])
    beta2 = parse_float(row["beta2"])
    alpha = parse_float(row.get("alpha", "0.0"))

    out: dict[str, float | int | str] = {
        "optimizer": optimizer,
        "batch": batch,
        "steps": steps,
        "eta": eta,
        "beta1": beta1,
        "beta2": beta2,
        "alpha": alpha,
        "search_risk": parse_float(row["risk"]) if "risk" in row else float("nan"),
        "risk_mean": float("nan"),
        "risk_se": 0.0,
        "samples": 0,
        "method": "",
    }
    if optimizer in LINEAR_OPTIMIZERS:
        out["risk_mean"] = exact_linear_risk(
            optimizer, eta, beta1, alpha, batch, steps, args.noise_scale, device
        )
        out["risk_se"] = 0.0
        out["samples"] = 0
        out["method"] = "exact_second_moment_float64"
        return out
    if optimizer in NONLINEAR_OPTIMIZERS:
        mean, se, samples = nonlinear_samples(
            optimizer,
            eta,
            beta1,
            beta2,
            alpha,
            batch,
            steps,
            args.chains,
            args.chain_chunk,
            args.repeats,
            args.seed + 104729 * steps,
            args,
            device,
            dtype,
        )
        out["risk_mean"] = mean
        out["risk_se"] = se
        out["samples"] = samples
        out["method"] = f"mc_{args.dtype}"
        return out
    raise ValueError(optimizer)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "optimizer",
        "batch",
        "steps",
        "eta",
        "beta1",
        "beta2",
        "alpha",
        "search_risk",
        "risk_mean",
        "risk_se",
        "samples",
        "method",
    ]
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
    rows = read_rows(args.input_csv)
    shard_rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row in shard_rows:
        result = validate_row(row, args, device, dtype)
        out_rows.append(result)
        print(json.dumps(result), flush=True)

    out = args.outdir / f"validation_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.csv"
    write_csv(out, out_rows)
    summary = {
        "out": str(out),
        "input": str(args.input_csv),
        "rows": len(out_rows),
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.outdir / f"summary_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
