#!/usr/bin/env python3
"""Local refinement sweep for momentum NQM hyperparameters.

The broad momentum sweep is useful for finding the right region, but its beta
grid is intentionally sparse away from one.  This script reads incumbent
hyperparameters and runs a denser per-row search around each incumbent.  Linear
optimizers are evaluated by exact second-moment recurrences; SignSGD-M and Adam
use the same vectorized Monte Carlo simulator as the broad sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import sweep_nqm_momentum_finite_volume_torch as exact_sweep  # noqa: E402
import sweep_nqm_momentum_torch as mc_sweep  # noqa: E402


OPTIMIZERS = ("sgd_hb", "newton_hb", "signsgd_hb", "adam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--chains", type=int, default=2048)
    parser.add_argument("--max-configs-per-chunk", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--no-adam-bias-correction", action="store_true")
    parser.add_argument("--divergence-threshold", type=float, default=1.0e12)
    parser.add_argument("--noise-scale", type=float, default=1.0)

    parser.add_argument("--eta-points", type=int, default=49)
    parser.add_argument("--adam-eta-points", type=int, default=41)
    parser.add_argument("--eta-radius", type=float, default=2.5)
    parser.add_argument("--eta-min", type=float, default=1.0e-5)
    parser.add_argument("--sgd-eta-max", type=float, default=1.0)
    parser.add_argument("--newton-eta-max", type=float, default=2.0)
    parser.add_argument("--sign-eta-max", type=float, default=1.2)
    parser.add_argument("--adam-eta-max", type=float, default=2.0)

    parser.add_argument("--beta1-points", type=int, default=61)
    parser.add_argument("--adam-beta1-points", type=int, default=41)
    parser.add_argument("--beta1-radius", type=float, default=0.25)
    parser.add_argument("--beta-gap-min", type=float, default=1.0e-7)
    parser.add_argument("--beta-gap-radius", type=float, default=100.0)

    parser.add_argument("--beta2-points", type=int, default=61)
    parser.add_argument("--beta2-gap-min", type=float, default=1.0e-7)
    parser.add_argument("--beta2-gap-radius", type=float, default=1000.0)
    return parser.parse_args()


def parse_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    if value.lower() in {"nan", "none"}:
        return default
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def unique_sorted(values: Iterable[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    raw = sorted({float(v) for v in values if math.isfinite(float(v))})
    vals: list[float] = []
    for value in raw:
        if not vals or abs(value - vals[-1]) > 1.0e-12 * max(1.0, abs(value), abs(vals[-1])):
            vals.append(value)
    return torch.tensor(vals, device=device, dtype=dtype)


def log_values(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [math.sqrt(lo * hi)]
    return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1)) for i in range(n)]


def eta_grid(
    optimizer: str,
    center: float,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hi = {
        "sgd_hb": args.sgd_eta_max,
        "newton_hb": args.newton_eta_max,
        "signsgd_hb": args.sign_eta_max,
        "adam": args.adam_eta_max,
    }[optimizer]
    n = args.adam_eta_points if optimizer == "adam" else args.eta_points
    lo = max(args.eta_min, center / args.eta_radius)
    hi = min(hi, center * args.eta_radius)
    return unique_sorted([center, *log_values(lo, hi, n)], device, dtype)


def beta_grid(
    center: float,
    points: int,
    linear_radius: float,
    gap_min: float,
    gap_radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    center = min(max(center, 0.0), 1.0 - gap_min)
    vals = [center, 0.0, 0.5, 0.9, 0.99, 0.999]

    lo = max(0.0, center - linear_radius)
    hi = min(1.0 - gap_min, center + linear_radius)
    if points > 1:
        vals.extend(lo + (hi - lo) * i / (points - 1) for i in range(points))
    else:
        vals.append(center)

    gap = max(gap_min, 1.0 - center)
    gap_lo = max(gap_min, gap / gap_radius)
    gap_hi = min(1.0, gap * gap_radius)
    vals.extend(1.0 - g for g in log_values(gap_lo, gap_hi, points))
    vals = [min(max(v, 0.0), 1.0 - gap_min) for v in vals]
    return unique_sorted(vals, device, dtype)


def make_configs(
    row: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    optimizer = row["optimizer"]
    eta = eta_grid(optimizer, parse_float(row["eta"]), args, device, dtype)
    beta1_points = args.adam_beta1_points if optimizer == "adam" else args.beta1_points
    beta1 = beta_grid(
        parse_float(row["beta1"], 0.0),
        beta1_points,
        args.beta1_radius,
        args.beta_gap_min,
        args.beta_gap_radius,
        device,
        dtype,
    )
    alpha_value = parse_float(row.get("alpha"), 0.0)
    alpha = torch.tensor([min(max(alpha_value, 0.0), 1.0)], device=device, dtype=dtype)
    if optimizer != "adam":
        eta_m, beta1_m, alpha_m = torch.meshgrid(eta, beta1, alpha, indexing="ij")
        return eta_m.reshape(-1), beta1_m.reshape(-1), None, alpha_m.reshape(-1)

    beta2 = beta_grid(
        parse_float(row["beta2"], 0.0),
        args.beta2_points,
        linear_radius=0.25,
        gap_min=args.beta2_gap_min,
        gap_radius=args.beta2_gap_radius,
        device=device,
        dtype=dtype,
    )
    eta_m, beta1_m, beta2_m, alpha_m = torch.meshgrid(eta, beta1, beta2, alpha, indexing="ij")
    return eta_m.reshape(-1), beta1_m.reshape(-1), beta2_m.reshape(-1), alpha_m.reshape(-1)


def top_records(
    records: list[dict[str, float | int | str]],
    candidates: list[tuple[float, int]],
    eta: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor | None,
    alpha: torch.Tensor,
    row: dict[str, str],
    top_k: int,
    num_configs: int,
) -> None:
    optimizer = row["optimizer"]
    batch = parse_float(row["batch"])
    steps = int(round(parse_float(row["steps"])))
    for rank, (risk, idx) in enumerate(sorted(candidates, key=lambda item: item[0])[:top_k]):
        records.append(
            {
                "optimizer": optimizer,
                "batch": batch,
                "steps": steps,
                "eta": float(eta[idx].detach().cpu()),
                "beta1": float(beta1[idx].detach().cpu()),
                "beta2": float(beta2[idx].detach().cpu()) if beta2 is not None else float("nan"),
                "alpha": float(alpha[idx].detach().cpu()),
                "risk": float(risk),
                "rank": rank,
                "num_configs": num_configs,
                "source_eta": parse_float(row["eta"]),
                "source_beta1": parse_float(row["beta1"], 0.0),
                "source_beta2": parse_float(row.get("beta2"), float("nan")),
                "source_risk": parse_float(row.get("risk"), float("nan")),
            }
        )


def evaluate_row(
    row: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float | int | str]]:
    optimizer = row["optimizer"]
    batch = parse_float(row["batch"])
    steps = int(round(parse_float(row["steps"])))
    eta_all, beta1_all, beta2_all, alpha_all = make_configs(row, args, device, dtype)
    h = torch.tensor([0.2, 1.0, 5.0], device=device, dtype=dtype)
    c = args.noise_scale * torch.tensor([10.0, 1.0, 0.1], device=device, dtype=dtype)
    w0 = torch.tensor([0.3, 1.0, 3.0], device=device, dtype=dtype)

    candidates: list[tuple[float, int]] = []
    chunk = args.max_configs_per_chunk
    optimizer_offset = {name: 10000 * idx for idx, name in enumerate(mc_sweep.OPTIMIZERS)}[optimizer]
    seed = args.seed + 104729 * steps + optimizer_offset
    for start in range(0, eta_all.numel(), chunk):
        end = min(start + chunk, eta_all.numel())
        eta = eta_all[start:end]
        beta1 = beta1_all[start:end]
        beta2 = beta2_all[start:end] if beta2_all is not None else None
        alpha = alpha_all[start:end]
        if optimizer in ("sgd_hb", "newton_hb"):
            risks = exact_sweep.linear_hb_risk(
                optimizer, eta, beta1, alpha, batch, steps, args.noise_scale, device, torch.float64
            )
        else:
            risks = mc_sweep.risk_for_chunk(
                optimizer,
                eta,
                beta1,
                beta2,
                alpha,
                h,
                c,
                w0,
                batch,
                steps,
                args,
                seed,
                device,
                dtype,
            )
        finite = torch.isfinite(risks)
        if bool(finite.any()):
            masked = torch.where(finite, risks, torch.full_like(risks, float("inf")))
            k = min(args.top_k, masked.numel())
            values, indices = torch.topk(masked, k=k, largest=False)
            candidates.extend((float(v.detach().cpu()), start + int(i.detach().cpu())) for v, i in zip(values, indices))
            candidates = sorted(candidates, key=lambda item: item[0])[: args.top_k]

    records: list[dict[str, float | int | str]] = []
    top_records(records, candidates, eta_all, beta1_all, beta2_all, alpha_all, row, args.top_k, int(eta_all.numel()))
    return records


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
        "risk",
        "rank",
        "num_configs",
        "source_eta",
        "source_beta1",
        "source_beta2",
        "source_risk",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("bad shard index")
    device = mc_sweep.choose_device(args.device)
    dtype = mc_sweep.torch_dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    rows = [row for row in read_rows(args.input_csv) if row["optimizer"] in set(args.optimizers)]
    shard_rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, float | int | str]] = []
    for row in shard_rows:
        refined = evaluate_row(row, args, device, dtype)
        out_rows.extend(refined)
        for item in refined:
            print(json.dumps(item), flush=True)

    out = args.outdir / f"refined_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.csv"
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
