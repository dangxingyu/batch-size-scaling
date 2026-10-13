"""CUDA-only parent / anchor / branch stages, usable with one GPU or torchrun.

All branches restore a common parent; anchors never modify the parent. The
Hessian is a joint hidden-matrix FP32 mathematical-model Hessian, including
cross-matrix products, measured on the preceding training batch. It excludes
auxiliary parameters and decoupled WD. Training retains Track-3 BF16 kernels.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from bsz_lm.branchout.config import PROJECT, Config
from bsz_lm.branchout.data import Batches
from bsz_lm.branchout.optimization import (
    CoreConfig,
    MatrixCore,
    SplitAccumulator,
    apply_matrix_delta,
)
from bsz_lm.branchout.spectral import Layout, TwoSpaces, top_ritz
from bsz_lm.canonical import digest_file, digest_tree, digest_value
from bsz_lm.model import GPT, initialize_model
from bsz_lm.optimizers import set_scheduled_hparams, setup_optimizers
from bsz_lm.protocol import load_protocol
from bsz_lm.recipes import load_recipes

CHECKPOINT_SCHEMA = "directional_cbs_checkpoint_v3"

# algorithm name -> (matrix wrapper family, matrix optimizer); branching is MuonW only
ALGORITHMS = {"muon": ("W", "muon")}


def runtime() -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: no CPU numerical fallback")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(minutes=120))
    return device, (dist.get_rank() if dist.is_initialized() else 0), (
        dist.get_world_size() if dist.is_initialized() else 1
    )


def barrier():
    if dist.is_initialized():
        dist.barrier()


def source_id() -> str:
    return digest_tree(PROJECT / "src" / "bsz_lm",
                       ignored_names={"__pycache__", ".pytest_cache", ".ruff_cache"})


def json_write(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def tree_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_cpu(v) for v in value)
    return copy.deepcopy(value)


def atomic_save(path: Path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


class Experiment:
    def __init__(self, config: Config, algorithm: str):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {sorted(ALGORITHMS)}")
        self.cfg, self.algorithm = config, algorithm
        self.family, self.optimizer = ALGORITHMS[algorithm]
        self.device, self.rank, self.world = runtime()
        if not config.development and self.world != 8:
            raise ValueError("non-development runs require one eight-GPU node")
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        matches = [r for r in load_recipes(config.recipes) if r.family == self.family
                   and r.optimizer == self.optimizer and r.batch_tokens == 131072]
        if len(matches) != 1:
            raise ValueError(f"{config.recipes} must have exactly one {self.family}/{self.optimizer} "
                             f"row at batch_tokens 131072, found {len(matches)}")
        self.recipe = matches[0]
        self.profile = load_protocol(PROJECT / "configs" / "protocol.yaml").value[
            "training"]["schedule_profiles"][f"{self.family}/{self.optimizer}"]
        self.model = GPT(**config.model_kwargs()).to(self.device)
        initialize_model(self.model, self.family)
        self.parameters = {n: p for n, p in self.model.named_parameters()
                           if n.startswith("blocks.") and p.ndim == 2}
        self.layout = Layout.from_parameters(self.parameters)
        self.auxiliary, self.schedule_holder = setup_optimizers(self.model, self.recipe)
        self.core = MatrixCore(self.layout, CoreConfig(
            self.optimizer, momentum=0.95 if self.recipe.matrix_mu is None else self.recipe.matrix_mu,
        ), self.device)
        self.data = Batches(config, self.device, self.rank, self.world)
        self.source = source_id()
        self.step = 0

    def apply_delta(self, delta: torch.Tensor, learning_rate: float, weight_decay: float) -> None:
        """Apply a hidden-matrix delta: decoupled weight decay, then add."""
        apply_matrix_delta(self.parameters, self.layout, delta,
                           learning_rate=learning_rate, weight_decay=weight_decay)

    def schedule(self):
        set_scheduled_hparams([self.auxiliary, self.schedule_holder], step=self.step,
                             train_steps=self.cfg.schedule_steps,
                             auxiliary_cooldown=self.recipe.aux_cooldown_frac,
                             schedule_profile=self.profile)
        group = self.schedule_holder.param_groups[0]
        return float(group["lr"]), float(group["weight_decay"])

    def gradients(self):
        inputs, targets = self.data.batch(self.step)
        self.model.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.device)
        micro = self.cfg.microbatch_sequences
        for start in range(0, len(inputs), micro):
            loss = self.model(inputs[start:start + micro], targets[start:start + micro])
            loss.backward()
            total += loss.detach()
        for parameter in self.model.parameters():
            if parameter.grad is None:
                raise RuntimeError("missing training gradient")
            if self.world > 1:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        if self.world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        gradient = self.layout.pack({n: p.grad for n, p in self.parameters.items()})
        return gradient, float(total / self.cfg.base_batch_tokens)

    @torch.no_grad()
    def evaluate(self, *, split: str = "val", tokens: int | None = None) -> float:
        count = self.cfg.validation_tokens if tokens is None else tokens
        previous_mode = self.model.training
        before_rng = rng_state()
        try:
            self.model.eval()
            total = torch.zeros((), device=self.device)
            inputs, targets = self.data.batch(0, split=split, tokens=count)
            micro = self.cfg.microbatch_sequences
            for start in range(0, len(inputs), micro):
                total += self.model(inputs[start:start + micro], targets[start:start + micro])
            if self.world > 1:
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            result = float(total / count)
        finally:
            self.model.train(previous_mode)
            restore_rng(before_rng)
        if not np.isfinite(result):
            raise FloatingPointError("nonfinite endpoint validation loss")
        return result

    def advance(self, split: SplitAccumulator | None = None) -> dict:
        torch.cuda.synchronize()
        start = time.perf_counter()
        lr, wd = self.schedule()
        gradient, training_loss = self.gradients()
        torch.cuda.synchronize()
        after_gradient = time.perf_counter()
        if split is None:
            delta = -lr * self.core.direction(gradient)
            metrics = {}
        else:
            delta, metrics = split.advance(gradient, lr)
        self.apply_delta(delta, lr, wd)
        self.auxiliary.step()
        self.model.zero_grad(set_to_none=True)
        self.step += 1
        torch.cuda.synchronize()
        end = time.perf_counter()
        return {"step": self.step, "tokens": self.step * self.cfg.base_batch_tokens,
                "training_loss": training_loss, "matrix_lr": lr, "matrix_wd": wd,
                "gradient_seconds": after_gradient - start,
                "optimizer_seconds": end - after_gradient, "step_seconds": end - start,
                "max_memory_bytes": torch.cuda.max_memory_allocated(), "bands": metrics}

    def checkpoint(self, path: Path, *, split: SplitAccumulator | None = None):
        local_rng = rng_state()
        all_rng = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(all_rng, local_rng)
        else:
            all_rng[0] = local_rng
        if self.rank == 0:
            state = {"schema": CHECKPOINT_SCHEMA, "source_id": self.source,
                     "config": self.cfg.to_dict(), "algorithm": self.algorithm,
                     "family": self.family,
                     "recipe": self.recipe.to_dict(), "step": self.step,
                     "schedule_profile": copy.deepcopy(self.profile),
                     "world_size": self.world, "rng_by_rank": all_rng,
                     "model": tree_cpu(self.model.state_dict()),
                     "core": tree_cpu(self.core.state_dict()),
                     "auxiliary": tree_cpu(self.auxiliary.state_dict()),
                     "layout": self.layout.metadata(), "data": self.data.metadata,
                     "data_next_base_step": self.step, "split": None}
            if split is not None:
                state["split"] = tree_cpu({
                    "cores": {b: c.state_dict() for b, c in split.cores.items()},
                    "accumulators": split.accumulators, "base_steps": split.base_steps,
                    "lr_accumulator": split.lr_accumulator, "periods": split.periods,
                    "project_input": split.project_input, "project_output": split.project_output,
                    "lr_power": split.lr_power,
                })
            atomic_save(path, state)
        barrier()

    def restore(self, path: Path) -> dict:
        # Only load locally generated, trusted experiment checkpoints.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for key, expected in (("schema", CHECKPOINT_SCHEMA), ("source_id", self.source),
                              ("config", self.cfg.to_dict()),
                              ("algorithm", self.algorithm), ("family", self.family),
                              ("world_size", self.world),
                              ("layout", self.layout.metadata()), ("data", self.data.metadata),
                              ("recipe", self.recipe.to_dict()),
                              ("schedule_profile", self.profile)):
            if saved.get(key) != expected:
                raise ValueError(f"checkpoint mismatch: {key}")
        if saved.get("split") is not None:
            raise ValueError("branch parents must be vanilla trajectory checkpoints")
        self.model.load_state_dict(saved["model"])
        self.core.load_state_dict(saved["core"])
        self.auxiliary.load_state_dict(saved["auxiliary"])
        self.step = int(saved["step"])
        if self.step != saved["data_next_base_step"]:
            raise ValueError("checkpoint data cursor mismatch")
        restore_rng(saved["rng_by_rank"][self.rank])
        return saved


def prepare_output(exp: Experiment, root: Path, stage: str, **metadata):
    if exp.rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        json_write(root / "started.json", {
            "schema": "directional_cbs_stage_v1", "stage": stage,
            "config": exp.cfg.to_dict(), "source_id": exp.source,
            "algorithm": exp.algorithm, "family": exp.family, "world_size": exp.world,
            "torch_version": torch.__version__,
            "data": exp.data.metadata,
            **metadata,
        })
    barrier()


def event(exp: Experiment, root: Path, value: dict):
    if exp.rank == 0:
        with (root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(value, sort_keys=True), flush=True)


def curve_event(exp: Experiment, root: Path):
    if not exp.cfg.curve_eval_interval:
        return
    loss = exp.evaluate(split="curve", tokens=exp.cfg.curve_eval_tokens)
    if exp.rank == 0:
        row = {"step": exp.step, "tokens": exp.step * exp.cfg.base_batch_tokens,
               "progress_percent": 100 * exp.step / exp.cfg.schedule_steps,
               "holdout_loss": loss, "holdout_tokens": exp.cfg.curve_eval_tokens,
               "metric": "fixed excluded-train holdout cross entropy, nats/token",
               "holdout_id": digest_value(exp.data.metadata.get("curve", exp.data.metadata)),
               "recorded_at": time.time()}
        with (root / "curve.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def run_parent(config: Config, algorithm: str, root: Path) -> list[Path]:
    exp = Experiment(config, algorithm)
    prepare_output(exp, root, "parent")
    curve_event(exp, root)
    checkpoints = []
    for _ in range(config.parent_steps):
        event(exp, root, exp.advance())
        if config.curve_eval_interval and (exp.step % config.curve_eval_interval == 0
                                          or exp.step % config.anchor_interval == 0):
            curve_event(exp, root)
        if exp.step % config.anchor_interval == 0:
            path = root / f"checkpoint-{exp.step:06d}.pt"
            exp.checkpoint(path)
            checkpoints.append(path)
    endpoint = exp.evaluate()
    if exp.rank == 0:
        json_write(root / "result.json", {"status": "complete", "validation_loss": endpoint,
                                         "steps": exp.step, "source_id": exp.source,
                                         "checkpoints": [str(p) for p in checkpoints]})
    barrier()
    return checkpoints


def run_anchor(config: Config, algorithm: str, checkpoint: Path, root: Path) -> Path:
    exp = Experiment(config, algorithm)
    exp.restore(checkpoint)
    if exp.step < 1:
        raise ValueError("Hessian anchors require a preceding training batch")
    checkpoint_hash = digest_file(checkpoint)
    prepare_output(exp, root, "anchor", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash)
    before_rng = rng_state()
    model = copy.deepcopy(exp.model).float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [dict(model.named_parameters())[name] for name in exp.layout.names]
    for parameter in selected:
        parameter.requires_grad_(True)
    inputs, targets = exp.data.batch(exp.step - 1, tokens=config.hessian_tokens)
    micro = config.microbatch_sequences
    calls = 0

    def hvp(vector):
        nonlocal calls
        calls += 1
        vector_parts = tuple(exp.layout.unpack(vector).values())
        total = torch.zeros_like(vector)
        for start in range(0, len(inputs), micro):
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                loss = model(inputs[start:start + micro], targets[start:start + micro])
                loss = loss / config.hessian_tokens
                first = torch.autograd.grad(loss, selected, create_graph=True)
                dot = sum((g * v).sum() for g, v in zip(first, vector_parts, strict=True))
                second = torch.autograd.grad(dot, selected, allow_unused=True)
                values = {name: (torch.zeros_like(p) if g is None else g)
                          for name, p, g in zip(exp.layout.names, selected, second, strict=True)}
                total += exp.layout.pack(values)
                del first, second, dot, loss, values
        if exp.world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return total

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    generator = torch.Generator(device=exp.device).manual_seed(config.seed + exp.step + 913)
    probes = [torch.randn(exp.layout.dimension, device=exp.device, generator=generator)
              for _ in range(2)]
    probes = [v / v.norm() for v in probes]
    products = [hvp(v) for v in probes]
    uv, vu = torch.dot(probes[0], products[1]), torch.dot(probes[1], products[0])
    symmetry = float((uv - vu).abs() / (
        products[0].norm() + products[1].norm()).clamp_min(1e-20))
    del probes, products
    basis, report = top_ritz(hvp, dimension=exp.layout.dimension, top_rank=config.top_rank,
                            iterations=config.lanczos_iterations, device=exp.device,
                            seed=config.seed + exp.step)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    accepted = symmetry <= 1e-4 and report["max_relative_residual"] <= config.ritz_relative_tolerance
    report.update({"schema": "directional_cbs_anchor_v1", "accepted": accepted,
                   "checkpoint_sha256": checkpoint_hash, "parent_step": exp.step,
                   "source_id": exp.source, "hvp_calls": calls,
                   "hvp_symmetry_error": symmetry, "seconds": elapsed,
                   "seconds_per_hvp_including_solver": elapsed / calls,
                   "max_memory_bytes": torch.cuda.max_memory_allocated(),
                   "hessian_tokens": config.hessian_tokens,
                   "hessian_batch_base_step": exp.step - 1,
                   "hessian_precision": "FP32 mathematical-model proxy at saved weights",
                   "layout": exp.layout.metadata(), "config_id": digest_value(config.to_dict())})
    restore_rng(before_rng)
    if exp.rank == 0:
        json_write(root / "result.json", report)
        if accepted:
            atomic_save(root / "basis.pt", {"basis": basis.cpu(), "metadata": report})
    barrier()
    if not accepted:
        raise RuntimeError(f"Hessian gate failed: symmetry={symmetry}, "
                           f"max Ritz residual={report['max_relative_residual']}")
    return root / "basis.pt"


def run_branch(config: Config, algorithm: str, checkpoint: Path, basis_path: Path,
               case: dict, root: Path) -> dict:
    exp = Experiment(config, algorithm)
    exp.restore(checkpoint)
    parent_step = exp.step
    horizon = config.branch_horizon(parent_step)
    if horizon % case["remainder_period"]:
        raise ValueError("branch horizon does not flush accumulation")
    checkpoint_hash = digest_file(checkpoint)
    payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    for key, value in (("checkpoint_sha256", checkpoint_hash), ("parent_step", parent_step),
                       ("source_id", exp.source), ("layout", exp.layout.metadata()),
                       ("config_id", digest_value(config.to_dict())), ("accepted", True)):
        if metadata[key] != value:
            raise ValueError(f"anchor identity mismatch: {key}")
    spaces = TwoSpaces(payload["basis"].to(exp.device))
    del payload
    split = None if case["vanilla"] else SplitAccumulator(
        exp.core, spaces, project_input=case["project_input"],
        project_output=case["project_output"], remainder_period=case["remainder_period"],
        lr_power=case["lr_power"],
    )
    prepare_output(exp, root, "branch", case=case, checkpoint_sha256=checkpoint_hash,
                   basis_sha256=digest_file(basis_path), parent_step=parent_step,
                   branch_steps=horizon)
    curve_event(exp, root)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for offset in range(horizon):
        record = exp.advance(split)
        timings.append(record["step_seconds"])
        event(exp, root, record)
        if config.curve_eval_interval and ((offset + 1) % config.curve_eval_interval == 0
                                          or offset + 1 == horizon):
            curve_event(exp, root)
    if split is not None:
        split.assert_flushed()
    endpoint = exp.evaluate()
    if config.save_branch_checkpoint:
        exp.checkpoint(root / "final.pt", split=split)
    result = {"schema": "directional_cbs_branch_v1", "status": "complete", "case": case,
              "algorithm": algorithm, "parent_step": parent_step,
              "parent_checkpoint_sha256": checkpoint_hash, "source_id": exp.source,
              "basis_sha256": digest_file(basis_path), "config_id": digest_value(config.to_dict()),
              "branch_steps": horizon,
              "additional_tokens": horizon * config.base_batch_tokens,
              "final_data_next_base_step": exp.step, "validation_loss": endpoint,
              "validation_tokens": config.validation_tokens,
              "median_step_seconds": float(np.median(timings)),
              "mean_step_seconds": float(np.mean(timings)),
              "max_memory_bytes": torch.cuda.max_memory_allocated(),
              "checkpoint": str(root / "final.pt") if config.save_branch_checkpoint else None}
    if exp.rank == 0:
        json_write(root / "result.json", result)
    barrier()
    return result
