"""Validated, explicit experiment contracts; no implicit batch/LR transport."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    schema: str = "directional_cbs_v1"
    development: bool = False
    synthetic: bool = False
    seed: int = 1
    base_batch_tokens: int = 131072
    sequence_length: int = 1024
    microbatch_sequences: int = 2
    parent_steps: int = 64
    schedule_steps: int = 13000
    anchor_interval: int = 32
    branch_steps: int = 64
    top_rank: int = 16
    lanczos_iterations: int = 128
    ritz_relative_tolerance: float = 0.01
    hessian_tokens: int = 131072
    validation_tokens: int = 10485760
    remainder_periods: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    lr_powers: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    trajectory: bool = False
    curve_eval_interval: int = 0
    curve_eval_tokens: int = 1048576
    curve_holdout_shards: list[str] = field(default_factory=list)
    save_branch_checkpoint: bool = True
    data_root: str = "data/fineweb10B"
    # recipe CSV (format in README.md); the base run uses its W/muon row at base_batch_tokens
    recipes: str = ""
    vocabulary_size: int = 50304
    layers: int = 12
    width: int = 768
    head_dimension: int = 128
    mlp_multiplier: int = 4
    logit_softcap: float = 15.0

    def __post_init__(self):
        if self.schema != "directional_cbs_v1":
            raise ValueError("unsupported configuration schema")
        if not isinstance(self.recipes, str) or not self.recipes:
            raise ValueError("recipes: give the path of a recipe CSV with the 128K MuonW row")
        for name in ("base_batch_tokens", "sequence_length", "microbatch_sequences",
                     "parent_steps", "schedule_steps", "anchor_interval", "branch_steps",
                     "top_rank", "lanczos_iterations", "hessian_tokens", "validation_tokens",
                     "vocabulary_size", "layers", "width", "head_dimension", "mlp_multiplier"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("base_batch_tokens", "hessian_tokens", "validation_tokens", "curve_eval_tokens"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if getattr(self, name) % self.sequence_length:
                raise ValueError(f"{name} must be divisible by sequence_length")
        if self.hessian_tokens > self.base_batch_tokens:
            raise ValueError("Hessian sample must fit the preceding base batch")
        if self.parent_steps % self.anchor_interval:
            raise ValueError("parent horizon must end at an anchor")
        if self.parent_steps > self.schedule_steps:
            raise ValueError("parent exceeds token-clock schedule")
        if not self.trajectory and self.parent_steps + self.branch_steps > self.schedule_steps:
            raise ValueError("parent + branch exceeds token-clock schedule")
        if (not isinstance(self.curve_eval_interval, int) or self.curve_eval_interval < 0
                or isinstance(self.curve_eval_interval, bool)):
            raise ValueError("curve_eval_interval must be a nonnegative integer")
        if len(set(self.curve_holdout_shards)) != len(self.curve_holdout_shards):
            raise ValueError("duplicate curve holdout shards")
        if any(Path(name).name != name or not name.startswith("fineweb_train_")
               or not name.endswith(".bin") for name in self.curve_holdout_shards):
            raise ValueError("curve holdout must name complete training shards")
        if self.curve_eval_interval and not (self.synthetic or self.curve_holdout_shards):
            raise ValueError("curve evaluation requires an excluded training-shard holdout")
        if self.trajectory and not self.curve_eval_interval:
            raise ValueError("trajectory runs require holdout curves")
        if self.lanczos_iterations <= self.top_rank:
            raise ValueError("Lanczos iterations must exceed top_rank")
        if not self.remainder_periods or self.remainder_periods[0] != 1:
            raise ValueError("every sweep must include a k=1 control")
        if self.remainder_periods != sorted(set(self.remainder_periods)):
            raise ValueError("periods must be unique and sorted")
        if any(not isinstance(k, int) or k < 1 or self.branch_steps % k
               for k in self.remainder_periods):
            raise ValueError("each positive integer period must divide the branch horizon")
        if not self.lr_powers or any(not math.isfinite(p) for p in self.lr_powers):
            raise ValueError("LR powers must be finite and nonempty")
        if self.width % self.head_dimension or self.head_dimension % 4:
            raise ValueError("invalid attention dimensions")
        if not 0 < self.ritz_relative_tolerance < 1:
            raise ValueError("ritz_relative_tolerance must lie in (0, 1)")
        if not self.development:
            expected = (131072, 1024, 16, 50304, 12, 768, 128, 4)
            actual = (self.base_batch_tokens, self.sequence_length, self.top_rank,
                      self.vocabulary_size, self.layers, self.width,
                      self.head_dimension, self.mlp_multiplier)
            if actual != expected or self.synthetic:
                raise ValueError("noncanonical size/data requires development=true")

    def model_kwargs(self) -> dict:
        names = ("vocabulary_size", "layers", "width", "head_dimension",
                 "mlp_multiplier", "logit_softcap")
        return {name: getattr(self, name) for name in names}

    def to_dict(self) -> dict:
        return asdict(self)

    def branch_horizon(self, parent_step: int) -> int:
        """All cells at an anchor use the same fully flushed token horizon."""
        horizon = self.branch_steps
        if self.trajectory:
            horizon = min(horizon, self.schedule_steps - parent_step)
            multiple = math.lcm(*self.remainder_periods)
            horizon = horizon // multiple * multiple
        if horizon <= 0 or parent_step + horizon > self.schedule_steps:
            raise ValueError("no complete branch window remains")
        return horizon


def load_config(path: str | Path) -> Config:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return Config(**value)


def cases(config: Config) -> list[dict]:
    result = [{"id": "vanilla", "vanilla": True, "project_input": False,
               "project_output": False, "remainder_period": 1, "lr_power": 0.0}]
    for pi in (False, True):
        for po in (False, True):
            for k in config.remainder_periods:
                for power in ([0.0] if k == 1 else config.lr_powers):
                    result.append({
                        "id": f"gin{int(pi)}-uout{int(po)}-k{k}-p{power:g}",
                        "vanilla": False, "project_input": pi, "project_output": po,
                        "remainder_period": k, "lr_power": power,
                    })
    return result
