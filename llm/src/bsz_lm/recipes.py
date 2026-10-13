"""Recipe CSV: one row of absolute hyperparameters per (family, optimizer, batch) case."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from bsz_lm.canonical import digest_value

Family = Literal["W", "H"]
OptimizerName = Literal["muon", "klsoap", "shampoo", "adam", "lion"]

FAMILIES: tuple[Family, ...] = ("W", "H")
OPTIMIZERS: tuple[OptimizerName, ...] = ("muon", "klsoap", "shampoo", "adam", "lion")
BATCHES = (131072, 524288, 1048576, 2097152)
# batches a recipe may run at: the four crossover batches plus the 256K transfer target
RUNNABLE_BATCHES = BATCHES + (262144,)
# matrix-optimizer columns each optimizer must set; the other ones of
# (matrix_mu, matrix_beta1, matrix_beta2, shampoo_beta) must be empty
MATRIX_COEFFICIENTS: dict[str, frozenset[str]] = {
    "muon": frozenset({"matrix_mu"}),
    "klsoap": frozenset({"matrix_beta1", "matrix_beta2", "shampoo_beta"}),
    "shampoo": frozenset({"matrix_beta1", "matrix_beta2"}),
    "adam": frozenset({"matrix_beta1", "matrix_beta2"}),
    "lion": frozenset({"matrix_beta1", "matrix_beta2"}),
}
OPTIONAL_COEFFICIENTS = ("matrix_mu", "matrix_beta1", "matrix_beta2", "shampoo_beta")


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


@dataclass(frozen=True)
class Recipe:
    family: Family
    optimizer: OptimizerName
    label: str
    batch: str
    batch_tokens: int
    train_steps: int
    matrix_algorithm: str
    matrix_lr: float
    matrix_mu: float | None
    matrix_beta1: float | None
    matrix_beta2: float | None
    shampoo_beta: float | None
    matrix_wd_peak: float
    aux_lr_emb: float
    aux_lr_proj: float
    aux_lr_misc: float
    aux_beta1: float
    aux_beta2: float
    aux_cooldown_frac: float
    aux_wd_peak: float

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Recipe:
        family = row["family"]
        optimizer = row["optimizer"]
        if family not in FAMILIES:
            raise ValueError(f"unknown family: {family}")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer: {optimizer}")
        value = cls(
            family=family,
            optimizer=optimizer,
            label=row["label"],
            batch=row["batch"],
            batch_tokens=int(row["batch_tokens"]),
            train_steps=int(row["train_steps"]),
            matrix_algorithm=row["matrix_algorithm"],
            matrix_lr=float(row["matrix_lr"]),
            matrix_mu=_optional_float(row["matrix_mu"]),
            matrix_beta1=_optional_float(row["matrix_beta1"]),
            matrix_beta2=_optional_float(row["matrix_beta2"]),
            shampoo_beta=_optional_float(row["shampoo_beta"]),
            matrix_wd_peak=float(row["matrix_wd_peak"]),
            aux_lr_emb=float(row["aux_lr_emb"]),
            aux_lr_proj=float(row["aux_lr_proj"]),
            aux_lr_misc=float(row["aux_lr_misc"]),
            aux_beta1=float(row["aux_beta1"]),
            aux_beta2=float(row["aux_beta2"]),
            aux_cooldown_frac=float(row["aux_cooldown_frac"]),
            aux_wd_peak=float(row["aux_wd_peak"]),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer: {self.optimizer}")
        if self.batch_tokens not in RUNNABLE_BATCHES:
            raise ValueError(f"unexpected Track-3 batch: {self.batch_tokens}")
        expected_steps = math.ceil(1_703_936_000 / self.batch_tokens)
        if self.train_steps != expected_steps:
            raise ValueError(
                f"{self.label}/{self.batch}: train_steps must be {expected_steps} "
                f"at {self.batch_tokens} tokens per batch"
            )
        if self.family == "H" and self.matrix_wd_peak != 0:
            raise ValueError("HyperBall recipes cannot use matrix weight decay")
        if self.family == "H" and self.aux_wd_peak != 0:
            raise ValueError("HyperBall recipes cannot use auxiliary weight decay")
        required = MATRIX_COEFFICIENTS[self.optimizer]
        for name in OPTIONAL_COEFFICIENTS:
            given = getattr(self, name) is not None
            if name in required and not given:
                raise ValueError(f"{self.optimizer} requires {name}")
            if name not in required and given:
                raise ValueError(f"{self.optimizer} does not use {name}; leave it empty")
        for name, number in asdict(self).items():
            if isinstance(number, float) and not math.isfinite(number):
                raise ValueError(f"{name} must be finite")

    @property
    def recipe_id(self) -> str:
        return digest_value(self.scientific_dict())

    def scientific_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "recipe_id": self.recipe_id}


RECIPE_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(Recipe))


def load_recipes(path: str | Path) -> list[Recipe]:
    """Read and validate a recipe CSV (format in README.md); any number of rows."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing = [name for name in RECIPE_COLUMNS if name not in header]
        unknown = [name for name in header if name not in RECIPE_COLUMNS]
        if missing or unknown or len(set(header)) != len(header):
            raise ValueError(
                f"recipe CSV columns must be exactly {', '.join(RECIPE_COLUMNS)}; "
                f"missing={missing}, unknown={unknown}"
            )
        rows = list(reader)
    recipes = []
    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"recipe CSV line {line} does not have {len(RECIPE_COLUMNS)} fields")
        try:
            recipes.append(Recipe.from_row(row))
        except ValueError as error:
            raise ValueError(f"recipe CSV line {line}: {error}") from error
    if not recipes:
        raise ValueError(f"recipe CSV has no rows: {path}")
    return recipes
