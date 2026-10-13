"""Strict protocol loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bsz_lm.canonical import digest_value

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "name",
    "seed",
    "model",
    "data",
    "training",
    "search",
    "runtime",
}


@dataclass(frozen=True)
class Protocol:
    value: dict[str, Any]
    protocol_id: str

    @property
    def token_horizon(self) -> int:
        return int(self.value["training"]["token_horizon"])

    @property
    def batches(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.value["training"]["batches"])

    def train_steps(self, batch_tokens: int) -> int:
        return math.ceil(self.token_horizon / batch_tokens)


def _require_keys(name: str, value: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")


def load_protocol(path: str | Path) -> Protocol:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol must be a mapping")
    unknown = sorted(set(value) - EXPECTED_TOP_LEVEL)
    missing = sorted(EXPECTED_TOP_LEVEL - set(value))
    if unknown or missing:
        raise ValueError(f"protocol top-level mismatch: missing={missing}, unknown={unknown}")
    if value["schema_version"] != 2:
        raise ValueError("unsupported protocol schema")
    _require_keys(
        "training",
        value["training"],
        {
            "reference_batch_tokens",
            "token_horizon",
            "batches",
            "gpus_per_case",
            "schedule_profiles",
        },
    )
    batches = tuple(int(item) for item in value["training"]["batches"])
    if batches != tuple(sorted(set(batches))) or any(batch <= 0 for batch in batches):
        raise ValueError("training batches must be unique, positive, and sorted")
    horizon = int(value["training"]["token_horizon"])
    reference = int(value["training"]["reference_batch_tokens"])
    if horizon != reference * 13000:
        raise ValueError("Track-3 token horizon must equal 13,000 reference-batch steps")
    model = value["model"]
    expected_model = {
        "vocabulary_size": 50304,
        "layers": 12,
        "width": 768,
        "heads": 6,
        "head_dimension": 128,
        "sequence_length": 1024,
        "mlp_multiplier": 4,
        "logit_softcap": 15.0,
    }
    if model != expected_model:
        raise ValueError("model does not match the Track-3 architecture")
    training = value["training"]
    if int(training["gpus_per_case"]) != 8 or training.get("gradient_reduction") != "sum":
        raise ValueError("Track-3 requires eight ranks and SUM gradient reduction")
    if training.get("precision") != "bfloat16":
        raise ValueError("Track-3 precision must be bfloat16")
    profiles = training["schedule_profiles"]
    expected_profiles = {
        f"{family}/{optimizer}"
        for family in ("W", "H")
        for optimizer in ("muon", "klsoap", "shampoo", "adam", "lion")
    }
    if not isinstance(profiles, dict) or set(profiles) != expected_profiles:
        raise ValueError("training schedule profiles must cover all ten family/optimizer pairs")
    profile_fields = {
        "matrix_cooldown",
        "matrix_warmup_steps",
        "matrix_warmup_mode",
        "auxiliary_warmup_steps",
        "auxiliary_warmup_mode",
    }
    for key, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) != profile_fields:
            raise ValueError(f"invalid schedule profile fields for {key}")
        if profile["matrix_cooldown"] not in {"auxiliary", "full"}:
            raise ValueError(f"invalid matrix cooldown policy for {key}")
        for role in ("matrix", "auxiliary"):
            steps = int(profile[f"{role}_warmup_steps"])
            mode = profile[f"{role}_warmup_mode"]
            if steps < 0 or mode not in {"none", "multiplicative", "exclusive"}:
                raise ValueError(f"invalid {role} warmup for {key}")
            if (steps == 0) != (mode == "none"):
                raise ValueError(f"{role} warmup steps/mode disagree for {key}")
    data = value["data"]
    if int(data["validation_tokens"]) != 10_485_760:
        raise ValueError("Track-3 validation uses exactly 10,485,760 tokens")
    shampoo_source = value["runtime"].get("shampoo_source", {})
    _require_keys(
        "runtime.shampoo_source",
        shampoo_source,
        {"repository", "commit", "subdirectory", "package_tree"},
    )
    return Protocol(value=value, protocol_id=digest_value(value))
