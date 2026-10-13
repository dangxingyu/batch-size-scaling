"""Build immutable scientific case plans independently of cluster resources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from bsz_lm.canonical import digest_file, digest_tree, digest_value, write_json
from bsz_lm.data_manifest import load_data_manifest
from bsz_lm.protocol import Protocol, load_protocol
from bsz_lm.recipes import Recipe, load_recipes


@dataclass(frozen=True)
class CaseSelection:
    families: tuple[str, ...] = ()
    optimizers: tuple[str, ...] = ()
    batches: tuple[int, ...] = ()

    def accepts(self, recipe: Recipe) -> bool:
        return (
            (not self.families or recipe.family in self.families)
            and (not self.optimizers or recipe.optimizer in self.optimizers)
            and (not self.batches or recipe.batch_tokens in self.batches)
        )


def _case(
    protocol: Protocol,
    recipe: Recipe,
    *,
    stage: str,
    max_steps: int | None,
    data_id: str,
    source_id: str,
) -> dict[str, Any]:
    effective_steps = (
        recipe.train_steps if max_steps is None else min(recipe.train_steps, max_steps)
    )
    scientific = {
        "schema": "bsz_lm_case_v1",
        "stage": stage,
        "protocol_id": protocol.protocol_id,
        "data_id": data_id,
        "source_id": source_id,
        "recipe": recipe.scientific_dict(),
        "effective_train_steps": effective_steps,
        "seed": int(protocol.value["seed"]),
    }
    fingerprint = digest_value(scientific)
    return {
        **scientific,
        "case_id": (
            f"{stage}-{recipe.family.lower()}-{recipe.optimizer}-"
            f"b{recipe.batch_tokens}-{fingerprint[:12]}"
        ),
        "run_fingerprint": fingerprint,
    }


def build_plan(
    protocol_path: str | Path,
    recipes_path: str | Path,
    data_manifest_path: str | Path,
    *,
    source_root: str | Path | None = None,
    selection: CaseSelection | None = None,
    stage: str = "tuned",
    max_steps: int | None = None,
) -> dict[str, Any]:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    protocol = load_protocol(protocol_path)
    recipes = load_recipes(recipes_path)
    data_manifest = load_data_manifest(data_manifest_path)
    data_contract = {key: value for key, value in data_manifest.items() if key != "root"}
    package_root = Path(source_root) if source_root is not None else Path(__file__).parent
    source_id = digest_tree(
        package_root,
        ignored_names={"__pycache__", ".pytest_cache", ".ruff_cache"},
    )
    selected = selection or CaseSelection()
    cases = [
        _case(
            protocol,
            recipe,
            stage=stage,
            max_steps=max_steps,
            data_id=str(data_manifest["data_id"]),
            source_id=source_id,
        )
        for recipe in recipes
        if selected.accepts(recipe)
    ]
    if not cases:
        raise ValueError("case selection is empty")
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("generated duplicate case identifiers")
    plan_body = {
        "schema": "bsz_lm_plan_v1",
        "protocol": protocol.value,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": digest_file(protocol_path),
        "recipes_sha256": digest_file(recipes_path),
        "data": data_contract,
        "source_id": source_id,
        "selection": asdict(selected),
        "stage": stage,
        "cases": cases,
    }
    return {**plan_body, "plan_id": digest_value(plan_body)}


def write_plan(path: str | Path, plan: dict[str, Any]) -> None:
    write_json(path, plan)


def selected_values(values: Iterable[str] | None) -> tuple[str, ...]:
    return () if values is None else tuple(values)
