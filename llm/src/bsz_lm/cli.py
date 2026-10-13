"""Command-line interface: validate recipes, build data manifests and plans, collect results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bsz_lm.collection import collect_results, write_results_csv
from bsz_lm.data_manifest import build_data_manifest, write_data_manifest
from bsz_lm.planning import CaseSelection, build_plan, write_plan
from bsz_lm.protocol import load_protocol
from bsz_lm.recipes import load_recipes
from bsz_lm.runtime import verify_shampoo_checkout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "protocol.yaml"


def _csv_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str | None) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_values(value))


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _verify_recipes(args: argparse.Namespace) -> None:
    recipes = load_recipes(args.recipes)
    _print(
        {
            "recipes": len(recipes),
            "families": sorted({recipe.family for recipe in recipes}),
            "optimizers": sorted({recipe.optimizer for recipe in recipes}),
            "batches": sorted({recipe.batch_tokens for recipe in recipes}),
        }
    )


def _manifest_data(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    data = protocol.value["data"]
    manifest = build_data_manifest(
        args.root,
        train_glob=data["train_glob"],
        validation_glob=data["validation_glob"],
        expected_magic=int(data["binary_magic"]),
        expected_version=int(data["binary_version"]),
    )
    validation_available = sum(item["tokens"] for item in manifest["validation"])
    if validation_available <= int(data["validation_tokens"]):
        raise ValueError("validation shards do not cover the fixed validation token count")
    for batch in protocol.batches:
        available_steps = sum(max(0, (item["tokens"] - 2) // batch) for item in manifest["train"])
        if available_steps < protocol.train_steps(batch):
            raise ValueError(
                f"training shards provide {available_steps} batches at {batch}, "
                f"need {protocol.train_steps(batch)}"
            )
    write_data_manifest(args.output, manifest)
    _print({"data_id": manifest["data_id"], "output": str(Path(args.output).resolve())})


def _plan(args: argparse.Namespace) -> None:
    selection = CaseSelection(
        families=_csv_values(args.families),
        optimizers=_csv_values(args.optimizers),
        batches=_csv_ints(args.batches),
    )
    plan = build_plan(
        args.protocol,
        args.recipes,
        args.data_manifest,
        selection=selection,
        stage=args.stage,
        max_steps=args.max_steps,
    )
    write_plan(args.output, plan)
    _print({"plan_id": plan["plan_id"], "cases": len(plan["cases"]), "output": args.output})


def _collect(args: argparse.Namespace) -> None:
    rows = collect_results(args.plan, args.results_root)
    write_results_csv(args.output, rows)
    _print({"results": len(rows), "output": str(Path(args.output).resolve())})


def _verify_runtime(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    source = verify_shampoo_checkout(
        args.shampoo_repository, protocol.value["runtime"]["shampoo_source"]
    )
    _print({"shampoo_import_root": str(source), "status": "verified"})


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="bsz-lm")
    commands = root.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-recipes", help="check a recipe CSV against the format")
    verify.add_argument("--recipes", required=True, help="recipe CSV (format in README.md)")
    verify.set_defaults(handler=_verify_recipes)

    manifest = commands.add_parser("manifest-data")
    manifest.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(handler=_manifest_data)

    plan = commands.add_parser("plan")
    plan.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    plan.add_argument("--recipes", required=True, help="recipe CSV (format in README.md)")
    plan.add_argument("--data-manifest", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--families", help="comma-separated W,H selection")
    plan.add_argument("--optimizers", help="comma-separated optimizer selection")
    plan.add_argument("--batches", help="comma-separated global token batches")
    plan.add_argument("--stage", default="tuned")
    plan.add_argument("--max-steps", type=int)
    plan.set_defaults(handler=_plan)

    collect = commands.add_parser("collect")
    collect.add_argument("--plan", required=True)
    collect.add_argument("--results-root", required=True)
    collect.add_argument("--output", required=True)
    collect.set_defaults(handler=_collect)

    runtime = commands.add_parser("verify-runtime")
    runtime.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    runtime.add_argument("--shampoo-repository", required=True)
    runtime.set_defaults(handler=_verify_runtime)

    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    main()
