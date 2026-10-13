"""Strict collection of the terminal result of every case in a plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def collect_results(plan_path: str | Path, results_root: str | Path) -> list[dict[str, Any]]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema") != "bsz_lm_plan_v1":
        raise ValueError("unsupported plan schema")
    rows = []
    for case in plan["cases"]:
        result_path = Path(results_root) / case["case_id"] / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"case {case['case_id']} has no result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {
            "schema": "bsz_lm_result_v1",
            "status": "complete",
            "plan_id": plan["plan_id"],
            "case_id": case["case_id"],
            "run_fingerprint": case["run_fingerprint"],
        }
        mismatches = {
            name: (result.get(name), value)
            for name, value in expected.items()
            if result.get(name) != value
        }
        if mismatches:
            raise ValueError(f"result mismatch for {case['case_id']}: {mismatches}")
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["recipe"]["family"],
                "optimizer": case["recipe"]["optimizer"],
                "batch_tokens": case["recipe"]["batch_tokens"],
                "terminal_validation_loss": float(result["terminal_validation_loss"]),
                "run_fingerprint": case["run_fingerprint"],
            }
        )
    return rows


def write_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result collection")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
