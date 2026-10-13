"""Directional-branching stages (MuonW): plan, parent, anchor, branch, and a small campaign."""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from pathlib import Path

from bsz_lm.branchout.config import cases, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "parent", "anchor", "branch", "campaign"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["muon"], default="muon")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--basis", type=Path)
    parser.add_argument("--case", default="vanilla")
    parser.add_argument("--all-cases", action="store_true",
                        help="campaign: opt in to every registered LR/period case")
    args = parser.parse_args()
    config = load_config(args.config)
    registered = cases(config)
    if args.stage == "plan":
        print(json.dumps({"config": config.to_dict(), "algorithm": args.algorithm,
                          "cases": registered, "case_count": len(registered),
                          "default_campaign": "vanilla + four k=1 cells + four k=2,p=1 cells"},
                         indent=2))
        return
    if args.output is None:
        parser.error("execution requires a new --output directory")
    if args.stage in {"anchor", "branch"} and args.checkpoint is None:
        parser.error("anchor/branch requires --checkpoint")
    if args.stage == "branch" and args.basis is None:
        parser.error("branch requires --basis")
    if args.stage == "campaign" and not config.save_branch_checkpoint:
        parser.error("campaign requires save_branch_checkpoint: true for its exact replay check")
    import torch
    import torch.distributed as dist

    from bsz_lm.branchout.runner import (
        barrier,
        json_write,
        run_anchor,
        run_branch,
        run_parent,
    )

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
        barrier()

    try:
        if args.stage == "parent":
            run_parent(config, args.algorithm, args.output)
        elif args.stage == "anchor":
            run_anchor(config, args.algorithm, args.checkpoint, args.output)
        elif args.stage == "branch":
            case = next((c for c in registered if c["id"] == args.case), None)
            if case is None:
                raise ValueError(f"unknown registered case: {args.case}")
            run_branch(config, args.algorithm, args.checkpoint, args.basis, case, args.output)
        else:
            # One parent anchor, not the whole trajectory/grid.
            checkpoint = run_parent(config, args.algorithm, args.output / "parent")[-1]
            cleanup()
            basis = run_anchor(config, args.algorithm, checkpoint, args.output / "anchor")
            cleanup()
            selected = registered if args.all_cases else [
                c for c in registered if c["remainder_period"] == 1
                or (c["remainder_period"] == 2 and c["lr_power"] == 1)
            ]
            results = []
            for case in selected:
                results.append(run_branch(config, args.algorithm, checkpoint, basis, case,
                                          args.output / "branches" / case["id"]))
                cleanup()
            # A second vanilla continuation must restore model, optimizer and data exactly.
            replay = run_branch(config, args.algorithm, checkpoint, basis, registered[0],
                                args.output / "replay-vanilla")
            original = torch.load(results[0]["checkpoint"], map_location="cpu", weights_only=False)
            repeated = torch.load(replay["checkpoint"], map_location="cpu", weights_only=False)
            # Compare on CUDA too: numerical tests do not fall back to CPU.
            equal = all(torch.equal(value.cuda(), repeated["model"][key].cuda())
                        for key, value in original["model"].items())
            if not equal or results[0]["validation_loss"] != replay["validation_loss"]:
                raise AssertionError("common-checkpoint vanilla replay was not bitwise identical")
            if not dist.is_initialized() or dist.get_rank() == 0:
                json_write(args.output / "result.json", {
                    "status": "complete", "algorithm": args.algorithm,
                    "config": config.to_dict(), "branches": results,
                    "exact_vanilla_replay": equal,
                })
    except Exception:
        if int(os.environ.get("RANK", "0")) == 0:
            # Keep the failure outside existing stage directories; never replace success.
            failure = args.output.with_name(args.output.name + ".failure.json")
            failure.parent.mkdir(parents=True, exist_ok=True)
            if not failure.exists():
                json_write(failure, {"status": "failed", "stage": args.stage,
                                     "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
