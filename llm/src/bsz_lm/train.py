"""Eight-GPU execution entrypoint for one Track-3 case."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from bsz_lm.canonical import digest_tree, write_json
from bsz_lm.data import distributed_data_generator
from bsz_lm.data_manifest import verify_runtime_data
from bsz_lm.model import GPT, initialize_model
from bsz_lm.optimizers import set_scheduled_hparams, setup_optimizers
from bsz_lm.recipes import Recipe
from bsz_lm.runtime import verify_shampoo_checkout


def _recipe(case: dict[str, Any]) -> Recipe:
    recipe = Recipe(**case["recipe"])
    recipe.validate()
    return recipe


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "world_size": dist.get_world_size(),
    }


def _validate_plan(plan: dict[str, Any], case_index: int) -> dict[str, Any]:
    if plan.get("schema") != "bsz_lm_plan_v1":
        raise ValueError("unsupported plan schema")
    if not 0 <= case_index < len(plan["cases"]):
        raise IndexError(f"case index outside plan: {case_index}")
    source_id = digest_tree(
        Path(__file__).parent,
        ignored_names={"__pycache__", ".pytest_cache", ".ruff_cache"},
    )
    if source_id != plan["source_id"]:
        raise ValueError(f"source identity mismatch: running={source_id}, plan={plan['source_id']}")
    return plan["cases"][case_index]


def _validation_loss(
    model: GPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    microbatch_sequences: int,
    validation_tokens: int,
) -> float:
    model.eval()
    total = torch.zeros((), device=inputs.device)
    with torch.no_grad():
        if len(inputs) % microbatch_sequences:
            raise ValueError("validation sequences do not divide microbatch size")
        for start in range(0, len(inputs), microbatch_sequences):
            stop = start + microbatch_sequences
            total += model(inputs[start:stop], targets[start:stop])
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return float(total / validation_tokens)


def run(plan_path: str | Path, case_index: int) -> None:
    plan_file = Path(plan_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    case = _validate_plan(plan, case_index)
    recipe = _recipe(case)
    protocol = plan["protocol"]

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    timeout = datetime.timedelta(minutes=int(os.environ.get("TRACK3_DIST_TIMEOUT_MINUTES", "180")))
    dist.init_process_group("nccl", device_id=device, timeout=timeout)
    rank, world = dist.get_rank(), dist.get_world_size()
    expected_world = int(protocol["training"]["gpus_per_case"])
    if world != expected_world:
        raise ValueError(f"Track-3 cases require exactly {expected_world} ranks, got {world}")

    runtime = _runtime_metadata(device)
    data_root = Path(os.environ["TRACK3_DATA_ROOT"]).resolve()
    verify_runtime_data(
        plan["data"],
        data_root,
        hash_contents=os.environ.get("TRACK3_VERIFY_DATA_HASHES") == "1",
    )
    if recipe.optimizer == "shampoo":
        repository = os.environ.get("TRACK3_SHAMPOO_REPOSITORY")
        if not repository:
            raise RuntimeError("Shampoo cases require TRACK3_SHAMPOO_REPOSITORY")
        import_root = verify_shampoo_checkout(repository, protocol["runtime"]["shampoo_source"])
        sys.path.insert(0, str(import_root))

    seed = int(case["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = protocol["model"]
    model = GPT(
        vocabulary_size=int(model_config["vocabulary_size"]),
        layers=int(model_config["layers"]),
        width=int(model_config["width"]),
        head_dimension=int(model_config["head_dimension"]),
        mlp_multiplier=int(model_config["mlp_multiplier"]),
        logit_softcap=float(model_config["logit_softcap"]),
    ).to(device)
    initialize_model(model, recipe.family)
    for parameter in model.parameters():
        dist.broadcast(parameter.detach(), 0)
    if bool(protocol["training"]["compile"]):
        model.compile(dynamic=False)
    optimizers = setup_optimizers(model, recipe)

    sequence_length = int(model_config["sequence_length"])
    validation_tokens = int(protocol["data"]["validation_tokens"])
    batch_tokens = recipe.batch_tokens
    microbatch_sequences = max(1, min(64, batch_tokens // (world * sequence_length)))
    validation_inputs, validation_targets = next(
        distributed_data_generator(
            str(data_root / protocol["data"]["validation_glob"]),
            validation_tokens,
            sequence_length=sequence_length,
            device=device,
        )
    )
    train_loader = distributed_data_generator(
        str(data_root / protocol["data"]["train_glob"]),
        batch_tokens,
        sequence_length=sequence_length,
        device=device,
    )

    output_root = Path(os.environ.get("TRACK3_OUTPUT_ROOT", plan_file.parent / "results"))
    case_root = output_root / case["case_id"]
    if rank == 0:
        case_root.mkdir(parents=True, exist_ok=False)
        write_json(
            case_root / "started.json",
            {
                "schema": "bsz_lm_started_v1",
                "plan_id": plan["plan_id"],
                "case": case,
                "runtime": runtime,
            },
        )
    dist.barrier()

    train_steps = int(case["effective_train_steps"])
    validation_frequency = int(protocol["training"]["validation_frequency"])
    final_frequency = int(protocol["training"]["final_validation_frequency"])
    final_fraction = float(protocol["training"]["final_validation_fraction"])
    validation_history: list[dict[str, float | int]] = []
    training_time = 0.0
    last_validation_step = 0
    dist.barrier()
    clock = time.perf_counter()
    for step in range(train_steps + 1):
        frequency = validation_frequency if step / train_steps < final_fraction else final_frequency
        if step == train_steps or step % frequency == 0:
            dist.barrier()
            elapsed = time.perf_counter() - clock
            training_time += elapsed
            loss = _validation_loss(
                model,
                validation_inputs,
                validation_targets,
                microbatch_sequences=microbatch_sequences,
                validation_tokens=validation_tokens,
            )
            event = {"step": step, "validation_loss": loss, "training_seconds": training_time}
            validation_history.append(event)
            if rank == 0:
                print(json.dumps(event, sort_keys=True), flush=True)
            last_validation_step = step
            dist.barrier()
            clock = time.perf_counter()
        if step == train_steps:
            break

        inputs, targets = next(train_loader)
        if len(inputs) % microbatch_sequences:
            raise ValueError("training sequences do not divide microbatch size")
        for start in range(0, len(inputs), microbatch_sequences):
            stop = start + microbatch_sequences
            model(inputs[start:stop], targets[start:stop]).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient: {name}")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        set_scheduled_hparams(
            optimizers,
            step=step,
            train_steps=train_steps,
            auxiliary_cooldown=recipe.aux_cooldown_frac,
            schedule_profile=protocol["training"]["schedule_profiles"][
                f"{recipe.family}/{recipe.optimizer}"
            ],
        )
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)

    if rank == 0:
        write_json(
            case_root / "result.json",
            {
                "schema": "bsz_lm_result_v1",
                "status": "complete",
                "plan_id": plan["plan_id"],
                "case_id": case["case_id"],
                "run_fingerprint": case["run_fingerprint"],
                "terminal_validation_loss": validation_history[-1]["validation_loss"],
                "validation_history": validation_history,
                "runtime": runtime,
                "training_seconds": training_time,
                "last_validation_step": last_validation_step,
            },
        )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--case-index", required=True, type=int)
    args = parser.parse_args()
    run(args.plan, args.case_index)


if __name__ == "__main__":  # pragma: no cover
    main()
