"""Train the CIFAR GroupNorm ResNet-18 on one fixed CIFAR-5M data stream.

The command line extends that of ``train_groupnorm_resnet.py`` (the CIFAR-10
trainer), so the coordinate-descent tuner can drive either script with the same
hyperparameter flags.  Here an "epoch" means one data checkpoint of
``--train_size`` new examples.  Samples are never reused.  With 100 checkpoints
and train_size 45056, every run processes exactly 4,505,600 examples, as many
as 100 epochs of the CIFAR-10 trainer over its 45,056 training examples.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_groupnorm_resnet import (
    GroupNormResNet18,
    Muon,
    apply_learning_rate_rule,
    apply_momentum_rule,
    apply_weight_decay_rule,
    atomic_write_json,
    evaluate,
    maybe_initialize_wandb,
    module_weight_norms,
    split_optimizer_parameters,
)


PART_COUNT = 6
TRAIN_PARTS = (0, 1, 2, 3, 4)
VALIDATION_PARTS = (5,)
NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)


def _extract_npy_member(archive: zipfile.ZipFile, member: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    with archive.open(member) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    os.replace(temporary, target)


def prepare_cifar5m(data_root: Path) -> dict[str, Any]:
    """Extract the six official NPZ parts into memory-mappable NPY files."""
    raw_root = data_root / "raw"
    prepared_root = data_root / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_parts: list[dict[str, Any]] = []
    for part_index in range(PART_COUNT):
        source = raw_root / f"part{part_index}.npz"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Download all six official CIFAR-5M parts first."
            )
        image_target = prepared_root / f"images_part{part_index}.npy"
        label_target = prepared_root / f"labels_part{part_index}.npy"
        if not image_target.exists() or not label_target.exists():
            candidates: list[tuple[str, tuple[int, ...], np.dtype[Any]]] = []
            extracted: list[Path] = []
            with zipfile.ZipFile(source) as archive:
                members = [name for name in archive.namelist() if name.endswith(".npy")]
                if len(members) < 2:
                    raise RuntimeError(f"Expected image and label arrays in {source}")
                for member_index, member in enumerate(members):
                    candidate_path = prepared_root / (
                        f"part{part_index}_member{member_index}.npy"
                    )
                    _extract_npy_member(archive, member, candidate_path)
                    array = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
                    candidates.append((member, tuple(array.shape), array.dtype))
                    extracted.append(candidate_path)
                    del array
            image_indices = [
                index
                for index, (_, shape, _) in enumerate(candidates)
                if len(shape) == 4 and 3 in (shape[1], shape[-1])
            ]
            if len(image_indices) != 1:
                raise RuntimeError(f"Could not identify one image array in {source}: {candidates}")
            image_index = image_indices[0]
            image_count = candidates[image_index][1][0]
            label_indices = [
                index
                for index, (_, shape, _) in enumerate(candidates)
                if index != image_index
                and len(shape) in (1, 2)
                and shape[0] == image_count
                and (len(shape) == 1 or shape[1] == 1)
            ]
            if len(label_indices) != 1:
                raise RuntimeError(f"Could not identify one label array in {source}: {candidates}")
            label_index = label_indices[0]
            os.replace(extracted[image_index], image_target)
            os.replace(extracted[label_index], label_target)
            for index, path in enumerate(extracted):
                if index not in (image_index, label_index):
                    path.unlink(missing_ok=True)

        images = np.load(image_target, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_target, mmap_mode="r", allow_pickle=False)
        if images.ndim != 4 or images.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Invalid prepared arrays for part {part_index}: "
                f"images={images.shape}, labels={labels.shape}"
            )
        manifest_parts.append(
            {
                "part": part_index,
                "count": int(images.shape[0]),
                "images": str(image_target),
                "labels": str(label_target),
                "image_shape": list(images.shape),
                "image_dtype": str(images.dtype),
                "label_shape": list(labels.shape),
                "label_dtype": str(labels.dtype),
                "source": str(source),
            }
        )
        del images, labels
    manifest = {
        "status": "complete",
        "parts": manifest_parts,
        "train_parts": list(TRAIN_PARTS),
        "validation_parts": list(VALIDATION_PARTS),
    }
    atomic_write_json(prepared_root / "manifest.json", manifest)
    (prepared_root / "PREPARE_COMPLETE").write_text(
        f"completed_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n",
        encoding="utf-8",
    )
    return manifest


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class ShardedCifar5m(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sequential view of prepared shards with deterministic augmentation."""

    def __init__(
        self,
        prepared_root: Path,
        parts: tuple[int, ...],
        size: int,
        augment: bool,
        augmentation_seed: int,
    ) -> None:
        self.image_paths = [prepared_root / f"images_part{part}.npy" for part in parts]
        self.label_paths = [prepared_root / f"labels_part{part}.npy" for part in parts]
        counts = [
            int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
            for path in self.image_paths
        ]
        self.offsets = np.cumsum([0, *counts], dtype=np.int64).tolist()
        if not 0 < size <= self.offsets[-1]:
            raise ValueError(f"Requested {size} samples from a pool of {self.offsets[-1]}")
        self.size = size
        self.augment = augment
        self.augmentation_seed = augmentation_seed
        self.images: list[np.ndarray[Any, Any]] | None = None
        self.labels: list[np.ndarray[Any, Any]] | None = None
        self.mean = torch.tensor(NORMALIZATION_MEAN, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(NORMALIZATION_STD, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return self.size

    def _open(self) -> None:
        if self.images is None:
            self.images = [
                np.load(path, mmap_mode="r", allow_pickle=False)
                for path in self.image_paths
            ]
            self.labels = [
                np.load(path, mmap_mode="r", allow_pickle=False)
                for path in self.label_paths
            ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        self._open()
        assert self.images is not None and self.labels is not None
        shard = bisect.bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[shard]
        image_array = self.images[shard][local_index]
        if image_array.shape[0] == 3:
            image = torch.from_numpy(np.array(image_array, copy=True))
        elif image_array.shape[-1] == 3:
            image = torch.from_numpy(np.array(image_array, copy=True)).permute(2, 0, 1)
        else:
            raise RuntimeError(f"Unexpected CIFAR-5M image shape {image_array.shape}")
        image = image.float().div_(255.0)
        if self.augment:
            random_bits = _mix64(self.augmentation_seed ^ index)
            image = F.pad(image, (4, 4, 4, 4), mode="reflect")
            top = random_bits % 9
            left = (random_bits >> 8) % 9
            image = image[:, top : top + 32, left : left + 32]
            if ((random_bits >> 16) & 1) == 1:
                image = image.flip(-1)
        image = (image - self.mean) / self.std
        target = torch.tensor(int(np.asarray(self.labels[shard][local_index]).reshape(-1)[0]))
        return image, target


def make_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    total_train_examples = args.epochs * args.train_size
    if args.train_size % args.batch_size != 0:
        raise ValueError(
            f"train_size={args.train_size} must be divisible by batch_size={args.batch_size}"
        )
    prepared_root = args.data / "prepared"
    train_data = ShardedCifar5m(
        prepared_root=prepared_root,
        parts=TRAIN_PARTS,
        size=total_train_examples,
        augment=True,
        augmentation_seed=args.split_seed + 1,
    )
    validation_data = ShardedCifar5m(
        prepared_root=prepared_root,
        parts=VALIDATION_PARTS,
        size=args.validation_size,
        augment=False,
        augmentation_seed=args.split_seed + 2,
    )
    common = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    return {
        "train": DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "validation": DataLoader(
            validation_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }


def parameter_norm(parameters: list[nn.Parameter]) -> float:
    squared = sum(parameter.detach().float().pow(2).sum().item() for parameter in parameters)
    return math.sqrt(squared)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--prepare_data_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--summary_window", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--base_batch_size", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=45056)
    parser.add_argument("--validation_size", type=int, default=4944)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--split_seed", type=int, default=20260823)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--group_norm_groups", type=int, default=32)
    parser.add_argument("--muon_lr", type=float, default=0.05)
    parser.add_argument("--adam_lr", type=float, default=5e-4)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--muon_lr_rule", choices=("fixed", "sqrt", "linear"), default="fixed")
    parser.add_argument("--adam_lr_rule", choices=("fixed", "sqrt", "linear"), default="fixed")
    parser.add_argument("--muon_momentum_rule", choices=("fixed", "sample"), default="fixed")
    parser.add_argument("--adam_beta1_rule", choices=("fixed", "sample"), default="fixed")
    parser.add_argument("--adam_beta2_rule", choices=("fixed", "sample"), default="fixed")
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_decay_rule", choices=("fixed", "sqrt", "linear"), default="fixed")
    parser.add_argument("--muon_weight_decay_rule", choices=("fixed", "sqrt", "linear"),
                        help="Override the shared weight-decay rule for Muon only")
    parser.add_argument("--adam_weight_decay_rule", choices=("fixed", "sqrt", "linear"),
                        help="Override the shared weight-decay rule for AdamW only")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--run_name", type=str, default="cifar5m_groupnorm")
    parser.add_argument("--wandb_project", type=str, default="cifar-batch-size")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_root", type=Path)
    return parser.parse_args()


def realized_weight_decays(args: argparse.Namespace, ratio: float) -> dict[str, float]:
    return {
        "muon_weight_decay": apply_weight_decay_rule(
            args.muon_weight_decay, ratio, args.muon_weight_decay_rule or args.weight_decay_rule),
        "adam_weight_decay": apply_weight_decay_rule(
            args.adam_weight_decay, ratio, args.adam_weight_decay_rule or args.weight_decay_rule),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.prepare_data_only:
        manifest = prepare_cifar5m(args.data)
        print(json.dumps(manifest, sort_keys=True))
        return manifest
    if args.result is None:
        raise ValueError("--result is required for training runs")
    if args.epochs < 1 or args.summary_window < 1:
        raise ValueError("epochs and summary_window must be positive")
    if args.validation_size < 1:
        raise ValueError("validation_size must be positive")
    if args.muon_weight_decay < 0.0 or args.adam_weight_decay < 0.0:
        raise ValueError("Weight decay values cannot be negative")
    if not (args.data / "prepared" / "PREPARE_COMPLETE").exists():
        raise FileNotFoundError("CIFAR-5M is not prepared. Run --prepare_data_only first.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0")
    ratio = args.batch_size / args.base_batch_size
    realized = {
        "muon_lr": apply_learning_rate_rule(args.muon_lr, ratio, args.muon_lr_rule),
        "adam_lr": apply_learning_rate_rule(args.adam_lr, ratio, args.adam_lr_rule),
        "muon_momentum": apply_momentum_rule(args.muon_momentum, ratio, args.muon_momentum_rule),
        "adam_beta1": apply_momentum_rule(args.adam_beta1, ratio, args.adam_beta1_rule),
        "adam_beta2": apply_momentum_rule(args.adam_beta2, ratio, args.adam_beta2_rule),
        **realized_weight_decays(args, ratio),
    }
    for name in ("muon_momentum", "adam_beta1", "adam_beta2"):
        if not 0.0 <= realized[name] < 1.0:
            raise ValueError(f"Realized {name} must be in [0, 1), received {realized[name]}")

    loaders = make_loaders(args)
    model = GroupNormResNet18(groups=args.group_norm_groups).to(device)
    muon_parameters, adam_parameters, muon_names, adam_names = split_optimizer_parameters(model)
    muon_optimizer = Muon(
        muon_parameters,
        lr=realized["muon_lr"],
        momentum=realized["muon_momentum"],
        weight_decay=realized["muon_weight_decay"],
    )
    adam_optimizer = torch.optim.AdamW(
        adam_parameters,
        lr=realized["adam_lr"],
        betas=(realized["adam_beta1"], realized["adam_beta2"]),
        weight_decay=realized["adam_weight_decay"],
    )
    total_examples = args.epochs * args.train_size
    configuration = {
        **vars(args),
        "data": str(args.data),
        "result": str(args.result),
        "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else "",
        "dataset": "CIFAR-5M",
        "training_protocol": "single_pass_fixed_stream",
        "data_checkpoint_examples": args.train_size,
        "total_processed_examples": total_examples,
        "batch_ratio": ratio,
        "base_hparams": {
            "muon_lr": args.muon_lr,
            "adam_lr": args.adam_lr,
            "muon_momentum": args.muon_momentum,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "muon_weight_decay": args.muon_weight_decay,
            "adam_weight_decay": args.adam_weight_decay,
        },
        "realized_hparams": realized,
        "muon_parameter_names": muon_names,
        "adam_parameter_names": adam_names,
    }
    wandb_run = maybe_initialize_wandb(
        args.wandb_mode,
        args.wandb_project,
        args.run_name,
        args.wandb_group,
        configuration,
    )
    use_bfloat16 = args.precision == "bfloat16"
    started_at = time.time()
    zero_validation_loss, zero_validation_accuracy = evaluate(
        model, loaders["validation"], device, use_bfloat16
    )
    history: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "data_checkpoint": 0,
            "processed_examples": 0,
            "optimizer_steps": 0,
            "train_loss": float("nan"),
            "train_accuracy": float("nan"),
            "validation_loss": zero_validation_loss,
            "validation_accuracy": zero_validation_accuracy,
        }
    ]
    if wandb_run is not None:
        wandb_run.log(history[0], step=0)
    print(
        f"data_checkpoint=0 processed_examples=0 validation_loss={zero_validation_loss:.6f}",
        flush=True,
    )

    model.train()
    processed = 0
    optimizer_steps = 0
    interval_loss_sum = 0.0
    interval_correct = 0
    interval_examples = 0
    divergence_error = ""
    try:
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            muon_optimizer.zero_grad(set_to_none=True)
            adam_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss {loss.item()}")
            loss.backward()
            muon_optimizer.step()
            adam_optimizer.step()
            count = int(targets.numel())
            processed += count
            optimizer_steps += 1
            interval_examples += count
            interval_loss_sum += loss.item() * count
            interval_correct += int((logits.argmax(dim=1) == targets).sum().item())
            if processed % args.train_size == 0:
                checkpoint_index = processed // args.train_size
                validation_loss, validation_accuracy = evaluate(
                    model, loaders["validation"], device, use_bfloat16
                )
                if not math.isfinite(validation_loss):
                    raise FloatingPointError(f"Nonfinite validation loss {validation_loss}")
                row: dict[str, float | int] = {
                    "epoch": checkpoint_index,
                    "data_checkpoint": checkpoint_index,
                    "processed_examples": processed,
                    "optimizer_steps": optimizer_steps,
                    "train_loss": interval_loss_sum / interval_examples,
                    "train_accuracy": interval_correct / interval_examples,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
                history.append(row)
                if wandb_run is not None:
                    wandb_run.log(row, step=checkpoint_index)
                print(
                    f"data_checkpoint={checkpoint_index} processed_examples={processed} "
                    f"optimizer_steps={optimizer_steps} train_loss={row['train_loss']:.6f} "
                    f"validation_loss={validation_loss:.6f}",
                    flush=True,
                )
                interval_loss_sum = 0.0
                interval_correct = 0
                interval_examples = 0
                model.train()
        if processed != total_examples:
            raise RuntimeError(f"Processed {processed} examples, expected {total_examples}")
    except FloatingPointError as exception:
        divergence_error = str(exception)
        print(f"[diverged] {divergence_error}", flush=True)

    validation_losses = np.asarray(
        [float(row["validation_loss"]) for row in history[-args.summary_window :]],
        dtype=np.float64,
    )
    final_weight_norms = module_weight_norms(model)
    final_weight_norm = parameter_norm(list(model.parameters()))
    final_muon_weight_norm = parameter_norm(muon_parameters)
    final_adam_weight_norm = parameter_norm(adam_parameters)
    checkpoint_path: Path | None = None
    checkpoint_bytes = 0
    if args.save_checkpoint:
        checkpoint_root = args.checkpoint_root or args.result.parent
        checkpoint_path = checkpoint_root / args.result.parent.name / f"{args.result.stem}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "muon_optimizer": muon_optimizer.state_dict(),
                "adam_optimizer": adam_optimizer.state_dict(),
                "configuration": configuration,
                "processed_examples": processed,
                "optimizer_steps": optimizer_steps,
                "final_weight_norm": final_weight_norm,
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size

    result = {
        "status": "diverged" if divergence_error else "complete",
        "divergence_error": divergence_error,
        "configuration": configuration,
        "metrics": {
            "validation_loss_final": float(validation_losses[-1]),
            "validation_loss_mean": float(validation_losses.mean()),
            "validation_loss_std": float(validation_losses.std(ddof=0)),
            "validation_loss_min": float(validation_losses.min()),
            "runtime_seconds": time.time() - started_at,
            "processed_examples": processed,
            "optimizer_steps": optimizer_steps,
            "final_weight_norm": final_weight_norm,
            "final_muon_weight_norm": final_muon_weight_norm,
            "final_adam_weight_norm": final_adam_weight_norm,
        },
        "history": history,
        "final_weight_norms": final_weight_norms,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_bytes": checkpoint_bytes,
    }
    atomic_write_json(args.result, result)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                **result["metrics"],
                **{f"final_weight_norm/{name}": value for name, value in final_weight_norms.items()},
                "checkpoint_path": result["checkpoint_path"],
                "checkpoint_bytes": checkpoint_bytes,
            }
        )
        wandb_run.finish()
    print(json.dumps(result["metrics"], sort_keys=True), flush=True)
    return result


def main() -> None:
    args = parse_arguments()
    try:
        run(args)
    except Exception as exception:
        failure = {
            "status": "failed",
            "error": repr(exception),
            "traceback": traceback.format_exc(),
        }
        if args.result is not None:
            atomic_write_json(args.result, failure)
        print(failure["traceback"], flush=True)
        raise


if __name__ == "__main__":
    main()
