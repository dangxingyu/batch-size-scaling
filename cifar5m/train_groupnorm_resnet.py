"""Train a GroupNorm ResNet-18 on CIFAR-10 with MuonW and AdamW.

Each call trains one configuration.  Convolution weights are updated by Muon with
decoupled weight decay (MuonW); the classifier, biases and GroupNorm affine
parameters are updated by AdamW.  The model, the Muon implementation and the
scaling-rule helpers are shared with ``train_groupnorm_resnet_cifar5m.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


# Newton-Schulz iteration and its quintic coefficients (3.4445, -4.7750, 2.0315) adapted from
# Keller Jordan's Muon, https://github.com/KellerJordan/Muon (MIT License).
def zeropower_via_newtonschulz5(
    matrix: torch.Tensor,
    steps: int = 5,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Approximate the polar factor of a two dimensional matrix."""
    if matrix.ndim != 2:
        raise ValueError(f"Muon expects a matrix, received shape {tuple(matrix.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.bfloat16()
    x = x / (x.norm() + epsilon)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * gram @ gram
        x = a * x + correction @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for tensors with at least two dimensions.

    Convolution kernels are flattened to output channels by remaining
    coordinates before orthogonalization, then restored to their original
    shape.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        momentum: float,
        weight_decay: float = 0.0,
        backend_steps: int = 5,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Muon momentum must be in [0, 1), received {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "backend_steps": backend_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            backend_steps = int(group["backend_steps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim < 2:
                    raise ValueError("Muon parameter groups may only contain matrix shaped tensors")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["step"] = 0
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(parameter.grad, alpha=1.0 - momentum)
                state["step"] += 1
                bias_correction = 1.0 - momentum ** int(state["step"])
                update = buffer / bias_correction
                matrix = update.reshape(update.shape[0], -1)
                orthogonal_update = zeropower_via_newtonschulz5(
                    matrix,
                    steps=backend_steps,
                ).reshape_as(update)
                if weight_decay:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(orthogonal_update, alpha=-learning_rate)
        return loss


class GroupNormResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, groups: int = 32) -> None:
        super().__init__()

        def norm_layer(channels: int) -> nn.Module:
            if channels % groups != 0:
                raise ValueError(f"{channels} channels are not divisible by {groups} groups")
            return nn.GroupNorm(groups, channels)

        model = tv_models.resnet18(
            weights=None,
            num_classes=num_classes,
            norm_layer=norm_layer,
        )
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def split_optimizer_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str], list[str]]:
    muon_parameters: list[nn.Parameter] = []
    adam_parameters: list[nn.Parameter] = []
    muon_names: list[str] = []
    adam_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_classifier = any(token in name for token in (".fc.", ".head.", ".classifier."))
        if parameter.ndim >= 2 and not is_classifier:
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            adam_parameters.append(parameter)
            adam_names.append(name)
    if not muon_parameters or not adam_parameters:
        raise RuntimeError("Both Muon and Adam parameter groups must be nonempty")
    return muon_parameters, adam_parameters, muon_names, adam_names


def apply_learning_rate_rule(base_value: float, ratio: float, rule: str) -> float:
    powers = {"fixed": 0.0, "sqrt": 0.5, "linear": 1.0}
    if rule not in powers:
        raise ValueError(f"Unknown learning rate rule {rule}")
    return base_value * ratio ** powers[rule]


def apply_momentum_rule(base_value: float, ratio: float, rule: str) -> float:
    if rule == "fixed":
        return base_value
    if rule == "sample":
        return base_value**ratio
    raise ValueError(f"Unknown momentum rule {rule}")


def apply_weight_decay_rule(base_value: float, ratio: float, rule: str) -> float:
    powers = {"fixed": 0.0, "sqrt": 0.5, "linear": 1.0}
    if rule not in powers:
        raise ValueError(f"Unknown weight decay rule {rule}")
    return base_value * ratio ** powers[rule]


def prepare_cifar10(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)


def make_loaders(
    data_root: Path,
    batch_size: int,
    evaluation_batch_size: int,
    train_size: int,
    split_seed: int,
    workers: int,
) -> dict[str, DataLoader]:
    if train_size % batch_size != 0:
        raise ValueError(
            f"train_size={train_size} must be divisible by batch_size={batch_size}"
        )
    normalize = T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    training_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = T.Compose([T.ToTensor(), normalize])
    augmented_training_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=training_transform,
    )
    evaluation_training_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=evaluation_transform,
    )
    test_data = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=evaluation_transform,
    )
    if not 0 < train_size < len(augmented_training_data):
        raise ValueError(f"train_size must be between 1 and 49999, received {train_size}")
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(augmented_training_data), generator=generator).tolist()
    training_indices = permutation[:train_size]
    validation_indices = permutation[train_size:]
    persistent_workers = workers > 0
    training_generator = torch.Generator().manual_seed(split_seed + 1)
    common_evaluation = {
        "batch_size": evaluation_batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers,
    }
    return {
        "train": DataLoader(
            Subset(augmented_training_data, training_indices),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            generator=training_generator,
        ),
        "train_eval": DataLoader(
            Subset(evaluation_training_data, training_indices),
            **common_evaluation,
        ),
        "validation": DataLoader(
            Subset(evaluation_training_data, validation_indices),
            **common_evaluation,
        ),
        "test": DataLoader(test_data, **common_evaluation),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            logits = model(inputs)
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        examples += int(targets.numel())
    return loss_sum / examples, correct / examples


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[float, float]:
    model.train()
    muon_optimizer, adam_optimizer = optimizers
    loss_sum = 0.0
    correct = 0
    examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        muon_optimizer.zero_grad(set_to_none=True)
        adam_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss {loss.item()}")
        loss.backward()
        muon_optimizer.step()
        adam_optimizer.step()
        loss_sum += loss.item() * targets.numel()
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        examples += int(targets.numel())
    return loss_sum / examples, correct / examples


def module_weight_norms(model: nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for module_name, module in model.named_modules():
        squared_norm = 0.0
        found_weight = False
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name == "weight":
                found_weight = True
                squared_norm += parameter.detach().float().pow(2).sum().item()
        if found_weight:
            norms[module_name or "root"] = math.sqrt(squared_norm)
    return norms


def maybe_initialize_wandb(
    mode: str,
    project: str,
    run_name: str,
    group: str,
    configuration: dict[str, Any],
) -> Any:
    if mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=project,
            name=run_name,
            group=group or None,
            mode=mode,
            config=configuration,
        )
    except Exception as exception:
        print(f"[warn] W&B initialization failed, continuing without W&B: {exception}", flush=True)
        return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=False)
    parser.add_argument("--prepare_data_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--summary_window", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--base_batch_size", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=45056)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--split_seed", type=int, default=20260822)
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
    parser.add_argument(
        "--weight_decay_rule",
        choices=("fixed", "sqrt", "linear"),
        default="fixed",
    )
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--run_name", type=str, default="cifar_groupnorm")
    parser.add_argument("--wandb_project", type=str, default="cifar-batch-size")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--save_checkpoint", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.prepare_data_only:
        prepare_cifar10(args.data)
        print(f"CIFAR 10 is ready under {args.data}")
        return {"status": "data_ready", "data": str(args.data)}
    if args.result is None:
        raise ValueError("--result is required for training runs")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.summary_window < 1:
        raise ValueError("--summary_window must be positive")
    if args.muon_weight_decay < 0.0 or args.adam_weight_decay < 0.0:
        raise ValueError("Weight decay values cannot be negative")
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
        "muon_momentum": apply_momentum_rule(
            args.muon_momentum,
            ratio,
            args.muon_momentum_rule,
        ),
        "adam_beta1": apply_momentum_rule(args.adam_beta1, ratio, args.adam_beta1_rule),
        "adam_beta2": apply_momentum_rule(args.adam_beta2, ratio, args.adam_beta2_rule),
        "muon_weight_decay": apply_weight_decay_rule(
            args.muon_weight_decay,
            ratio,
            args.weight_decay_rule,
        ),
        "adam_weight_decay": apply_weight_decay_rule(
            args.adam_weight_decay,
            ratio,
            args.weight_decay_rule,
        ),
    }
    for key in ("muon_momentum", "adam_beta1", "adam_beta2"):
        if not 0.0 <= realized[key] < 1.0:
            raise ValueError(f"Realized {key} must be in [0, 1), received {realized[key]}")
    for optimizer_name in ("muon", "adam"):
        learning_rate = realized[f"{optimizer_name}_lr"]
        weight_decay = realized[f"{optimizer_name}_weight_decay"]
        if learning_rate * weight_decay >= 1.0:
            print(
                f"[warn] realized {optimizer_name} learning_rate * weight_decay is "
                f"{learning_rate * weight_decay:.6g}, so the decoupled decay factor "
                "is nonpositive",
                flush=True,
            )

    loaders = make_loaders(
        data_root=args.data,
        batch_size=args.batch_size,
        evaluation_batch_size=args.eval_batch_size,
        train_size=args.train_size,
        split_seed=args.split_seed,
        workers=args.workers,
    )
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
    configuration = {
        **vars(args),
        "data": str(args.data),
        "result": str(args.result),
        "batch_ratio": ratio,
        "validation_size": 50000 - args.train_size,
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
        mode=args.wandb_mode,
        project=args.wandb_project,
        run_name=args.run_name,
        group=args.wandb_group,
        configuration=configuration,
    )
    use_bfloat16 = args.precision == "bfloat16"
    started_at = time.time()
    zero_train_loss, zero_train_accuracy = evaluate(
        model,
        loaders["train_eval"],
        device,
        use_bfloat16,
    )
    zero_validation_loss, zero_validation_accuracy = evaluate(
        model,
        loaders["validation"],
        device,
        use_bfloat16,
    )
    zero_weight_norms = module_weight_norms(model)
    history: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "train_loss": zero_train_loss,
            "train_accuracy": zero_train_accuracy,
            "validation_loss": zero_validation_loss,
            "validation_accuracy": zero_validation_accuracy,
        }
    ]
    if wandb_run is not None:
        wandb_run.log(
            {
                **history[0],
                **{f"weight_norm/{name}": value for name, value in zero_weight_norms.items()},
            },
            step=0,
        )
    print(
        f"epoch=0 train_loss={zero_train_loss:.6f} "
        f"validation_loss={zero_validation_loss:.6f}",
        flush=True,
    )

    divergence_error = ""
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_epoch(
                model,
                loaders["train"],
                (muon_optimizer, adam_optimizer),
                device,
                use_bfloat16,
            )
            validation_loss, validation_accuracy = evaluate(
                model,
                loaders["validation"],
                device,
                use_bfloat16,
            )
            if not math.isfinite(validation_loss):
                raise FloatingPointError(f"Nonfinite validation loss {validation_loss}")
            epoch_metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            history.append(epoch_metrics)
            if wandb_run is not None:
                wandb_run.log(epoch_metrics, step=epoch)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"validation_loss={validation_loss:.6f}",
                flush=True,
            )
    except FloatingPointError as exception:
        divergence_error = str(exception)
        print(f"[diverged] {divergence_error}", flush=True)

    if divergence_error:
        test_loss, test_accuracy = float("nan"), float("nan")
    else:
        test_loss, test_accuracy = evaluate(model, loaders["test"], device, use_bfloat16)
    validation_losses = np.asarray(
        [float(row["validation_loss"]) for row in history[-args.summary_window :]],
        dtype=np.float64,
    )
    final_weight_norms = module_weight_norms(model)
    checkpoint_path = args.result.with_suffix(".pt") if args.save_checkpoint else None
    checkpoint_bytes = 0
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".tmp"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "muon_optimizer": muon_optimizer.state_dict(),
                "adam_optimizer": adam_optimizer.state_dict(),
                "configuration": configuration,
            },
            temporary_checkpoint_path,
        )
        os.replace(temporary_checkpoint_path, checkpoint_path)
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
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "runtime_seconds": time.time() - started_at,
        },
        "history": history,
        "zero_step_weight_norms": zero_weight_norms,
        "final_weight_norms": final_weight_norms,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "checkpoint_bytes": checkpoint_bytes,
    }
    atomic_write_json(args.result, result)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                **result["metrics"],
                **{
                    f"final_weight_norm/{name}": value
                    for name, value in final_weight_norms.items()
                },
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
