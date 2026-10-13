"""Hidden-matrix optimizers and the W/H wrappers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from bsz_lm.model import GPT, parameter_groups
from bsz_lm.recipes import Recipe


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.mT)


def _initial_orthogonal_matrix(matrix: Tensor) -> Tensor:
    matrix = _symmetrize(matrix.float())
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.float32)
    try:
        _, vectors = torch.linalg.eigh(matrix + 1e-30 * eye)
    except RuntimeError:
        _, vectors = torch.linalg.eigh((matrix + 1e-30 * eye).double())
        vectors = vectors.float()
    return torch.flip(vectors, dims=[1]).contiguous()


def zeropower_via_newton_schulz(matrix: Tensor) -> Tensor:
    """The Track-3 12-step quintic Newton--Schulz map, including BF16 arithmetic."""

    if matrix.ndim < 2:
        raise ValueError("Muon accepts matrix gradients only")
    value = matrix.bfloat16()
    transposed = value.size(-2) > value.size(-1)
    if transposed:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        gram = value @ value.mT
        value = 2 * value + (-1.5 * gram + 0.5 * gram @ gram) @ value
    return value.mT if transposed else value


def hyperball_update_(
    parameter: Tensor, update: Tensor, learning_rate: float, *, epsilon: float = 1e-10
) -> None:
    """Apply the exact norm-preserving matrix wrapper from the paper."""

    radius = parameter.float().norm()
    direction_norm = update.float().norm().clamp_min(epsilon)
    candidate = parameter.float() - learning_rate * radius * update.float() / direction_norm
    parameter.copy_((radius * candidate / candidate.norm().clamp_min(epsilon)).to(parameter.dtype))


def wd_update_(
    parameter: Tensor, update: Tensor, learning_rate: float, weight_decay: float
) -> None:
    parameter.mul_(1 - learning_rate * weight_decay)
    parameter.add_(update.to(parameter.dtype), alpha=-learning_rate)


def _project_to_basis(gradient: Tensor, state: dict[str, Any]) -> Tensor:
    left, right = state["Q"]
    return left.mT @ gradient.float() @ right


def _project_from_basis(gradient: Tensor, state: dict[str, Any]) -> Tensor:
    left, right = state["Q"]
    return left @ gradient.float() @ right.mT


def _initialize_klsoap(state: dict[str, Any], gradient: Tensor, shampoo_beta: float) -> None:
    gradient = gradient.detach().float()
    rows, columns = gradient.shape
    state.update(
        step=0,
        GG=[
            (gradient @ gradient.mT / columns).contiguous(),
            (gradient.mT @ gradient / rows).contiguous(),
        ],
        exp_avg=torch.zeros_like(gradient),
        exp_avg_sq=torch.zeros_like(gradient),
        shampoo_beta=shampoo_beta,
    )
    state["Q"] = [
        _initial_orthogonal_matrix(state["GG"][0]),
        _initial_orthogonal_matrix(state["GG"][1]),
    ]
    initial_inverse = 0.1**-0.5
    state["eigen_sqrt_inv"] = [
        torch.full((rows,), initial_inverse, device=gradient.device),
        torch.full((columns,), initial_inverse, device=gradient.device),
    ]


def _update_eigen_inverse(state: dict[str, Any], diagonal: Tensor, index: int) -> None:
    beta = state["shampoo_beta"]
    previous = state["eigen_sqrt_inv"][index].float().square().reciprocal()
    previous = previous.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalue = beta * previous + (1 - beta) * diagonal.detach().float()
    state["eigen_sqrt_inv"][index] = (
        eigenvalue.clamp_min(1e-30)
        .rsqrt()
        .clamp(max=4000)
        .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        .contiguous()
    )


def _update_klsoap_preconditioner(gradient: Tensor, state: dict[str, Any]) -> None:
    gradient = gradient.detach().float()
    left, right = state["Q"]
    inverse_left, inverse_right = state["eigen_sqrt_inv"]
    beta = state["shampoo_beta"]
    rows, columns = gradient.shape
    right_whitened = (right.mT @ gradient.mT) * inverse_right.view(-1, 1)
    left_target = right_whitened.mT @ right_whitened / columns
    left_whitened = (left.mT @ gradient) * inverse_left.view(-1, 1)
    right_target = left_whitened.mT @ left_whitened / rows
    state["GG"][0].mul_(beta).add_(left_target, alpha=1 - beta)
    state["GG"][1].mul_(beta).add_(right_target, alpha=1 - beta)
    state["GG"] = [_symmetrize(value).contiguous() for value in state["GG"]]
    projected = left.mT @ gradient @ right
    _update_eigen_inverse(state, (projected * inverse_right.view(1, -1)).square().mean(dim=1), 0)
    _update_eigen_inverse(state, (projected * inverse_left.view(-1, 1)).square().mean(dim=0), 1)


def _refresh_klsoap_basis(state: dict[str, Any]) -> None:
    momentum = _project_from_basis(state["exp_avg"], state)
    state["Q"] = [
        torch.linalg.qr(gram.float() @ basis.float())[0].contiguous()
        for gram, basis in zip(state["GG"], state["Q"], strict=True)
    ]
    state["exp_avg"] = _project_to_basis(momentum, state).contiguous()


class Track3MatrixOptimizer(torch.optim.Optimizer):
    """Owner-sharded implementation of Muon, KL-SOAP, Adam, and Lion.

    Training gradients have already been summed across ranks. Only optimizer work is sharded;
    parameters are gathered after every update. Grouping by exact shape keeps collectives
    shape-uniform.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        algorithm: str,
        family: str,
        lr: float,
        beta1: float | None,
        beta2: float | None,
        momentum: float | None,
        shampoo_beta: float | None,
        weight_decay: float,
        epsilon: float = 1e-10,
    ) -> None:
        values = list(parameters)
        if not values or any(value.ndim != 2 for value in values):
            raise ValueError("Track-3 matrix optimizers require nonempty 2D parameters")
        if algorithm not in {"muon", "klsoap", "adam", "lion"}:
            raise ValueError(f"unsupported native matrix algorithm: {algorithm}")
        if family not in {"W", "H"}:
            raise ValueError(f"unknown wrapper family: {family}")
        defaults = {
            "role": "matrix",
            "algorithm": algorithm,
            "family": family,
            "lr": lr,
            "initial_lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "momentum": momentum,
            "shampoo_beta": shampoo_beta,
            "weight_decay": weight_decay,
            "initial_weight_decay": weight_decay,
            "epsilon": epsilon,
        }
        groups = [
            {"params": [value for value in values if tuple(value.shape) == shape]}
            for shape in sorted({tuple(value.shape) for value in values}, reverse=True)
        ]
        super().__init__(groups, defaults)

    def _direction(
        self, gradient: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> Tensor | None:
        algorithm = group["algorithm"]
        if algorithm == "muon":
            momentum = state.setdefault("momentum_buffer", torch.zeros_like(gradient))
            momentum.lerp_(gradient, 1 - group["momentum"])
            source = gradient.lerp(momentum, group["momentum"])
            update = zeropower_via_newton_schulz(source)
            update = update * math.sqrt(max(1.0, gradient.size(-2) / gradient.size(-1)))
        elif algorithm == "klsoap":
            if not state:
                _initialize_klsoap(state, gradient, group["shampoo_beta"])
                return None
            state["step"] += 1
            projected = _project_to_basis(gradient, state)
            state["exp_avg"].mul_(group["beta1"]).add_(projected, alpha=1 - group["beta1"])
            state["exp_avg_sq"].mul_(group["beta2"]).addcmul_(
                projected, projected, value=1 - group["beta2"]
            )
            normalized = state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1e-8)
            update = _project_from_basis(normalized, state)
            _update_klsoap_preconditioner(gradient, state)
            _refresh_klsoap_basis(state)
            return update
        elif algorithm == "adam":
            step = state.setdefault("step", 0) + 1
            state["step"] = step
            momentum = state.setdefault("exp_avg", torch.zeros_like(gradient))
            variance = state.setdefault("exp_avg_sq", torch.zeros_like(gradient))
            momentum.mul_(group["beta1"]).add_(gradient, alpha=1 - group["beta1"])
            variance.mul_(group["beta2"]).addcmul_(gradient, gradient, value=1 - group["beta2"])
            source = momentum / (1 - group["beta1"] ** step)
            variance_hat = variance / (1 - group["beta2"] ** step)
            update = source / (variance_hat.sqrt() + group["epsilon"])
        else:
            momentum = state.setdefault("exp_avg", torch.zeros_like(gradient))
            source = momentum.mul(group["beta1"]).add(gradient, alpha=1 - group["beta1"])
            update = source.sign()
            momentum.mul_(group["beta2"]).add_(gradient, alpha=1 - group["beta2"])
        return update

    @torch.no_grad()
    def step(self, closure=None):
        rank, world = _rank_world()
        for group in self.param_groups:
            parameters = group["params"]
            for base in range(0, len(parameters), world):
                owned_index = base + rank
                if owned_index < len(parameters):
                    parameter = parameters[owned_index]
                    if parameter.grad is None:
                        raise RuntimeError("matrix optimizer requires every gradient")
                    update = self._direction(parameter.grad, self.state[parameter], group)
                    if update is not None:
                        if group["family"] == "H":
                            hyperball_update_(parameter, update, group["lr"])
                        else:
                            wd_update_(
                                parameter,
                                update,
                                group["lr"],
                                group["weight_decay"],
                            )
                if world > 1:
                    count = min(world, len(parameters) - base)
                    outputs = list(parameters[base : base + count])
                    outputs.extend(torch.empty_like(parameters[base]) for _ in range(world - count))
                    source = (
                        parameters[owned_index] if owned_index < len(parameters) else outputs[rank]
                    )
                    dist.all_gather(outputs, source)


class HyperBallOptimizer:
    """Take a library optimizer's unit-LR raw direction, then apply HyperBall."""

    def __init__(
        self, delegate: torch.optim.Optimizer, parameters: Iterable[Tensor], lr: float
    ) -> None:
        self.delegate = delegate
        self.parameters = tuple(parameters)
        self.param_groups = [
            {
                "params": list(self.parameters),
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": 0.0,
                "initial_weight_decay": 0.0,
                "role": "matrix",
            }
        ]

    @property
    def state(self):
        return self.delegate.state

    @torch.no_grad()
    def step(self, closure=None):
        previous = [parameter.detach().float().clone() for parameter in self.parameters]
        self.delegate.step(closure=closure)
        learning_rate = self.param_groups[0]["lr"]
        for parameter, before in zip(self.parameters, previous, strict=True):
            direction = before - parameter.float()
            parameter.copy_(before.to(parameter.dtype))
            hyperball_update_(parameter, direction, learning_rate)

    def zero_grad(self, *args, **kwargs):
        return self.delegate.zero_grad(*args, **kwargs)

    def state_dict(self):
        return self.delegate.state_dict()

    def load_state_dict(self, value):
        return self.delegate.load_state_dict(value)


def _distributed_shampoo(parameters: list[Tensor], recipe: Recipe, *, unit_lr: bool):
    try:
        from distributed_shampoo import (
            AdamPreconditionerConfig,
            DDPDistributedConfig,
            DistributedShampoo,
            EigenConfig,
            PseudoInverseConfig,
            RootInvShampooPreconditionerConfig,
            SingleDeviceDistributedConfig,
            WeightDecayType,
        )
    except ImportError as error:  # pragma: no cover - exercised in staged CUDA runtime
        raise RuntimeError(
            "Shampoo cases require the pinned distributed_shampoo runtime"
        ) from error
    _, world = _rank_world()
    distributed_config = (
        DDPDistributedConfig(
            communication_dtype=torch.float32,
            num_trainers_per_group=world,
            communicate_params=True,
        )
        if world > 1
        else SingleDeviceDistributedConfig()
    )
    assert recipe.matrix_beta1 is not None and recipe.matrix_beta2 is not None
    return DistributedShampoo(
        parameters,
        lr=1.0 if unit_lr else recipe.matrix_lr,
        betas=(recipe.matrix_beta1, recipe.matrix_beta2),
        epsilon=0.0,
        weight_decay=0.0 if unit_lr else recipe.matrix_wd_peak,
        weight_decay_type=WeightDecayType.DECOUPLED,
        max_preconditioner_dim=8192,
        precondition_frequency=1,
        start_preconditioning_step=-1,
        preconditioner_config=RootInvShampooPreconditionerConfig(
            amortized_computation_config=EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig()
            )
        ),
        grafting_config=AdamPreconditionerConfig(beta2=recipe.matrix_beta2, epsilon=1e-15),
        distributed_config=distributed_config,
    )


def setup_optimizers(model: GPT, recipe: Recipe) -> list[Any]:
    groups = parameter_groups(model)
    auxiliary = torch.optim.AdamW(
        [
            {"params": groups["embedding"], "lr": recipe.aux_lr_emb},
            {"params": groups["head"], "lr": recipe.aux_lr_proj},
            {"params": groups["misc"], "lr": recipe.aux_lr_misc},
        ],
        betas=(recipe.aux_beta1, recipe.aux_beta2),
        eps=1e-10,
        weight_decay=recipe.aux_wd_peak,
        fused=torch.cuda.is_available(),
    )
    for group in auxiliary.param_groups:
        group["role"] = "auxiliary"
        group["initial_lr"] = group["lr"]
        group["initial_weight_decay"] = group["weight_decay"]
    if recipe.optimizer == "shampoo":
        delegate = _distributed_shampoo(groups["matrix"], recipe, unit_lr=recipe.family == "H")
        if recipe.family == "H":
            matrix: Any = HyperBallOptimizer(delegate, groups["matrix"], recipe.matrix_lr)
        else:
            matrix = delegate
            for group in matrix.param_groups:
                group["role"] = "matrix"
                group["initial_lr"] = group["lr"]
                group["initial_weight_decay"] = group["weight_decay"]
    else:
        matrix = Track3MatrixOptimizer(
            groups["matrix"],
            algorithm=recipe.optimizer,
            family=recipe.family,
            lr=recipe.matrix_lr,
            beta1=recipe.matrix_beta1,
            beta2=recipe.matrix_beta2,
            momentum=recipe.matrix_mu,
            shampoo_beta=recipe.shampoo_beta,
            weight_decay=recipe.matrix_wd_peak,
        )
    return [auxiliary, matrix]


def set_scheduled_hparams(
    optimizers: Iterable[Any],
    *,
    step: int,
    train_steps: int,
    auxiliary_cooldown: float,
    schedule_profile: dict[str, Any],
) -> None:
    progress = step / train_steps
    if not 0 <= progress < 1:
        raise ValueError("schedule step must lie within training horizon")

    matrix_cooldown = (
        auxiliary_cooldown
        if schedule_profile["matrix_cooldown"] == "auxiliary"
        else 1.0
    )
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            cooldown = auxiliary_cooldown if group["role"] == "auxiliary" else matrix_cooldown
            cooldown_scale = (
                1.0
                if cooldown == 0 or progress < 1 - cooldown
                else (1 - progress) / cooldown
            )
            role = "auxiliary" if group["role"] == "auxiliary" else "matrix"
            warmup_steps = int(schedule_profile[f"{role}_warmup_steps"])
            warmup_mode = schedule_profile[f"{role}_warmup_mode"]
            if warmup_mode == "none":
                eta = cooldown_scale
            elif warmup_mode == "multiplicative":
                warmup_scale = min(1.0, (step + 1) / warmup_steps)
                eta = warmup_scale * cooldown_scale
            elif warmup_mode == "exclusive":
                eta = step / warmup_steps if step < warmup_steps else cooldown_scale
            else:  # pragma: no cover - validated when the protocol is loaded
                raise ValueError(f"unknown warmup mode: {warmup_mode}")
            group["lr"] = group["initial_lr"] * eta
            group["weight_decay"] = group.get("initial_weight_decay", 0.0)
