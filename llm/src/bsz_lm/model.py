"""The exact 12-layer Modded NanoGPT Track-3 architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dimension))

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(value, (value.size(-1),), weight=self.gains.type_as(value))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight.type_as(value), self.bias.type_as(value))


class Rotary(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        angular = (1 / 1024) ** torch.linspace(0, 1, steps=dimension // 4)
        self.register_buffer(
            "angular_freq", torch.cat([angular, angular.new_zeros(dimension // 4)])
        )

    def forward(self, value: Tensor) -> Tensor:
        positions = torch.arange(value.size(1), dtype=torch.float32, device=value.device)
        theta = torch.outer(positions, self.angular_freq)[None, :, None, :]
        cosine, sine = theta.cos(), theta.sin()
        first, second = value.float().chunk(2, dim=-1)
        return torch.cat(
            (first * cosine + second * sine, first * (-sine) + second * cosine), dim=3
        ).type_as(value)


class CausalSelfAttention(nn.Module):
    def __init__(self, dimension: int, head_dimension: int = 128) -> None:
        super().__init__()
        self.num_heads = dimension // head_dimension
        self.head_dimension = head_dimension
        hidden = self.num_heads * head_dimension
        self.q = Linear(dimension, hidden)
        self.k = Linear(dimension, hidden)
        self.v = Linear(dimension, hidden)
        self.proj = Linear(hidden, dimension)
        self.rotary = Rotary(head_dimension)

    def forward(self, value: Tensor) -> Tensor:
        batch, tokens = value.shape[:2]
        shape = (batch, tokens, self.num_heads, self.head_dimension)
        query, key, projected_value = (
            self.q(value).view(shape),
            self.k(value).view(shape),
            self.v(value).view(shape),
        )
        query = self.rotary(F.rms_norm(query, (query.size(-1),)))
        key = self.rotary(F.rms_norm(key, (key.size(-1),)))
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            projected_value.transpose(1, 2),
            scale=0.12,
            is_causal=True,
        ).transpose(1, 2)
        return self.proj(output.contiguous().view(batch, tokens, -1))


class MLP(nn.Module):
    def __init__(self, dimension: int, multiplier: int = 4) -> None:
        super().__init__()
        hidden = multiplier * dimension
        self.fc = Linear(dimension, hidden)
        self.proj = Linear(hidden, dimension)

    def forward(self, value: Tensor) -> Tensor:
        return self.proj(self.fc(value).relu().square())


class Block(nn.Module):
    def __init__(self, dimension: int, head_dimension: int, mlp_multiplier: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(dimension, head_dimension)
        self.mlp = MLP(dimension, mlp_multiplier)
        self.norm1 = RMSNorm(dimension)
        self.norm2 = RMSNorm(dimension)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class GPT(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int = 50304,
        layers: int = 12,
        width: int = 768,
        head_dimension: int = 128,
        mlp_multiplier: int = 4,
        logit_softcap: float = 15.0,
    ) -> None:
        super().__init__()
        self.logit_softcap = logit_softcap
        self.embed = nn.Embedding(vocabulary_size, width).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(width, head_dimension, mlp_multiplier) for _ in range(layers)]
        )
        self.proj = Linear(width, vocabulary_size)
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)

    def logits(self, inputs: Tensor) -> Tensor:
        value = self.norm1(self.embed(inputs))
        for block in self.blocks:
            value = block(value)
        logits = self.proj(self.norm2(value)).float()
        cap = self.logit_softcap
        return cap * logits * (logits.square() + cap**2).rsqrt()

    def forward(self, inputs: Tensor, targets: Tensor, *, reduction: str = "sum") -> Tensor:
        logits = self.logits(inputs)
        return F.cross_entropy(
            logits.view(targets.numel(), -1), targets.view(-1), reduction=reduction
        )

    def per_sequence_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        losses = self(inputs, targets, reduction="none").view_as(targets)
        return losses.mean(dim=1)


@torch.no_grad()
def initialize_model(model: GPT, family: str) -> None:
    """Apply the W (zero output projections) or H initialization."""

    if family not in {"W", "H"}:
        raise ValueError(f"unknown matrix wrapper family: {family}")
    for name, parameter in model.named_parameters():
        value = parameter.data
        if name.endswith("weight"):
            if family == "W" and "proj" in name:
                value.zero_()
            elif "embed" in name:
                value.normal_()
            else:
                value.normal_(std=0.33**0.5 / value.size(-1) ** 0.5)
        elif name.endswith("bias"):
            value.zero_()
        elif name.endswith("gains"):
            value.fill_(1)
        else:  # pragma: no cover - architecture coverage guard
            raise AssertionError(f"uninitialized parameter: {name}")
        if family == "H":
            if name.endswith(".attn.proj.weight"):
                value.mul_(1.25)
            elif name.endswith(".mlp.proj.weight"):
                value.mul_(3.0)
            elif name.endswith(".mlp.fc.weight"):
                value.mul_(1.5)


def parameter_groups(model: GPT) -> dict[str, list[nn.Parameter]]:
    groups = {
        "matrix": [parameter for parameter in model.blocks.parameters() if parameter.ndim >= 2],
        "embedding": [model.embed.weight],
        "head": [model.proj.weight],
        "misc": [parameter for parameter in model.parameters() if parameter.ndim < 2],
    }
    flattened = [parameter for values in groups.values() for parameter in values]
    all_parameters = list(model.parameters())
    if len(flattened) != len(all_parameters) or {id(p) for p in flattened} != {
        id(p) for p in all_parameters
    }:
        raise AssertionError("optimizer groups do not exactly partition model parameters")
    return groups
