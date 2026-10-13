"""Joint parameter-space projectors and matrix-free, signed Hessian Ritz pairs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Layout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Tensor]) -> Layout:
        return cls(tuple(parameters), tuple(tuple(p.shape) for p in parameters.values()))

    @property
    def sizes(self) -> list[int]:
        return [int(torch.Size(shape).numel()) for shape in self.shapes]

    @property
    def dimension(self) -> int:
        return sum(self.sizes)

    def pack(self, values: Mapping[str, Tensor]) -> Tensor:
        return torch.cat([values[n].detach().float().reshape(-1) for n in self.names])

    def unpack(self, vector: Tensor) -> dict[str, Tensor]:
        if vector.ndim != 1 or vector.numel() != self.dimension:
            raise ValueError("flat vector does not match parameter layout")
        return {
            n: part.view(shape)
            for n, shape, part in zip(
                self.names, self.shapes, vector.split(self.sizes), strict=True
            )
        }

    def metadata(self) -> dict:
        return {"names": list(self.names), "shapes": [list(s) for s in self.shapes]}


class TwoSpaces:
    """Top Ritz span and its exact orthogonal complement, NOT flat-vs-sharp truth."""

    def __init__(self, basis: Tensor):
        if basis.ndim != 2 or not 0 < basis.shape[1] < basis.shape[0]:
            raise ValueError("basis must have shape (dimension, rank), 0 < rank < dimension")
        self.basis = basis.detach().float().contiguous()
        if not torch.isfinite(self.basis).all():
            raise ValueError("nonfinite basis")
        identity = torch.eye(basis.shape[1], device=basis.device)
        self.orthogonality_error = float((self.basis.T @ self.basis - identity).abs().max())
        if self.orthogonality_error > 1e-3:
            raise ValueError(f"nonorthogonal projector basis: {self.orthogonality_error}")

    def project(self, value: Tensor, band: str) -> Tensor:
        if value.shape != (self.basis.shape[0],):
            raise ValueError("projection vector/layout mismatch")
        top = self.basis @ (self.basis.T @ value.float())
        if band == "top":
            return top
        if band == "remainder":
            return value - top
        raise ValueError(f"unknown band {band}")

    def leakage(self, value: Tensor, band: str) -> float:
        residual = value - self.project(value, band)
        return float(residual.square().sum() / value.square().sum().clamp_min(1e-30))


@torch.no_grad()
def top_ritz(
    hvp: Callable[[Tensor], Tensor],
    *,
    dimension: int,
    top_rank: int,
    iterations: int,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, dict]:
    """Full-reorthogonalized Lanczos; largest ALGEBRAIC eigenvalues, not |lambda|.

    Orthogonal random restarts handle invariant subspaces and repeated eigenvalues.
    The recurrence residual estimate is logged; independent final HVPs check each
    retained Ritz pair. Finite rank does not certify the global spectral ordering.
    """
    if not 0 < top_rank < dimension or iterations <= top_rank:
        raise ValueError("need 0 < top_rank < dimension and iterations > top_rank")
    count = min(iterations, dimension)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(dimension, device=device, generator=generator)
    q /= q.norm()
    vectors = torch.empty((dimension, count), device=device, dtype=torch.float32)
    diagonal, off_diagonal = [], []
    restarts = 0
    beta = torch.zeros((), device=device)
    previous = torch.zeros_like(q)
    for j in range(count):
        vectors[:, j].copy_(q)
        product = hvp(q).detach().float()
        if product.shape != q.shape or not torch.isfinite(product).all():
            raise FloatingPointError("invalid Hessian-vector product")
        residual = product - beta * previous
        alpha = torch.dot(q, residual)
        residual -= alpha * q
        active = vectors[:, : j + 1]
        for _ in range(2):
            residual -= active @ (active.T @ residual)
        next_beta = residual.norm()
        diagonal.append(alpha)
        if j + 1 == count:
            break
        # A near-invariant Krylov space may omit repeated leading eigenvectors.
        if float(next_beta) <= 1e-6 * max(1.0, float(product.norm())):
            candidate = torch.randn(dimension, device=device, generator=generator)
            for _ in range(2):
                candidate -= active @ (active.T @ candidate)
            norm = candidate.norm()
            if float(norm) < 1e-6:
                break
            previous, q = q, candidate / norm
            beta = torch.zeros_like(next_beta)
            restarts += 1
        else:
            previous, q = q, residual / next_beta
            beta = next_beta
        off_diagonal.append(beta)
    count = len(diagonal)
    if count < top_rank:
        raise RuntimeError("insufficient independent Ritz vectors")
    small = torch.diag(torch.stack(diagonal).double())
    if count > 1:
        off = torch.stack(off_diagonal[: count - 1]).double()
        small += torch.diag(off, 1) + torch.diag(off, -1)
    values, rotations = torch.linalg.eigh(small)
    order = torch.argsort(values, descending=True)[:top_rank]
    selected = values[order].float()
    basis = vectors[:, :count] @ rotations[:, order].float()
    del vectors
    spaces = TwoSpaces(basis)
    radius = float(values.abs().max())
    absolute_residuals = []
    relative_residuals = []
    for j in range(top_rank):
        residual = hvp(basis[:, j]) - selected[j] * basis[:, j]
        absolute = float(residual.norm())
        absolute_residuals.append(absolute)
        relative_residuals.append(absolute / max(abs(float(selected[j])), radius * 1e-4, 1e-20))
    return spaces.basis, {
        "algorithm": "fully_reorthogonalized_lanczos_with_random_restarts",
        "ordering": "largest_algebraic_signed_eigenvalues",
        "iterations": count,
        "restarts": restarts,
        "top_rank": top_rank,
        "dimension": dimension,
        "eigenvalues": selected.tolist(),
        "all_ritz_values": values.tolist(),
        "absolute_residuals": absolute_residuals,
        "relative_residuals": relative_residuals,
        "max_relative_residual": max(relative_residuals),
        "orthogonality_error": spaces.orthogonality_error,
        "global_top_ordering_certified": count == dimension,
        "remainder_is_flat": False,
    }
