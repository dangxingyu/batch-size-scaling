"""The Gauss-Newton product, checked against a densely assembled J^T H_z J on CPU.

The driver's operator never forms J, so the only way to know the three-pass construction
is right is to build the same matrix explicitly on a problem small enough to do so.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from run_directional_collapse import gauss_newton_product  # noqa: E402

TOKENS, CLASSES, WIDTH = 6, 5, 4


class Tiny(nn.Module):
    """Small enough that the Jacobian of the logits can be formed exactly."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(WIDTH, WIDTH)
        self.second = nn.Linear(WIDTH, CLASSES)

    def forward(self, value):
        hidden = torch.tanh(self.first(value))
        logits = self.second(hidden)
        cap = 15.0  # the same softcap shape the real model applies
        return cap * logits * (logits.square() + cap**2).rsqrt()


def dense_gauss_newton(model, inputs, token_count):
    """J^T H_z J assembled column by column from autograd, with H_z = diag(p) - p p^T."""
    parameters = list(model.parameters())
    sizes = [p.numel() for p in parameters]
    dimension = sum(sizes)
    jacobian = torch.zeros(TOKENS * CLASSES, dimension, dtype=torch.float64)
    logits = model(inputs)
    flat = logits.reshape(-1)
    for row in range(flat.numel()):
        grads = torch.autograd.grad(flat[row], parameters, retain_graph=True, allow_unused=True)
        pieces = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for p, g in zip(parameters, grads)]
        jacobian[row] = torch.cat(pieces).double()
    probability = torch.softmax(logits.detach(), dim=-1).double()
    curvature = torch.zeros(TOKENS * CLASSES, TOKENS * CLASSES, dtype=torch.float64)
    for t in range(TOKENS):
        p = probability[t]
        block = torch.diag(p) - torch.outer(p, p)
        curvature[t * CLASSES:(t + 1) * CLASSES, t * CLASSES:(t + 1) * CLASSES] = block
    return jacobian.T @ (curvature / token_count) @ jacobian


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(20260916)
    model = Tiny().double()
    inputs = torch.randn(TOKENS, WIDTH, dtype=torch.float64)
    return model, inputs, dense_gauss_newton(model, inputs, TOKENS)


def product(model, inputs, vector):
    parameters = list(model.parameters())
    parts, offset = [], 0
    for p in parameters:
        parts.append(vector[offset:offset + p.numel()].view_as(p))
        offset += p.numel()
    out = gauss_newton_product(model(inputs), parameters, tuple(parts), TOKENS)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                      for p, g in zip(parameters, out)])


def test_matches_the_dense_matrix_on_every_basis_vector(fixture):
    model, inputs, dense = fixture
    dimension = dense.shape[0]
    built = torch.stack([product(model, inputs, torch.eye(dimension, dtype=torch.float64)[i])
                         for i in range(dimension)], dim=1)
    assert torch.allclose(built, dense, rtol=1e-8, atol=1e-10)


def test_is_symmetric_positive_semidefinite_and_label_free(fixture):
    model, inputs, dense = fixture
    assert torch.allclose(dense, dense.T, atol=1e-12)
    assert float(torch.linalg.eigvalsh(dense).min()) >= -1e-10
    # the operator never sees labels, so it cannot depend on them
    generator = torch.Generator().manual_seed(7)
    v = torch.randn(dense.shape[0], generator=generator, dtype=torch.float64)
    assert torch.allclose(product(model, inputs, v), dense @ v, rtol=1e-8, atol=1e-10)


def test_differs_from_the_exact_hessian(fixture):
    """The dropped term is nonzero here, so the two operators must disagree."""
    model, inputs, dense = fixture
    parameters = list(model.parameters())
    targets = torch.tensor([0, 1, 2, 3, 4, 0])
    loss = nn.functional.cross_entropy(model(inputs), targets, reduction="sum") / TOKENS
    first = torch.autograd.grad(loss, parameters, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in first])
    rows = []
    for i in range(flat.numel()):
        second = torch.autograd.grad(flat[i], parameters, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                               for p, g in zip(parameters, second)]))
    exact = torch.stack(rows)
    assert not torch.allclose(exact, dense, rtol=1e-3, atol=1e-6)
    assert float(torch.linalg.eigvalsh(exact).min()) < -1e-8  # exact Hessian is indefinite
