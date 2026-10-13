"""Muon's preconditioner: square root, symmetry, positive semidefiniteness, and agreement
with the Jacobian of the polar factor computed by autograd."""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from muon_preconditioner import MatrixPolarPreconditioner  # noqa: E402

SHAPES = [(6, 4), (4, 6), (5, 5)]


def polar(matrix):
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


@pytest.mark.parametrize("shape", SHAPES)
def test_half_squared_is_the_jacobian_and_matches_autograd(shape):
    torch.manual_seed(20260916)
    momentum = torch.randn(*shape, dtype=torch.float64)
    # keep the spectrum away from zero so the undamped Jacobian is the right comparison
    u, s, vh = torch.linalg.svd(momentum, full_matrices=False)
    momentum = (u * (s + 1.0)) @ vh
    pre = MatrixPolarPreconditioner(momentum.float(), damping=0.0)
    for _ in range(3):
        probe = torch.randn(*shape, dtype=torch.float32)
        squared = pre.half(pre.half(probe))
        assert torch.allclose(squared, pre.full(probe), rtol=2e-4, atol=1e-6)
        # the same Jacobian, independently, by differentiating the polar factor
        base = momentum.clone().requires_grad_(True)
        _, jvp = torch.autograd.functional.jvp(polar, base.detach(), probe.double(),
                                               create_graph=False)
        assert torch.allclose(pre.full(probe).double(), jvp, rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_symmetric_positive_semidefinite(shape):
    torch.manual_seed(7)
    momentum = torch.randn(*shape)
    pre = MatrixPolarPreconditioner(momentum, damping=1e-2)
    e = torch.randn(*shape)
    f = torch.randn(*shape)
    assert float((e * pre.full(f)).sum()) == pytest.approx(float((pre.full(e) * f).sum()), rel=1e-4)
    assert float((e * pre.half(f)).sum()) == pytest.approx(float((pre.half(e) * f).sum()), rel=1e-4)
    for probe in (e, f, e + f):
        assert float((probe * pre.full(probe)).sum()) >= -1e-6
        assert float((pre.half(probe) ** 2).sum()) == pytest.approx(
            float((probe * pre.full(probe)).sum()), rel=1e-4, abs=1e-8)


def test_damping_bounds_the_amplification():
    """Without damping a tiny singular value blows the map up; damping bounds it."""
    torch.manual_seed(11)
    u, _, vh = torch.linalg.svd(torch.randn(6, 4), full_matrices=False)
    momentum = (u * torch.tensor([1.0, 0.5, 0.25, 1e-6])) @ vh
    probe = torch.randn(6, 4)
    loose = MatrixPolarPreconditioner(momentum, damping=0.0)
    tight = MatrixPolarPreconditioner(momentum, damping=1e-1)
    assert float(loose.full(probe).abs().max()) > 1e3 * float(tight.full(probe).abs().max())
