"""map_and_orthonormalize: the preconditioned basis, pushed back into parameter space.

A preconditioned eigenvector u solves P^{1/2} H P^{1/2} u = lam u, so the subspace it names in
parameter space is spanned by P^{1/2} u.  Using u directly as a projector would slow the wrong
subspace, which is the whole point of the mapping, so it is checked against a dense
construction here rather than on a GPU anchor.
"""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from directional_wide_basis import ShardedBasis, map_and_orthonormalize  # noqa: E402

DIM, WIDTH = 60, 5


@pytest.fixture
def basis():
    torch.manual_seed(20260916)
    columns = torch.linalg.qr(torch.randn(DIM, WIDTH)).Q
    return ShardedBasis(columns.contiguous(), DIM, 0, 1)


def test_spans_the_image_and_is_orthonormal(basis):
    torch.manual_seed(3)
    scale = torch.rand(DIM) + 0.25          # a diagonal P^{1/2}
    mapped, report = map_and_orthonormalize(basis, lambda v: v * scale)
    assert report["orthogonality_error"] < 1e-5
    got = mapped.gather_columns(WIDTH)
    assert torch.allclose(got.T @ got, torch.eye(WIDTH), atol=1e-5)
    # same span as the dense image of the map, and a different span from the input
    dense = (basis.gather_columns(WIDTH) * scale[:, None])
    expected = torch.linalg.qr(dense).Q
    cosines = torch.linalg.svdvals(got.T @ expected)
    assert float(cosines.min()) > 1 - 1e-5
    assert float(torch.linalg.svdvals(got.T @ basis.gather_columns(WIDTH)).min()) < 0.99


def test_projector_differs_from_the_unmapped_one(basis):
    """The point of the mapping: projecting with u is not projecting with P^{1/2}u."""
    torch.manual_seed(4)
    scale = torch.rand(DIM) + 0.25
    mapped, _ = map_and_orthonormalize(basis, lambda v: v * scale)
    probe = torch.randn(DIM)
    assert not torch.allclose(mapped.project(probe), basis.project(probe), atol=1e-3)


def test_identity_map_is_a_no_op_up_to_sign(basis):
    mapped, report = map_and_orthonormalize(basis, lambda v: v)
    a, b = mapped.gather_columns(WIDTH), basis.gather_columns(WIDTH)
    assert float(torch.linalg.svdvals(a.T @ b).min()) > 1 - 1e-5
    assert report["orthogonality_error"] < 1e-5


def test_rejects_a_rank_deficient_image(basis):
    """A singular map cannot yield an orthonormal basis; Cholesky must fail loudly."""
    killer = torch.ones(DIM)
    killer[:DIM // 2] = 0.0
    with pytest.raises(Exception):
        map_and_orthonormalize(basis, lambda v: v * 0.0 + killer * float(v.sum()) * 0)


def test_inverse_factor_gemm_matches_the_triangular_solve():
    """The Cholesky-QR step X <- X R^{-1} is a GEMM with the inverted (width x width) factor;
    it agrees with torch.linalg.solve_triangular to fp32."""
    from directional_wide_basis import _apply_inverse_factor
    torch.manual_seed(11)
    x = torch.randn(5000, 6) @ torch.diag(torch.tensor([3.0, 1.0, 0.5, 0.2, 0.1, 0.05]))
    gram = x.T @ x
    factor = torch.linalg.cholesky(gram.double()).float()
    expected = torch.linalg.solve_triangular(factor, x.T, upper=False).T
    got = _apply_inverse_factor(x.clone(), factor, rows=1024)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)
    assert torch.allclose(got.T @ got, torch.eye(6), atol=1e-3)


def test_bf16_basis_maps_in_bf16_and_stays_a_projector():
    """Stored bases are bf16; the image must come back bf16 (half the memory of fp32 at
    rank 768) and still be an orthonormal projector to bf16 precision."""
    torch.manual_seed(7)
    columns = torch.linalg.qr(torch.randn(DIM, WIDTH)).Q.to(torch.bfloat16)
    basis = ShardedBasis(columns.contiguous(), DIM, 0, 1)
    scale = torch.rand(DIM) + 0.25
    mapped, report = map_and_orthonormalize(basis, lambda v: v * scale, rows=16)
    assert mapped.shard.dtype == torch.bfloat16
    got = mapped.gather_columns(WIDTH)
    assert report["orthogonality_error"] < 2e-2
    expected = torch.linalg.qr(basis.gather_columns(WIDTH) * scale[:, None]).Q
    assert float(torch.linalg.svdvals(got.T @ expected).min()) > 0.99
