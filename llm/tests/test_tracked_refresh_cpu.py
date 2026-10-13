"""A tracked preconditioned subspace must be refreshed in preconditioned coordinates.

The anchor stores eigenvectors u of P^{1/2} H P^{1/2}; the projector is orth(P^{1/2} U).  A
refresh that iterates the projector with the conjugated operator lands in the wrong coordinates
after one pass.  This checks `refresh_tracked_basis` against a dense construction and, as the
negative control, against that wrong construction; nothing is mocked.
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from directional_wide_basis import ShardedBasis, refresh_tracked_basis  # noqa: E402

DIM, WIDTH, SHIFT = 48, 4, 0.25


def _problem(seed=20260919):
    torch.manual_seed(seed)
    m = torch.randn(DIM, DIM)
    hess = m @ m.T / DIM + 0.1 * torch.eye(DIM)          # SPD "H"
    scale = torch.rand(DIM) * 3 + 0.2                    # diagonal P^{1/2}, far from identity
    conj = scale[:, None] * hess * scale[None, :]        # P^{1/2} H P^{1/2}
    values, vectors = torch.linalg.eigh(conj)
    top = vectors[:, -WIDTH:]
    warm = torch.linalg.qr(top + 0.05 * torch.randn(DIM, WIDTH)).Q  # warm start, slightly off
    return hess, scale, conj, warm


def _min_cos(a, b):
    return float(torch.linalg.svdvals(a.T @ b).min())


def test_refresh_matches_the_dense_construction():
    hess, scale, conj, warm = _problem()
    pre = ShardedBasis(warm.contiguous(), DIM, 0, 1)
    hvp = lambda v: scale * (hess @ (scale * v))  # noqa: E731
    new_pre, new_param, report = refresh_tracked_basis(
        hvp, pre, lambda v: v * scale, shift=SHIFT, audit=WIDTH)
    # reference: one block pass in preconditioned coordinates, then the image under P^{1/2}
    ref_pre = torch.linalg.qr((conj + SHIFT * torch.eye(DIM)) @ warm).Q
    ref_param = torch.linalg.qr(scale[:, None] * ref_pre).Q
    assert _min_cos(new_pre.gather_columns(WIDTH), ref_pre) > 1 - 1e-5
    assert _min_cos(new_param.gather_columns(WIDTH), ref_param) > 1 - 1e-5
    got = new_param.gather_columns(WIDTH)
    assert torch.allclose(got.T @ got, torch.eye(WIDTH), atol=1e-5)
    assert "parameter_space_mapping" in report


def test_repeated_refresh_converges_to_the_mapped_top_space_not_the_unmapped_one():
    """The failure mode this guards against compounds: iterating the projector with the conjugated
    operator converges to the top eigenvectors of P^{1/2}HP^{1/2} used *as* a parameter-space
    basis, i.e. orth(U), whereas the branch must slow orth(P^{1/2}U).  One pass barely
    separates the two when the spectral gap is small; sixty passes, as a 16-step cadence
    over a 1,024-step window implies, separate them fully."""
    hess, scale, conj, warm = _problem()
    hvp = lambda v: scale * (hess @ (scale * v))  # noqa: E731
    top = torch.linalg.eigh(conj).eigenvectors[:, -WIDTH:]
    mapped_top = torch.linalg.qr(scale[:, None] * top).Q
    pre = ShardedBasis(warm.contiguous(), DIM, 0, 1)
    wrong = torch.linalg.qr(scale[:, None] * warm).Q
    for _ in range(60):
        pre, param, _ = refresh_tracked_basis(hvp, pre, lambda v: v * scale, shift=SHIFT, audit=0)
        wrong = torch.linalg.qr((conj + SHIFT * torch.eye(DIM)) @ wrong).Q
    got = param.gather_columns(WIDTH)
    assert _min_cos(got, mapped_top) > 0.999          # the correct fixed point
    assert _min_cos(wrong, top) > 0.999               # the wrong fixed point
    # The two fixed points are different subspaces.  How different depends on the dynamic
    # range of P^{1/2}: here (0.2 to 3.2) the smallest principal cosine is about 0.98, so the
    # wrong projector misses the mapped top space by a margin the correct one never shows.
    assert _min_cos(wrong, mapped_top) < 0.99
    assert _min_cos(got, wrong) < 1 - 1e-3


def test_without_a_map_it_reduces_to_a_plain_refresh():
    hess, _, _, warm = _problem()
    pre = ShardedBasis(warm.contiguous(), DIM, 0, 1)
    hvp = lambda v: hess @ v  # noqa: E731
    new_pre, new_param, report = refresh_tracked_basis(hvp, pre, None, shift=SHIFT, audit=WIDTH)
    assert new_pre is new_param
    ref = torch.linalg.qr((hess + SHIFT * torch.eye(DIM)) @ warm).Q
    assert _min_cos(new_param.gather_columns(WIDTH), ref) > 1 - 1e-5
    assert "parameter_space_mapping" not in report


def test_a_shift_far_above_the_spectrum_freezes_the_iteration():
    """Why the refresh shift must be relative to the operator's own lam_1: with sigma >> lam_1
    one block pass is nearly the identity, so a "tracked" basis never leaves its warm start.
    An absolute shift of 0.25 would be sixteen times lam_1 of P^{1/2}HP^{1/2}."""
    hess, scale, conj, warm = _problem()
    hvp = lambda v: scale * (hess @ (scale * v))  # noqa: E731
    lam1 = float(torch.linalg.eigvalsh(conj)[-1])
    top = torch.linalg.eigh(conj).eigenvectors[:, -WIDTH:]
    frozen = ShardedBasis(warm.contiguous(), DIM, 0, 1)
    moving = ShardedBasis(warm.contiguous(), DIM, 0, 1)
    for _ in range(20):
        frozen, _, _ = refresh_tracked_basis(hvp, frozen, None, shift=16 * lam1, audit=0)
        moving, _, _ = refresh_tracked_basis(hvp, moving, None, shift=0.25 * lam1, audit=0)
    f, m = frozen.gather_columns(WIDTH), moving.gather_columns(WIDTH)
    assert _min_cos(m, top) > 0.99                     # a quarter-lam_1 shift converges
    assert _min_cos(f, top) < _min_cos(m, top)         # sixteen lam_1 lags behind it
    assert _min_cos(f, warm) > _min_cos(m, warm)       # because it barely left the warm start
