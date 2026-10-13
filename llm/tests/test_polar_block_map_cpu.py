"""The batched paths of the polar preconditioner and of the basis mapping must equal the
per-matrix / per-column paths they replace (systems speedup of the tracked refresh)."""

import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from directional_wide_basis import ShardedBasis, map_and_orthonormalize  # noqa: E402
from muon_preconditioner import MatrixPolarPreconditioner  # noqa: E402
from run_directional_collapse import make_preconditioner  # noqa: E402

SHAPES = {"a.weight": (6, 4), "b.weight": (6, 4), "c.weight": (3, 5), "d.weight": (5, 5)}


def test_batched_factors_and_half_batch_match_the_single_matrix_path():
    torch.manual_seed(20260922)
    momenta = {name: torch.randn(*shape) for name, shape in SHAPES.items()}
    scales = {name: 1.0 + i for i, name in enumerate(SHAPES)}
    batched = MatrixPolarPreconditioner.batch(momenta, damping=1e-3, scales=scales)
    for name, value in momenta.items():
        single = MatrixPolarPreconditioner(value, damping=1e-3, scale=scales[name])
        probes = torch.randn(3, *value.shape)
        expected = torch.stack([single.half(e) for e in probes])
        assert torch.allclose(batched[name].half_batch(probes), expected, atol=1e-5, rtol=1e-5)
        assert torch.allclose(batched[name].half(probes[0]), expected[0], atol=1e-5, rtol=1e-5)


def test_eigh_factorization_matches_the_svd_path():
    torch.manual_seed(11)
    momenta = {name: torch.randn(*shape) for name, shape in SHAPES.items()}
    # condition number ~300: the Gram route squares it, fp64 eigh absorbs that; beyond ~1e3 the
    # smallest-singular-value directions lose digits (measured 8e-3 at cond 2e3), so the eigh
    # factorization is an opt-in for well-conditioned momenta, not the default.
    ill = torch.randn(6, 4) @ torch.diag(torch.tensor([1.0, 1e-1, 3e-2, 1e-2])) @ torch.linalg.qr(torch.randn(4, 4)).Q
    momenta["ill.weight"] = ill
    by_svd = MatrixPolarPreconditioner.batch(momenta, damping=1e-3)
    by_eigh = MatrixPolarPreconditioner.batch(momenta, damping=1e-3, method="eigh")
    for name, value in momenta.items():
        probes = torch.randn(2, *value.shape)
        a, b = by_svd[name].half_batch(probes), by_eigh[name].half_batch(probes)
        assert torch.allclose(a, b, atol=2e-4, rtol=2e-4), name
        assert torch.allclose(by_svd[name].s, by_eigh[name].s, atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError):
        MatrixPolarPreconditioner.batch(momenta, method="qr")


class Layout:
    def __init__(self, shapes):
        self.shapes = dict(shapes)
        self.names = list(shapes)
        self.dimension = sum(r * c for r, c in shapes.values())

    def unpack(self, vector):
        out, offset = {}, 0
        for name, shape in self.shapes.items():
            count = shape[0] * shape[1]
            out[name] = vector[offset:offset + count].view(*shape)
            offset += count
        return out

    def pack(self, values):
        return torch.cat([values[name].reshape(-1) for name in self.shapes])


def _experiment():
    layout = Layout(SHAPES)
    torch.manual_seed(7)
    core = types.SimpleNamespace(m=torch.randn(layout.dimension),
                                 config=types.SimpleNamespace(algorithm="muon"), count=10)
    return types.SimpleNamespace(layout=layout, core=core)


def test_block_apply_equals_column_by_column():
    exp = _experiment()
    apply, metadata = make_preconditioner(exp, {"preconditioner": "muon-polar", "preconditioner_damping": 1e-3})
    assert metadata["matrices"] == len(SHAPES)
    matrix = torch.randn(exp.layout.dimension, 5)
    columns = torch.stack([apply(matrix[:, j]) for j in range(5)], dim=1)
    assert torch.allclose(apply.block(matrix), columns, atol=1e-5, rtol=1e-5)


def test_map_and_orthonormalize_block_path_matches_column_path():
    torch.manual_seed(3)
    dim, width = 96, 6
    basis = ShardedBasis(torch.linalg.qr(torch.randn(dim, width)).Q.contiguous(), dim, 0, 1)
    weights = torch.randn(dim, dim)

    def apply_map(vector):
        return weights @ vector

    apply_map.block = lambda matrix: weights @ matrix
    blocked, report = map_and_orthonormalize(basis, apply_map, block=4)
    plain, _ = map_and_orthonormalize(basis, apply_map, block=1)
    assert torch.allclose(blocked.shard.float(), plain.shard.float(), atol=1e-5, rtol=1e-5)
    assert report["orthogonality_error"] < 1e-5 and report["width"] == width
    assert report["phase_seconds"]["map"] >= 0.0 and report["matmul_precision"] == "highest"
    scoped, report2 = map_and_orthonormalize(basis, apply_map, block=4, matmul_precision="high")
    assert report2["matmul_precision"] == "high" and torch.get_float32_matmul_precision() == "highest"
    assert torch.allclose(scoped.shard.float(), plain.shard.float(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
