"""make_preconditioner, exercised without a GPU.

The task-key wiring is checked here rather than on a GPU anchor: shapes, the guards, and that
the returned map really is the square root of the per-matrix operator.
"""

import math
import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from muon_preconditioner import MatrixPolarPreconditioner  # noqa: E402
from run_directional_collapse import make_preconditioner  # noqa: E402

SHAPES = {"a.weight": (6, 4), "b.weight": (4, 8), "c.weight": (5, 5)}
DIM = sum(r * c for r, c in SHAPES.values())


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


def experiment(momentum, count=10):
    layout = Layout(SHAPES)
    config = types.SimpleNamespace(algorithm="muon")
    core = types.SimpleNamespace(m=momentum, config=config, count=count)
    return types.SimpleNamespace(layout=layout, core=core)


def test_identity_when_asked_for_none():
    apply, metadata = make_preconditioner(experiment(torch.randn(DIM)), {})
    assert apply is None and metadata["preconditioner"] == "none"


def test_muon_polar_is_the_square_root_of_the_per_matrix_operator():
    torch.manual_seed(20260916)
    momentum = torch.randn(DIM)
    exp = experiment(momentum)
    apply, metadata = make_preconditioner(exp, {"preconditioner": "muon-polar",
                                                "preconditioner_damping": 1e-2})
    assert metadata["matrices"] == len(SHAPES) and metadata["singular_min"] > 0
    probe = torch.randn(DIM)
    got = apply(apply(probe))
    pieces = exp.layout.unpack(momentum)
    expected = {}
    for name, value in pieces.items():
        scale = math.sqrt(max(1.0, value.shape[0] / value.shape[1]))
        factor = MatrixPolarPreconditioner(value, damping=1e-2, scale=scale)
        expected[name] = factor.full(exp.layout.unpack(probe)[name])
    assert torch.allclose(got, exp.layout.pack(expected), rtol=2e-4, atol=1e-6)
    # symmetry survives the pack/unpack round trip
    other = torch.randn(DIM)
    assert float(torch.dot(probe, apply(other))) == pytest.approx(
        float(torch.dot(apply(probe), other)), rel=1e-4)


def test_unknown_preconditioners_are_rejected():
    for name in ("identity", "no-such-preconditioner"):
        with pytest.raises(ValueError, match="unknown preconditioner"):
            make_preconditioner(experiment(torch.randn(DIM)), {"preconditioner": name})


def test_tracked_refresh_reads_the_base_clock_channel_not_the_frozen_parent():
    """A channel engine never steps the parent core, so a refresh-time P^{1/2} has to come from
    the period-1 channel, whose momentum is exactly a k=1 core's on the same gradients.
    Reading the parent instead would keep P frozen at the anchor."""
    from directional_extension_math import BandAccumulator, RankBands
    from run_directional_collapse import base_clock_core

    from bsz_lm.branchout.optimization import CoreConfig, MatrixCore

    torch.manual_seed(0)
    layout, config = Layout(SHAPES), CoreConfig(algorithm="muon")
    parent = MatrixCore(layout, config, torch.device("cpu"))
    for _ in range(30):
        parent.direction(torch.randn(DIM))
    basis, _ = torch.linalg.qr(torch.randn(DIM, 4))
    engine = BandAccumulator(parent, RankBands(basis, [4]), {"rank_band_0": 4, "remainder": 1},
                             project_input=False, project_output=True, lr_power=0.5)
    reference = MatrixCore(layout, config, torch.device("cpu"))
    reference.load_state_dict(parent.state_dict())
    for _ in range(8):
        gradient = torch.randn(DIM)
        engine.advance(gradient, 1e-3)
        reference.direction(gradient)

    exp = types.SimpleNamespace(layout=layout, core=parent)
    core, source = base_clock_core(engine, exp)
    assert source == "remainder" and core.count == 38
    assert torch.equal(core.m, reference.m)
    task = {"preconditioner": "muon-polar"}
    live, _ = make_preconditioner(exp, task, core=core)
    expected, _ = make_preconditioner(types.SimpleNamespace(layout=layout, core=reference), task)
    frozen, _ = make_preconditioner(exp, task)
    probe = torch.randn(DIM)
    assert torch.allclose(live(probe), expected(probe), rtol=1e-6)
    assert not torch.allclose(live(probe), frozen(probe), rtol=1e-3)
    assert base_clock_core(types.SimpleNamespace(), exp) == (parent, "parent")
