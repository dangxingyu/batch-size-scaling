"""Task-key wiring of the rule-robustness branches, checked without a GPU:
scaled-channel retention overrides, global-mode channel decay, and the clocks of the
random-complement mode."""

import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _part in ("src", "scripts"):
    sys.path.insert(0, str(_ROOT / _part))
from run_directional_collapse import (  # noqa: E402
    apply_matrix_hparam_overrides,
    apply_scaled_channel_overrides,
    global_channel_decay,
    wide_channel_periods,
)

from bsz_lm.branchout.optimization import CoreConfig, MatrixCore  # noqa: E402
from bsz_lm.branchout.spectral import Layout  # noqa: E402


def _core():
    layout = Layout.from_parameters({"a.weight": torch.zeros(3, 4), "b.weight": torch.zeros(2, 2)})
    return MatrixCore(layout, CoreConfig("muon"), torch.device("cpu"))


def test_scaled_override_hits_only_the_slowed_channel():
    cores = {"rank_band_0": _core(), "remainder": _core()}
    engine = types.SimpleNamespace(periods={"rank_band_0": 1, "remainder": 16}, cores=cores)
    exp = types.SimpleNamespace(core=_core())
    task = {"mode": "complement-out", "period": 16, "matrix_mu_scaled": 0.95 ** 16}
    apply_scaled_channel_overrides(engine, exp, task)
    assert cores["remainder"].config.momentum == pytest.approx(0.95 ** 16)
    assert cores["rank_band_0"].config.momentum == 0.95
    assert exp.core.config.momentum == 0.95


def test_scaled_override_covers_global_mode_and_refuses_unscaled_engines():
    exp = types.SimpleNamespace(core=_core())
    apply_scaled_channel_overrides(types.SimpleNamespace(), exp,
                                   {"mode": "global", "period": 16, "matrix_mu_scaled": 0.5})
    assert exp.core.config.momentum == 0.5
    with pytest.raises(ValueError):
        apply_scaled_channel_overrides(types.SimpleNamespace(), exp,
                                       {"mode": "global", "period": 1, "matrix_mu_scaled": 0.5})
    engine = types.SimpleNamespace(periods={"rank_band_0": 1, "remainder": 4},
                                   cores={"rank_band_0": _core(), "remainder": _core()})
    with pytest.raises(ValueError):
        apply_scaled_channel_overrides(engine, exp, {"mode": "complement-out", "period": 16,
                                                     "matrix_mu_scaled": 0.5})
    apply_scaled_channel_overrides(engine, exp, {"mode": "complement-out", "period": 16})  # no-op


def test_global_channel_decay_fires_only_on_due_steps():
    weights = torch.arange(6.0)
    engine = types.SimpleNamespace(period=16)
    assert torch.equal(global_channel_decay(weights, engine, {}, 0.025, 0.0), torch.zeros(6))
    metrics = {"global": {"effective_lr": 0.02 * 16}}          # lr -> k lr, lambda fixed
    decay = global_channel_decay(weights, engine, metrics, 0.025, 0.0)
    assert torch.allclose(decay, -(0.02 * 16 * 0.025) * weights)
    assert metrics["global"]["decay_coefficient"] == pytest.approx(0.008)
    assert metrics["global"]["effective_weight_decay"] == pytest.approx(0.025)
    metrics = {"global": {"effective_lr": 0.02}}               # lr fixed, lambda -> k lambda
    decay = global_channel_decay(weights, engine, metrics, 0.025, 1.0)
    assert torch.allclose(decay, -(0.02 * 0.025 * 16) * weights)
    assert metrics["global"]["effective_weight_decay"] == pytest.approx(0.4)


def test_wide_channel_periods():
    assert wide_channel_periods("random-complement-out", 16) == {"rank_band_0": 1, "remainder": 16}
    assert wide_channel_periods("complement-out", 16) == {"rank_band_0": 1, "remainder": 16}
    assert wide_channel_periods("random-out", 16) == {"rank_band_0": 16, "remainder": 1}
    assert wide_channel_periods("matrix-out", 4) == {"rank_band_0": 4, "remainder": 4}
    assert wide_channel_periods("top-out", 1) == {"rank_band_0": 1, "remainder": 1}


def test_branch_out_is_muon_only_and_momentum_overrides_hit_every_core():
    with pytest.raises(ValueError, match="Muon only"):
        CoreConfig("adam")
    cores = [_core(), _core()]
    apply_matrix_hparam_overrides(cores, {"matrix_mu": 0.9})
    assert [core.config.momentum for core in cores] == [0.9, 0.9]
    with pytest.raises(ValueError):
        apply_matrix_hparam_overrides(cores, {"matrix_mu": 1.0})


def test_core_state_round_trip_continues_bitwise():
    torch.manual_seed(3)
    layout = Layout.from_parameters({"a.weight": torch.zeros(6, 4), "b.weight": torch.zeros(4, 8)})
    first = MatrixCore(layout, CoreConfig("muon"), torch.device("cpu"))
    for _ in range(3):
        first.direction(torch.randn(layout.dimension))
    second = MatrixCore(layout, CoreConfig("muon"), torch.device("cpu"))
    second.load_state_dict(first.state_dict())
    gradient = torch.randn(layout.dimension)
    assert torch.equal(first.direction(gradient), second.direction(gradient))
    assert second.count == 4 and set(second.state_dict()) == {"m", "count"}
