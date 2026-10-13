from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bsz_lm.optimizers import set_scheduled_hparams
from bsz_lm.protocol import load_protocol


def _optimizers(matrix_lr: float = 2.0, auxiliary_lr: float = 3.0):
    matrix = {
        "role": "matrix",
        "lr": matrix_lr,
        "initial_lr": matrix_lr,
        "weight_decay": 0.7,
        "initial_weight_decay": 0.7,
    }
    auxiliary = {
        "role": "auxiliary",
        "lr": auxiliary_lr,
        "initial_lr": auxiliary_lr,
        "weight_decay": 0.0,
        "initial_weight_decay": 0.0,
    }
    return [SimpleNamespace(param_groups=[matrix]), SimpleNamespace(param_groups=[auxiliary])]


def _profile(key: str):
    path = Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
    return load_protocol(path).value["training"]["schedule_profiles"][key]


def test_w_family_shares_the_tabulated_cooldown() -> None:
    optimizers = _optimizers()
    set_scheduled_hparams(
        optimizers,
        step=900,
        train_steps=1000,
        auxiliary_cooldown=0.2,
        schedule_profile=_profile("W/lion"),
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(1.0)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(1.5)
    assert optimizers[0].param_groups[0]["weight_decay"] == 0.7


def test_h_family_keeps_full_matrix_cooldown() -> None:
    optimizers = _optimizers()
    set_scheduled_hparams(
        optimizers,
        step=900,
        train_steps=1000,
        auxiliary_cooldown=0.2,
        schedule_profile=_profile("H/muon"),
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(0.2)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(1.5)


def test_klsoap_w_keeps_full_matrix_cooldown_and_only_cools_auxiliary() -> None:
    optimizers = _optimizers()
    set_scheduled_hparams(
        optimizers,
        step=900,
        train_steps=1000,
        auxiliary_cooldown=0.2,
        schedule_profile=_profile("W/klsoap"),
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(0.2)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(1.5)


def test_adamw_uses_multiplicative_warmup_for_both_groups() -> None:
    optimizers = _optimizers()
    set_scheduled_hparams(
        optimizers,
        step=0,
        train_steps=813,
        auxiliary_cooldown=0.6,
        schedule_profile=_profile("W/adam"),
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(2.0 / 250)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(3.0 / 250)

    set_scheduled_hparams(
        optimizers,
        step=249,
        train_steps=813,
        auxiliary_cooldown=0.6,
        schedule_profile=_profile("W/adam"),
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(2.0)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(3.0)


def test_adamh_reproduces_exclusive_matrix_warmup_boundary() -> None:
    optimizers = _optimizers()
    profile = _profile("H/adam")
    set_scheduled_hparams(
        optimizers,
        step=0,
        train_steps=813,
        auxiliary_cooldown=0.25,
        schedule_profile=profile,
    )
    assert optimizers[0].param_groups[0]["lr"] == 0.0
    assert optimizers[1].param_groups[0]["lr"] == 3.0

    set_scheduled_hparams(
        optimizers,
        step=249,
        train_steps=813,
        auxiliary_cooldown=0.25,
        schedule_profile=profile,
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(2.0 * 249 / 250)

    set_scheduled_hparams(
        optimizers,
        step=250,
        train_steps=813,
        auxiliary_cooldown=0.25,
        schedule_profile=profile,
    )
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(2.0 * (1 - 250 / 813))
    assert optimizers[1].param_groups[0]["lr"] == 3.0
