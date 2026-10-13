"""Checks of the move rule on synthetic objectives."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cd_core import FAILED, Coord, Tuner, key  # noqa: E402


def run(tuner, loss_fn, max_rounds=500):
    finished = {}
    for _ in range(max_rounds):
        reqs = tuner.step(lambda v, s: finished.get((key(v), s)))
        if tuner.done:
            return finished
        assert reqs, "a live tuner must ask for work"
        for values, seed, _prio in reqs:
            finished[(key(values), seed)] = loss_fn(values, seed)
    raise AssertionError("did not converge")


def test_finds_the_lattice_optimum_of_a_separable_bowl():
    coords = [Coord("lr", "mult", ("lr",)), Coord("wd", "grid", ("wd",), (0.0, 0.1, 0.2, 0.4, 0.8)),
              Coord("mu", "refresh", ("mu",))]
    target = {"lr": 0.08, "wd": 0.4, "mu": 1 - 0.1 * 0.25}
    def loss(v, seed):
        return (math.log2(v["lr"] / target["lr"])) ** 2 * 0.01 + abs(v["wd"] - target["wd"]) * 0.05 \
            + math.log2((1 - v["mu"]) / (1 - target["mu"])) ** 2 * 0.01
    t = Tuner("bowl", coords, {"lr": 0.01, "wd": 0.1, "mu": 0.9}, (1, 2, 3), 0.002, 0.006)
    run(t, loss)
    assert math.isclose(t.center["lr"], 0.08, rel_tol=1e-9)
    assert t.center["wd"] == 0.4
    assert math.isclose(1 - t.center["mu"], 0.025, rel_tol=1e-9)
    assert t.events[-1]["decision"] in ("keep", "reject", "no candidates")


def test_a_lucky_single_seed_is_rejected_by_its_repeats():
    coords = [Coord("lr", "mult", ("lr",))]
    def loss(v, seed):          # lr=2 looks 0.01 better on seed 1 only: noise, not signal
        return 1.0 - (0.01 if (v["lr"] == 2.0 and seed == 1) else 0.0)
    t = Tuner("noise", coords, {"lr": 1.0}, (1, 2, 3), 0.002, 0.006)
    finished = run(t, loss)
    assert t.center == {"lr": 1.0}
    assert (key({"lr": 2.0}), 2) in finished          # the repeat was actually run
    assert [e["decision"] for e in t.events][:2] == ["confirm", "reject"]


def test_small_gains_need_three_seeds_and_failures_count_as_worse():
    coords = [Coord("lr", "mult", ("lr",))]
    def loss(v, seed):
        if v["lr"] == 0.5:
            return FAILED
        return 1.0 - (0.004 if v["lr"] == 2.0 else 0.0) - (0.004 if v["lr"] == 4.0 else 0.0)
    t = Tuner("small", coords, {"lr": 1.0}, (1, 2, 3), 0.002, 0.006)
    finished = run(t, loss)
    assert t.center["lr"] in (2.0, 4.0)
    assert (key({"lr": 2.0}), 3) in finished           # small gain: third seed required


def test_one_seed_centre_asks_for_a_single_centre_run():
    coords = [Coord("lr", "mult", ("lr",))]
    t = Tuner("one", coords, {"lr": 1.0}, (1, 2, 3), 0.002, 0.006, speculate=False, center_seeds=1)
    reqs = t.step(lambda v, s: None)
    centre = [(v, s) for v, s, _ in reqs if v == {"lr": 1.0}]
    assert centre == [({"lr": 1.0}, 1)]
    finished = run(t, lambda v, s: 1.0 - (0.01 if v["lr"] == 2.0 else 0.0))
    assert t.center == {"lr": 2.0} and (key({"lr": 1.0}), 2) not in finished


def test_two_seed_confirmation_with_a_medium_gain_adds_the_third_seed():
    coords = [Coord("lr", "mult", ("lr",))]
    # lr=2 gains 0.01 on seed 1 (>= large: two seeds), 0.0005 on seed 2 (two-seed median gain
    # 0.00525, between delta and large: third seed), nothing on seed 3 (median gain 0.0005).
    gains = {1: 0.01, 2: 0.0005, 3: 0.0}
    def loss(v, seed):
        return 1.0 - (gains[seed] if v["lr"] == 2.0 else 0.0)
    t = Tuner("escalate", coords, {"lr": 1.0}, (1, 2, 3), 0.002, 0.006)
    finished = run(t, loss)
    assert (key({"lr": 2.0}), 3) in finished
    assert t.events[0]["decision"] == "confirm" and t.events[0]["seeds"] == 2
    assert t.events[1]["decision"] == "reject" and t.center == {"lr": 1.0}
