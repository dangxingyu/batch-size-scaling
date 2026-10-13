"""Coordinate descent with the move criterion described in the paper appendix, as pure logic.

A Tuner holds one search (one batch size).  `step(lookup)` advances it as far as the finished
runs allow and returns the runs it still needs, each as (values, seed, priority).  `lookup`
maps (values, seed) to a loss, to FAILED (counted as +inf) or to None (not finished).

Move rule: at the current centre, sweep one coordinate over its candidates (one seed).  If the
best candidate beats the centre median by more than `delta`, repeat it: two seeds in total
when the gain is at least `large`, three otherwise; a two-seed repeat whose median gain lands
strictly between `delta` and `large` gets the third seed as well.  The move is accepted when
the median over those seeds still beats the centre median by more than `delta`.  Every centre
is evaluated on `center_seeds` seeds (default 2).  The search ends after `len(coords)`
consecutive coordinates without a move, so the final centre's immediate neighbours (and the
next ones out) have all been checked at it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

FAILED = "failed"
FACTORS = (0.5, 1 / math.sqrt(2), math.sqrt(2), 2.0)


@dataclass(frozen=True)
class Coord:
    name: str
    kind: str                      # "mult" | "refresh" | "grid"
    keys: tuple[str, ...]          # value keys moved together (tied auxiliary rates)
    grid: tuple[float, ...] = ()

    def candidates(self, center: dict) -> list[dict]:
        out = []
        if self.kind == "mult":
            for f in FACTORS:
                out.append({**center, **{k: center[k] * f for k in self.keys}})
        elif self.kind == "refresh":         # 1 - q' = m (1 - q)
            (k,) = self.keys
            for m in FACTORS:
                q = 1 - m * (1 - center[k])
                if 0 < q < 1:
                    out.append({**center, k: q})
        elif self.kind == "grid":            # two grid points below and two above the centre
            (k,) = self.keys
            v = center[k]
            below = [g for g in self.grid if g < v - 1e-12][-2:]
            above = [g for g in self.grid if g > v + 1e-12][:2]
            out = [{**center, k: g} for g in below + above]
        else:
            raise ValueError(self.kind)
        return out


def key(values: dict) -> tuple:
    return tuple(sorted((k, float(f"{v:.10g}")) for k, v in values.items()))


@dataclass
class Tuner:
    name: str
    coords: list[Coord]
    center: dict
    seeds: tuple[int, int, int]
    delta: float
    large: float
    speculate: bool = True
    center_seeds: int = 2
    index: int = 0
    since_move: int = 0
    confirming: dict | None = None
    needed: int = 0
    done: bool = False
    events: list = field(default_factory=list)

    # -- helpers -------------------------------------------------------------------------------
    def _loss(self, lookup, values, seed):
        v = lookup(values, seed)
        return math.inf if v == FAILED else v

    def center_losses(self, lookup):
        out = []
        for s in self.seeds:
            v = self._loss(lookup, self.center, s)
            if v is not None:
                out.append(v)
        return out

    def _advance(self, moved: bool):
        self.since_move = 0 if moved else self.since_move + 1
        self.index = (self.index + 1) % len(self.coords)
        self.confirming, self.needed = None, 0
        if self.since_move >= len(self.coords):
            self.done = True

    # -- one call ------------------------------------------------------------------------------
    def step(self, lookup) -> list[tuple[dict, int, int]]:
        for _ in range(4 * len(self.coords) + 8):          # advance while decisions are possible
            if self.done:
                return []
            requests, progressed = self._once(lookup)
            if not progressed:
                return requests
        return requests

    def _once(self, lookup):
        requests = []
        c_seeds = list(self.seeds[:self.center_seeds])
        for s in c_seeds:
            if lookup(self.center, s) is None:
                requests.append((self.center, s, 0))
        coord = self.coords[self.index]
        cands = coord.candidates(self.center)
        for cand in cands:
            if lookup(cand, self.seeds[0]) is None:
                requests.append((cand, self.seeds[0], 1))
        if self.speculate:
            nxt = self.coords[(self.index + 1) % len(self.coords)]
            for cand in nxt.candidates(self.center):
                if lookup(cand, self.seeds[0]) is None:
                    requests.append((cand, self.seeds[0], 2))
        centre = self.center_losses(lookup)
        if len(centre) < self.center_seeds or any(lookup(c, self.seeds[0]) is None for c in cands):
            return requests, False
        m_center = median(centre)
        if self.confirming is None:
            if not cands:
                self.events.append({"coord": coord.name, "decision": "no candidates"})
                self._advance(False)
                return requests, True
            best = min(cands, key=lambda c: self._loss(lookup, c, self.seeds[0]))
            gain = m_center - self._loss(lookup, best, self.seeds[0])
            if gain > self.delta:
                self.confirming, self.needed = best, (2 if gain >= self.large else 3)
                self.events.append({"coord": coord.name, "decision": "confirm", "gain": gain,
                                    "values": best, "seeds": self.needed})
                return requests, True
            self.events.append({"coord": coord.name, "decision": "keep", "best_gain": gain})
            self._advance(False)
            return requests, True
        seeds = self.seeds[:self.needed]
        missing = [s for s in seeds if lookup(self.confirming, s) is None]
        if missing:
            return [(self.confirming, s, 0) for s in missing] + requests, False
        m = median(self._loss(lookup, self.confirming, s) for s in seeds)
        gain = m_center - m
        if gain > self.delta and gain < self.large and self.needed == 2:
            self.needed = 3
            return requests, True
        if gain > self.delta:
            self.events.append({"coord": coord.name, "decision": "move", "gain": gain,
                                "values": self.confirming})
            self.center = dict(self.confirming)
            self._advance(True)
        else:
            self.events.append({"coord": coord.name, "decision": "reject", "gain": gain})
            self._advance(False)
        return requests, True
