"""Value distributions and reserve schedules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ValueDistribution:
    """Distribution of private values. Subclasses must be supported on [lo, hi]."""

    lo: float = 0.0
    hi: float = 1.0

    def sample(self, size, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, v):
        raise NotImplementedError

    def ppf(self, q):
        raise NotImplementedError


@dataclass(frozen=True)
class Uniform(ValueDistribution):
    lo: float = 0.0
    hi: float = 1.0

    def sample(self, size, rng):
        return rng.uniform(self.lo, self.hi, size=size)

    def cdf(self, v):
        return np.clip((np.asarray(v, dtype=float) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def ppf(self, q):
        return self.lo + np.asarray(q, dtype=float) * (self.hi - self.lo)


def make_value_dist(cfg: dict) -> ValueDistribution:
    kind = cfg.get("kind", "uniform")
    if kind == "uniform":
        return Uniform(cfg.get("lo", 0.0), cfg.get("hi", 1.0))
    # The "ad-like" distribution (factor F6) is added in M4.
    raise ValueError(f"unknown value distribution {kind!r}")


@dataclass(frozen=True)
class ReserveSchedule:
    """Per-auction reserve: r_high with probability p_high, else r_low.

    mode="none" gives r = 0 in every auction (used to train A2-r0).
    """

    r_low: float = 0.10
    r_high: float = 0.50
    p_high: float = 0.5
    mode: str = "mixed"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.mode == "none":
            return np.zeros(n)
        if self.mode != "mixed":
            raise ValueError(f"unknown reserve mode {self.mode!r}")
        return np.where(rng.random(n) < self.p_high, self.r_high, self.r_low)

    def with_p_high(self, p_high: float) -> "ReserveSchedule":
        return ReserveSchedule(self.r_low, self.r_high, p_high, self.mode)

    @property
    def levels(self) -> tuple[float, float]:
        return (self.r_low, self.r_high)
