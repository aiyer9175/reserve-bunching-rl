"""Fixed (non-learning) bidders: A0 equilibrium, A1 heuristic, truthful."""
from __future__ import annotations

import numpy as np

from .. import benchmarks
from ..distributions import ValueDistribution


class Bidder:
    """A bidding rule. bid() maps arrays (v, r) of equal shape to bids; NaN = no bid."""

    kind = "base"
    learned = False

    def bid(self, v, r, rng: np.random.Generator, mode: str = "deterministic") -> np.ndarray:
        raise NotImplementedError


class EquilibriumBidder(Bidder):
    """A0: Bayes-Nash equilibrium bid b*(v; r)."""

    kind = "A0"

    def __init__(self, n_bidders: int, dist: ValueDistribution):
        self.n_bidders, self.dist = n_bidders, dist

    def bid(self, v, r, rng=None, mode="deterministic"):
        if benchmarks._is_std_uniform(self.dist):
            return benchmarks.b_star_uniform(v, r, self.n_bidders)
        return benchmarks.b_star_numeric(v, r, self.n_bidders, self.dist)


class HeuristicBidder(Bidder):
    """A1: censoring heuristic max(b~(v), r)."""

    kind = "A1"

    def __init__(self, n_bidders: int, dist: ValueDistribution):
        self.n_bidders, self.dist = n_bidders, dist

    def bid(self, v, r, rng=None, mode="deterministic"):
        return benchmarks.heuristic_bid(v, r, self.n_bidders, self.dist)


class TruthfulBidder(Bidder):
    """Bids its value (the dominant strategy in a second-price auction)."""

    kind = "truthful"

    def bid(self, v, r, rng=None, mode="deterministic"):
        return np.asarray(v, dtype=float).copy()


def make_fixed_bidder(kind: str, n_bidders: int, dist: ValueDistribution) -> Bidder:
    if kind == "A0":
        return EquilibriumBidder(n_bidders, dist)
    if kind == "A1":
        return HeuristicBidder(n_bidders, dist)
    if kind == "truthful":
        return TruthfulBidder()
    raise ValueError(f"unknown fixed bidder {kind!r}")
