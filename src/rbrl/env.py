"""Vectorized single-item sealed-bid auctions with per-auction or per-bidder reserves.

Bids are arrays of shape (B, N). NaN means "no bid". A bid is valid if it is
>= the reserve that applies to that bidder. The highest valid bid wins; ties are
broken uniformly at random among tied highest valid bids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import ReserveSchedule, ValueDistribution

NO_BID = np.nan


@dataclass
class AuctionResult:
    valid: np.ndarray    # (B, N) bool: bid submitted and >= its reserve
    winner: np.ndarray   # (B,) int, -1 if unsold
    won: np.ndarray      # (B, N) bool
    price: np.ndarray    # (B,) float, 0 if unsold
    utility: np.ndarray  # (B, N) float
    tied: np.ndarray     # (B,) bool: winning bid tied with another valid bid

    @property
    def sold(self) -> np.ndarray:
        return self.winner >= 0


def run_auction(
    values: np.ndarray,
    bids: np.ndarray,
    reserves: np.ndarray,
    rng: np.random.Generator,
    mechanism: str = "first",
) -> AuctionResult:
    """Resolve a batch of auctions.

    values, bids: (B, N). reserves: (B,) for one reserve per auction, or (B, N)
    for per-bidder reserves. mechanism: "first" (pay own bid) or "second" (pay
    max(second-highest valid bid, winner's reserve)).
    """
    values = np.asarray(values, dtype=float)
    bids = np.asarray(bids, dtype=float)
    B, N = bids.shape
    res = np.asarray(reserves, dtype=float)
    res = np.broadcast_to(res[:, None] if res.ndim == 1 else res, (B, N))

    with np.errstate(invalid="ignore"):
        valid = bids >= res  # NaN compares False
    masked = np.where(valid, bids, -np.inf)
    top = masked.max(axis=1)
    sold = np.isfinite(top)
    is_top = valid & (masked == top[:, None])

    key = np.where(is_top, rng.random((B, N)), -1.0)
    winner = np.where(sold, key.argmax(axis=1), -1)
    tied = is_top.sum(axis=1) > 1

    rows = np.arange(B)
    w = np.maximum(winner, 0)
    won = np.zeros((B, N), dtype=bool)
    won[rows[sold], w[sold]] = True

    if mechanism == "first":
        price = np.where(sold, top, 0.0)
    elif mechanism == "second":
        others = masked.copy()
        others[rows, w] = -np.inf
        second = others.max(axis=1)
        price = np.where(sold, np.maximum(second, res[rows, w]), 0.0)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    utility = np.where(won, values - price[:, None], 0.0)
    return AuctionResult(valid, winner, won, price, utility, tied)


def make_streams(seed: int, names=("values", "reserves", "bidders", "ties")) -> dict:
    """Independent, reproducible random streams derived from one seed."""
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {n: np.random.default_rng(c) for n, c in zip(names, children)}


@dataclass
class AuctionEnv:
    n_bidders: int
    value_dist: ValueDistribution
    reserves: ReserveSchedule
    mechanism: str = "first"

    def sample_context(self, n_auctions: int, streams: dict):
        values = self.value_dist.sample((n_auctions, self.n_bidders), streams["values"])
        r = self.reserves.sample(n_auctions, streams["reserves"])
        return values, r

    def resolve(self, values, bids, r, streams: dict) -> AuctionResult:
        return run_auction(values, bids, r, streams["ties"], self.mechanism)
