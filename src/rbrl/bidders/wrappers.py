"""Censor wrapper for the two-stage learners (A2, A2b, A2-r0)."""
from __future__ import annotations

import numpy as np

from .fixed import Bidder


def censor(bids, v, r):
    """No bid if v < r; otherwise max(bid, r)."""
    v = np.asarray(v, dtype=float)
    r = np.broadcast_to(np.asarray(r, dtype=float), v.shape)
    return np.where(v >= r, np.maximum(bids, r), np.nan)


class CensorWrapper(Bidder):
    """Submits max(inner bid, r) when v >= r, else no bid."""

    def __init__(self, inner: Bidder, kind: str | None = None):
        self.inner = inner
        self.kind = kind or f"{inner.kind}+censor"
        self.learned = inner.learned

    def bid(self, v, r, rng, mode="deterministic"):
        return censor(self.inner.bid(v, r, rng, mode), v, r)
