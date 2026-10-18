"""Analytic benchmarks: Bayes-Nash equilibrium bid (A0) and censoring heuristic (A1)."""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .distributions import Uniform, ValueDistribution

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _is_std_uniform(dist) -> bool:
    return dist is None or (isinstance(dist, Uniform) and (dist.lo, dist.hi) == (0.0, 1.0))


def b_star_uniform(v, r, n_bidders: int):
    """Closed-form equilibrium bid for Uniform[0,1] values; NaN for v < r."""
    v = np.asarray(v, dtype=float)
    r = np.broadcast_to(np.asarray(r, dtype=float), v.shape)
    N = n_bidders
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (N - 1) / N * v + np.where(r > 0, r**N, 0.0) / (N * v ** (N - 1))
    b = np.where(v > 0, b, 0.0)
    return np.where(v >= r, b, np.nan)


def b_star_numeric(v, r, n_bidders: int, dist: ValueDistribution):
    """Riley-Samuelson: b*(v; r) = v - int_r^v [F(z)/F(v)]^(N-1) dz, by Gauss-Legendre."""
    v = np.asarray(v, dtype=float)
    r = np.broadcast_to(np.asarray(r, dtype=float), v.shape)
    lo = np.maximum(r, dist.lo)
    half = (v - lo) / 2.0
    z = lo[..., None] + half[..., None] * (_GL_X + 1.0)
    Fv = dist.cdf(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Fv[..., None] > 0, dist.cdf(z) / Fv[..., None], 0.0)
    integral = half * (ratio ** (n_bidders - 1) * _GL_W).sum(axis=-1)
    b = v - np.where(v > lo, integral, 0.0)
    return np.where(v >= r, b, np.nan)


def candidate_bid(v, n_bidders: int, dist: ValueDistribution | None = None):
    """Heuristic's reserve-blind candidate bid b~(v): the no-reserve equilibrium bid."""
    if _is_std_uniform(dist):
        return (n_bidders - 1) / n_bidders * np.asarray(v, dtype=float)
    return b_star_numeric(v, dist.lo, n_bidders, dist)


def heuristic_bid(v, r, n_bidders: int, dist: ValueDistribution | None = None):
    """Censoring heuristic: no bid if v < r, else max(b~(v), r)."""
    v = np.asarray(v, dtype=float)
    r = np.broadcast_to(np.asarray(r, dtype=float), v.shape)
    return np.where(v >= r, np.maximum(candidate_bid(v, n_bidders, dist), r), np.nan)


def v_tilde(r: float, n_bidders: int, dist: ValueDistribution | None = None) -> float:
    """Value above which the heuristic stops bunching: b~(v~) = r."""
    if _is_std_uniform(dist):
        return r * n_bidders / (n_bidders - 1)
    f = lambda v: float(candidate_bid(np.array(v), n_bidders, dist)) - r
    return optimize.brentq(f, r, dist.hi) if f(dist.hi) > 0 else dist.hi


def expected_revenue(bid_fn, r: float, n_bidders: int, dist: ValueDistribution) -> float:
    """Revenue per auction when the highest-value bidder wins and pays bid_fn(v).

    Valid for any weakly increasing symmetric bid function (ties at an atom do
    not change the price). The highest value has CDF F^N.
    """
    N = n_bidders
    eps = 1e-9

    def integrand(v):
        F = float(dist.cdf(v))
        dens = (float(dist.cdf(v + eps)) - float(dist.cdf(v - eps))) / (2 * eps)
        return float(bid_fn(np.array(v))) * N * F ** (N - 1) * dens

    lo = max(r, dist.lo)
    pts = [v_tilde(r, N, dist)] if lo < v_tilde(r, N, dist) < dist.hi else None
    val, _ = integrate.quad(integrand, lo, dist.hi, points=pts, limit=200)
    return val


def equilibrium_revenue(r: float, n_bidders: int, dist: ValueDistribution) -> float:
    return expected_revenue(lambda v: b_star_numeric(v, r, n_bidders, dist), r, n_bidders, dist)


def heuristic_revenue(r: float, n_bidders: int, dist: ValueDistribution) -> float:
    return expected_revenue(lambda v: heuristic_bid(v, r, n_bidders, dist), r, n_bidders, dist)
