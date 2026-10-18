"""Bunching, atom, scaling, tail and economic metrics (SPEC Sections 8.1-8.5, 8.7).

All bid-level functions take flat arrays `bids` and `r` (the reserve each bid
faced). NaN bids are abstentions and are excluded everywhere.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

DELTAS = (0.001, 0.0025, 0.005, 0.01, 0.02)
ATOM_TOL = 1e-4
EXACT_TOL = 1e-9
# Bin edges are inclusive; this guards against float error at r + delta.
_EDGE_EPS = 1e-12


def _above(bids, r, cond_r, r_high):
    """Bids >= r_high among auctions whose reserve was cond_r."""
    sel = (r == cond_r) & ~np.isnan(bids)
    b = bids[sel]
    return b[b >= r_high - _EDGE_EPS]


def bunching_shares(bids, r, r_high, r_low, deltas=DELTAS):
    """P_high(delta), P_low(delta), two-proportion z-test and 95% CI of the difference."""
    hi = _above(bids, r, r_high, r_high)
    lo = _above(bids, r, r_low, r_high)
    out = {}
    for d in deltas:
        k_h = int(np.sum(hi <= r_high + d + _EDGE_EPS))
        k_l = int(np.sum(lo <= r_high + d + _EDGE_EPS))
        n_h, n_l = hi.size, lo.size
        p_h = k_h / n_h if n_h else np.nan
        p_l = k_l / n_l if n_l else np.nan
        pooled = (k_h + k_l) / (n_h + n_l) if n_h + n_l else np.nan
        se0 = np.sqrt(pooled * (1 - pooled) * (1 / n_h + 1 / n_l)) if n_h and n_l else np.nan
        z = (p_h - p_l) / se0 if se0 and se0 > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        se = np.sqrt(p_h * (1 - p_h) / n_h + p_l * (1 - p_l) / n_l) if n_h and n_l else np.nan
        out[d] = dict(
            P_high=p_h, P_low=p_l, k_high=k_h, n_high=n_h, k_low=k_l, n_low=n_l,
            diff=p_h - p_l, ci_low=p_h - p_l - 1.96 * se, ci_high=p_h - p_l + 1.96 * se,
            z=z, p=p,
        )
    return out


def atom_share(bids, r, r_high, tol=ATOM_TOL):
    """Pr(|bid - r_high| < tol | reserve = r_high, bid >= r_high)."""
    hi = _above(bids, r, r_high, r_high)
    return float(np.mean(np.abs(hi - r_high) < tol)) if hi.size else np.nan


def atom_offset(bids, r, r_high, width=0.01):
    """Median (bid - r_high) among r_high-auction bids in [r_high, r_high + width] (H8)."""
    hi = _above(bids, r, r_high, r_high)
    near = hi[hi <= r_high + width + _EDGE_EPS]
    return float(np.median(near - r_high)) if near.size else np.nan


def modal_bid(bid_fn, r, v_lo, v_hi, step=0.001):
    """Most common deterministic bid for v in [v_lo, v_hi] at reserve r (H8), and its share."""
    v = np.arange(v_lo, v_hi + step / 2, step)
    b = np.round(bid_fn(v, np.full_like(v, r)), 6)
    vals, counts = np.unique(b[~np.isnan(b)], return_counts=True)
    i = counts.argmax()
    return dict(modal_bid=float(vals[i]), share=float(counts[i] / counts.sum()))


def kappa_fit(bids, r, r_high, deltas=DELTAS, n_boot=1000, rng=None):
    """Slope of log P_high(delta) on log delta, with a multinomial bootstrap 95% CI.

    The conditioning sample is split into disjoint bins [r, r+d1], (r+d1, r+d2], ...,
    (r+dK, inf). Resampling bin counts is equivalent to resampling bids.
    """
    rng = rng or np.random.default_rng(0)
    hi = _above(bids, r, r_high, r_high)
    n = hi.size
    if n == 0:
        return dict(kappa=np.nan, ci_low=np.nan, ci_high=np.nan, n=0)
    edges = np.array([r_high + d + _EDGE_EPS for d in deltas])
    cum = np.searchsorted(np.sort(hi), edges, side="right")
    counts = np.diff(np.concatenate([[0], cum, [n]]))
    logd = np.log(np.asarray(deltas))

    def slope(cum_counts):
        # OLS slope per row, dropping deltas with zero count (log 0).
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(cum_counts / n)
            ok = np.isfinite(y)
            x = np.where(ok, logd, np.nan)
            x = x - np.nanmean(x, axis=-1, keepdims=True)
            y = np.where(ok, y, np.nan)
            y = y - np.nanmean(y, axis=-1, keepdims=True)
            s = np.nansum(x * y, axis=-1) / np.nansum(x * x, axis=-1)
        return np.where(ok.sum(axis=-1) >= 2, s, np.nan)

    k = float(slope(cum.astype(float)))
    boot = rng.multinomial(n, counts / n, size=n_boot)
    boot_cum = np.cumsum(boot, axis=1)[:, : len(deltas)].astype(float)
    ks = slope(boot_cum)
    lo, hi_ = np.nanpercentile(ks, [2.5, 97.5])
    return dict(kappa=k, ci_low=float(lo), ci_high=float(hi_), n=int(n))


def tail_ks(bids, r, r_high, r_low, margin=0.01, sub_n=5000, n_sub=200, rng=None, n_match=None):
    """Two-sample KS on bids > r_high + margin, high vs low reserve.

    Also reports the rejection rate (alpha = 0.05) over n_sub subsamples of
    sub_n per group. If n_match is given, also reports one KS on a subsample of
    n_match per group (used to compare with SPEC 5.3's n ~ 65k).
    """
    rng = rng or np.random.default_rng(0)
    thr = r_high + margin + _EDGE_EPS
    X = _above(bids, r, r_high, r_high)
    Y = _above(bids, r, r_low, r_high)
    X, Y = X[X > thr], Y[Y > thr]
    out = dict(n_X=int(X.size), n_Y=int(Y.size))
    if X.size < 2 or Y.size < 2:
        return out | dict(D=np.nan, p=np.nan, sub_reject_rate=np.nan)
    res = stats.ks_2samp(X, Y)
    out |= dict(D=float(res.statistic), p=float(res.pvalue))
    m = min(sub_n, X.size, Y.size)
    rej = [
        stats.ks_2samp(rng.choice(X, m, replace=False), rng.choice(Y, m, replace=False)).pvalue < 0.05
        for _ in range(n_sub)
    ]
    out |= dict(sub_n=int(m), sub_reject_rate=float(np.mean(rej)))
    if n_match:
        m2 = min(n_match, X.size, Y.size)
        r2 = stats.ks_2samp(rng.choice(X, m2, replace=False), rng.choice(Y, m2, replace=False))
        out |= dict(match_n=int(m2), match_D=float(r2.statistic), match_p=float(r2.pvalue))
    return out


def bid_grid(r_levels, step=0.005, hi=1.0):
    """(v, r) evaluation grid: v in [r, 1] for each reserve level."""
    vs, rs = [], []
    for rl in r_levels:
        v = np.arange(rl, hi + step / 2, step)
        vs.append(v)
        rs.append(np.full_like(v, rl))
    return np.concatenate(vs), np.concatenate(rs)


def bid_function_distance(bid_fn, ref_fn, r_levels, step=0.005):
    """L2 (RMS, equal weight per reserve) and sup distance between two bid functions."""
    out = {}
    sq, sup = [], 0.0
    for rl in r_levels:
        v, rr = bid_grid([rl], step)
        d = bid_fn(v, rr) - ref_fn(v, rr)
        d = np.where(np.isnan(d), 0.0, d)
        sq.append(np.mean(d**2))
        sup = max(sup, float(np.max(np.abs(d))))
        out[f"L2_r{rl:g}"] = float(np.sqrt(np.mean(d**2)))
    out["L2"] = float(np.sqrt(np.mean(sq)))
    out["sup"] = sup
    return out


def economic_outcomes(values, r, result, r_levels=None):
    """Revenue, bidder surplus, unsold share, efficiency, tie frequency.

    Reported overall and per reserve level. Efficiency counts unsold auctions as
    inefficient; `efficiency_sold` conditions on a sale.
    """
    def summarize(mask):
        sold = result.winner[mask] >= 0
        hv = values[mask].argmax(axis=1)
        eff = sold & (result.winner[mask] == hv)
        return dict(
            n=int(mask.sum()),
            revenue=float(result.price[mask].mean()),
            surplus=float(result.utility[mask].sum(axis=1).mean()),
            unsold=float(1 - sold.mean()),
            efficiency=float(eff.mean()),
            efficiency_sold=float(eff[sold].mean()) if sold.any() else np.nan,
            tie_freq=float(result.tied[mask].mean()),
        )

    out = {"all": summarize(np.ones(len(r), dtype=bool))}
    for rl in r_levels or np.unique(r):
        m = r == rl
        if m.any():
            out[f"r{rl:g}"] = summarize(m)
    return out


def classify(L2_A0, L2_A1, atom, kappa, ks_D, ks_p, margin=0.01):
    """PREREGISTRATION Section 3."""
    if L2_A1 + margin <= L2_A0 and atom > 0.05:
        return "heuristic-like"
    if L2_A0 + margin <= L2_A1 and kappa > 0.3 and ks_p < 0.05 and ks_D > 0.02:
        return "equilibrium-like"
    return "neither"
