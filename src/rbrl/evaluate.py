"""Simulate a frozen bidder population and compute the full metric battery."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .distributions import ReserveSchedule
from .env import AuctionEnv, make_streams
from .exploitability import exploitability


def simulate(population, env: AuctionEnv, n_auctions: int, seed: int,
             mode: str = "deterministic", batch: int = 1_000_000) -> dict:
    """Run n_auctions auctions. Returns per-auction arrays (values, bids, r, result fields)."""
    streams = make_streams(seed)
    parts = []
    for start in range(0, n_auctions, batch):
        n = min(batch, n_auctions - start)
        values, r = env.sample_context(n, streams)
        bids = np.column_stack([
            b.bid(values[:, j], r, streams["bidders"], mode) for j, b in enumerate(population)
        ])
        res = env.resolve(values, bids, r, streams)
        parts.append(dict(values=values, bids=bids, r=r, valid=res.valid, won=res.won,
                          winner=res.winner, price=res.price, utility=res.utility, tied=res.tied))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def to_frame(sim: dict, population, run_id: str, seed: int, update_step: int, mode: str) -> pd.DataFrame:
    """Long format, one row per (auction, bidder): the SPEC Section 3 schema."""
    B, N = sim["bids"].shape
    won = sim["won"]
    return pd.DataFrame({
        "run_id": run_id,
        "seed": seed,
        "update_step": update_step,
        "bidder_kind": np.tile([b.kind for b in population], B),
        "policy_mode": mode,
        "v": sim["values"].ravel().astype(np.float32),
        "r": np.repeat(sim["r"], N).astype(np.float32),
        "bid": sim["bids"].ravel().astype(np.float32),
        "valid": sim["valid"].ravel(),
        "won": won.ravel(),
        "price_paid": np.where(won, sim["price"][:, None], 0.0).ravel().astype(np.float32),
        "utility": sim["utility"].ravel().astype(np.float32),
    })


class _ResultView:
    """Adapter so economic_outcomes can read a simulate() dict."""

    def __init__(self, sim):
        self.winner, self.price, self.utility, self.tied = (
            sim["winner"], sim["price"], sim["utility"], sim["tied"])


def bid_metrics(sim: dict, reserves: ReserveSchedule, rng, n_boot=1000, n_match=None,
                columns=None, n_sub=200) -> dict:
    """Bunching, atom, kappa and tail metrics on pooled bids (or a subset of bidder columns)."""
    bids = sim["bids"] if columns is None else sim["bids"][:, columns]
    N = bids.shape[1]
    b = bids.ravel()
    r = np.repeat(sim["r"], N)
    rh, rl = reserves.r_high, reserves.r_low
    shares = metrics.bunching_shares(b, r, rh, rl)
    return dict(
        shares={f"{d:g}": v for d, v in shares.items()},
        atom=metrics.atom_share(b, r, rh, metrics.ATOM_TOL),
        atom_exact=metrics.atom_share(b, r, rh, metrics.EXACT_TOL),
        atom_offset=metrics.atom_offset(b, r, rh),
        kappa=metrics.kappa_fit(b, r, rh, n_boot=n_boot, rng=rng),
        tail_ks=metrics.tail_ks(b, r, rh, rl, rng=rng, n_match=n_match, n_sub=n_sub),
    )


def evaluate_population(population, env: AuctionEnv, ref_fns: dict, *, n_auctions=1_000_000,
                        seed=100_000, mode="deterministic", eps_focal=None, eps_opponents=None,
                        n_match=None, eps_draws=200_000, bid_columns=None, n_boot=1000,
                        n_sub=200, eval_reserves: ReserveSchedule | None = None, discrete=False):
    """Full SPEC Section 8 battery for one population.

    Bid metrics use a balanced (50/50) evaluation; economic outcomes and
    exploitability use the env's own p_high (PREREGISTRATION Section 1).
    ref_fns: {"A0": f(v, r), "A1": f(v, r)} deterministic reference bid functions.
    eps_focal / eps_opponents: bidders for the exploitability calculation (optional).
    """
    rng = np.random.default_rng(seed + 1)
    if eval_reserves is not None:
        env = AuctionEnv(env.n_bidders, env.value_dist, eval_reserves, env.mechanism)
    balanced = AuctionEnv(env.n_bidders, env.value_dist, env.reserves.with_p_high(0.5), env.mechanism)
    sim_bal = simulate(population, balanced, n_auctions, seed, mode)
    out = dict(mode=mode, n_auctions=n_auctions, bids=bid_metrics(
        sim_bal, env.reserves, rng, n_boot=n_boot, n_match=n_match, columns=bid_columns, n_sub=n_sub))

    sim_mix = sim_bal if env.reserves.p_high == 0.5 else simulate(population, env, n_auctions, seed + 7, mode)
    out["economics"] = metrics.economic_outcomes(
        sim_mix["values"], sim_mix["r"], _ResultView(sim_mix), list(env.reserves.levels))

    focal = population[0]
    det_fn = lambda v, r: focal.bid(v, r, rng, "deterministic")
    out["distance"] = {k: metrics.bid_function_distance(det_fn, f, env.reserves.levels) for k, f in ref_fns.items()}

    if eps_focal is not None:
        p = env.reserves.p_high
        out["exploitability"] = exploitability(
            eps_focal, eps_opponents, env.value_dist, env.reserves.levels, (1 - p, p), rng,
            mode=mode, n_draws=eps_draws)

    bm = out["bids"]
    if "A0" in out["distance"] and "A1" in out["distance"]:
        out["classification"] = metrics.classify(
            out["distance"]["A0"]["L2"], out["distance"]["A1"]["L2"], bm["atom"],
            bm["kappa"]["kappa"], bm["tail_ks"]["D"], bm["tail_ks"]["p"], discrete=discrete)
    return out, sim_bal
