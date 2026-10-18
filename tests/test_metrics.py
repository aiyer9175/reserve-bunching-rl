import numpy as np
import pytest

from rbrl import metrics
from rbrl.bidders.fixed import EquilibriumBidder, HeuristicBidder
from rbrl.distributions import Uniform
from rbrl.exploitability import WinCurve, exploitability

RH, RL = 0.5, 0.1


def test_bunching_shares_by_hand():
    bids = np.array([0.5, 0.505, 0.52, 0.7, 0.3, np.nan, 0.5, 0.6, 0.6, 0.6])
    r = np.array([RH] * 6 + [RL] * 4)
    s = metrics.bunching_shares(bids, r, RH, RL, deltas=(0.01,))[0.01]
    assert (s["k_high"], s["n_high"]) == (2, 4)  # 0.3 and NaN are excluded
    assert (s["k_low"], s["n_low"]) == (1, 4)
    assert s["P_high"] == 0.5 and s["P_low"] == 0.25


def test_bin_edge_is_inclusive():
    s = metrics.bunching_shares(np.array([0.51, 0.51]), np.array([RH, RL]), RH, RL, deltas=(0.01,))
    assert s[0.01]["P_high"] == 1.0 and s[0.01]["P_low"] == 1.0


def test_atom_share_tolerances():
    bids = np.array([0.5, 0.5 + 5e-5, 0.5 + 2e-4, 0.8])
    r = np.full(4, RH)
    assert metrics.atom_share(bids, r, RH, 1e-4) == 0.5
    assert metrics.atom_share(bids, r, RH, 1e-9) == 0.25


def test_kappa_recovers_slopes():
    rng = np.random.default_rng(0)
    n = 400_000
    # P(bid - r <= d) = sqrt(d / 0.25) near r  ->  kappa = 0.5
    sq = RH + 0.25 * rng.random(n) ** 2
    k = metrics.kappa_fit(sq, np.full(n, RH), RH, n_boot=200, rng=rng)
    assert k["kappa"] == pytest.approx(0.5, abs=0.03)
    # An atom of 1/3 plus a smooth part -> kappa near 0
    at = np.where(rng.random(n) < 1 / 3, RH, RH + 0.5 * rng.random(n))
    k = metrics.kappa_fit(at, np.full(n, RH), RH, n_boot=200, rng=rng)
    assert k["kappa"] < 0.05 and k["ci_low"] <= k["kappa"] <= k["ci_high"]


def test_tail_ks_same_vs_different():
    rng = np.random.default_rng(1)
    n = 40_000
    x = 0.51 + 0.49 * rng.random(n)
    r = np.repeat([RH, RL], n // 2)
    same = metrics.tail_ks(x, r, RH, RL, n_sub=50, rng=rng)
    assert same["D"] < 0.02 and same["sub_reject_rate"] < 0.2
    shifted = np.where(r == RH, x, 0.51 + 0.49 * rng.random(n) ** 1.3)
    diff = metrics.tail_ks(shifted, r, RH, RL, n_sub=50, rng=rng)
    assert diff["p"] < 1e-6 and diff["sub_reject_rate"] > 0.8


def test_distance_between_identical_functions_is_zero():
    f = lambda v, r: HeuristicBidder(4, Uniform()).bid(v, r)
    d = metrics.bid_function_distance(f, f, (RL, RH))
    assert d["L2"] == 0.0 and d["sup"] == 0.0


def test_win_curve_handles_ties():
    opp = np.full((1000, 3), 0.5)  # three opponents always tied at 0.5
    w = WinCurve(opp, 0.5)
    assert w(np.array(0.5)) == pytest.approx(0.25)
    assert w(np.array(0.5001)) == pytest.approx(1.0)
    assert w(np.array(0.4999)) == 0.0


def test_equilibrium_is_not_exploitable_but_heuristic_is():
    rng = np.random.default_rng(2)
    d = Uniform()
    a0 = EquilibriumBidder(4, d)
    e0 = exploitability(a0, [a0] * 3, d, (RL, RH), (0.5, 0.5), rng, n_draws=100_000)
    a1 = HeuristicBidder(4, d)
    e1 = exploitability(a1, [a1] * 3, d, (RL, RH), (0.5, 0.5), rng, n_draws=100_000)
    assert abs(e0["eps"]) < 5e-5  # Monte Carlo floor is ~1e-5
    # Analytic: at r_high a bidder with v in (0.5, 2/3) gains by bidding just above r:
    # win prob (2/3)^3 instead of the tie-weighted 0.2025, so eps_rhigh ~ 0.0013.
    assert e1["per_reserve"]["r0.5"]["eps"] == pytest.approx(0.0013, abs=1e-4)
    assert e1["eps"] == pytest.approx(0.00065, abs=7e-5)


def test_discrete_classification_uses_grid_atom_cutoff():
    # A rounded-equilibrium bidder: atom 0.087, kappa 0.32, KS rejects.
    assert metrics.classify(0.005, 0.025, 0.087, 0.32, 0.05, 1e-10, discrete=True) == "equilibrium-like"
    # kappa just under 0.3: the continuous rule says "neither", the discrete rule ignores kappa.
    assert metrics.classify(0.005, 0.025, 0.087, 0.25, 0.05, 1e-10) == "neither"
    assert metrics.classify(0.005, 0.025, 0.087, 0.25, 0.05, 1e-10, discrete=True) == "equilibrium-like"
    assert metrics.classify(0.025, 0.005, 0.347, 0.05, 0.001, 0.6, discrete=True) == "heuristic-like"
