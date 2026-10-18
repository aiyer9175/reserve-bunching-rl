import numpy as np
import pytest
from scipy import stats

from rbrl.bidders.fixed import EquilibriumBidder, HeuristicBidder
from rbrl.distributions import ReserveSchedule, Uniform
from rbrl.env import AuctionEnv, run_auction
from rbrl.evaluate import simulate

NAN = np.nan


def rng():
    return np.random.default_rng(0)


def test_first_price_winner_pays_own_bid():
    v = np.array([[0.9, 0.8, 0.7]])
    b = np.array([[0.6, 0.65, 0.5]])
    res = run_auction(v, b, np.array([0.1]), rng(), "first")
    assert res.winner[0] == 1
    assert res.price[0] == 0.65
    np.testing.assert_allclose(res.utility[0], [0.0, 0.8 - 0.65, 0.0])


def test_second_price_pays_max_of_second_bid_and_reserve():
    v = np.array([[0.9, 0.8], [0.9, 0.8]])
    b = np.array([[0.9, 0.8], [0.9, 0.3]])
    res = run_auction(v, b, np.array([0.5, 0.5]), rng(), "second")
    np.testing.assert_allclose(res.price, [0.8, 0.5])  # second row: rival bid is below reserve
    np.testing.assert_allclose(res.utility[:, 0], [0.1, 0.4])


def test_bids_below_reserve_and_abstentions_are_rejected():
    v = np.array([[0.9, 0.8, 0.7], [0.4, 0.3, 0.2]])
    b = np.array([[0.49, NAN, 0.5], [0.4, 0.3, NAN]])
    res = run_auction(v, b, np.array([0.5, 0.5]), rng())
    np.testing.assert_array_equal(res.valid, [[False, False, True], [False, False, False]])
    assert res.winner.tolist() == [2, -1]
    assert res.price[1] == 0.0 and not res.won[1].any()
    np.testing.assert_allclose(res.utility[1], 0.0)


def test_bid_exactly_at_reserve_is_valid():
    res = run_auction(np.array([[0.6]]), np.array([[0.5]]), np.array([0.5]), rng())
    assert res.valid[0, 0] and res.winner[0] == 0


def test_per_bidder_reserves():
    v = np.array([[0.9, 0.9]])
    b = np.array([[0.6, 0.55]])
    res = run_auction(v, b, np.array([[0.7, 0.5]]), rng())  # bidder 0 faces a higher reserve
    assert res.winner[0] == 1


def test_tie_breaking_is_uniform():
    B = 300_000
    b = np.tile([0.5, 0.5, 0.5, 0.3], (B, 1))
    res = run_auction(np.ones((B, 4)), b, np.full(B, 0.1), rng())
    counts = np.bincount(res.winner, minlength=4)
    assert counts[3] == 0
    assert stats.chisquare(counts[:3]).pvalue > 1e-3
    assert res.tied.all()


def test_seed_reproducibility_and_independence():
    env = AuctionEnv(4, Uniform(), ReserveSchedule(p_high=0.5))
    pop = [HeuristicBidder(4, Uniform())] * 4
    a = simulate(pop, env, 50_000, seed=7)
    b = simulate(pop, env, 50_000, seed=7)
    c = simulate(pop, env, 50_000, seed=8)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert not np.array_equal(a["values"], c["values"])


def test_batching_does_not_change_results():
    env = AuctionEnv(4, Uniform(), ReserveSchedule(p_high=0.5))
    pop = [EquilibriumBidder(4, Uniform())] * 4
    a = simulate(pop, env, 30_000, seed=3, batch=30_000)
    b = simulate(pop, env, 30_000, seed=3, batch=10_000)
    # Each stream is consumed sequentially, so batch size must not change outputs.
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_reserve_schedule_mix_and_none_mode():
    r = ReserveSchedule(p_high=0.95).sample(200_000, rng())
    assert set(np.unique(r)) == {0.1, 0.5}
    assert abs(np.mean(r == 0.5) - 0.95) < 0.003
    assert (ReserveSchedule(mode="none").sample(10, rng()) == 0).all()
