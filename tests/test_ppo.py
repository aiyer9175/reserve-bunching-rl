import numpy as np
import pytest
import torch

from rbrl.bidders.ppo import BID_GRID, PPOBidder, PPOPolicy
from rbrl.bidders.wrappers import CensorWrapper, censor
from rbrl.train import Trainer

V = np.linspace(0.05, 1.0, 200)
R = np.where(np.arange(200) % 2, 0.5, 0.1)


def test_beta_init_is_near_truthful_and_deterministic_is_mean():
    torch.manual_seed(0)
    p = PPOPolicy("full", "beta")
    p.init_near_truthful()
    bids, x, _ = p.act(V, R, np.random.default_rng(0), "deterministic")
    np.testing.assert_allclose(x, 0.9, atol=0.01)
    np.testing.assert_allclose(bids, x * V)
    xs = np.stack([p.act(V, R, np.random.default_rng(s), "stochastic")[1] for s in range(200)])
    np.testing.assert_allclose(xs.mean(0), x, atol=0.02)  # mean of samples ~ deterministic bid


def test_beta_bids_never_exceed_value():
    p = PPOPolicy("blind", "beta")
    bids = p.act(V, R, np.random.default_rng(0), "stochastic")[0]
    assert (bids <= V).all() and (bids >= 0).all()


def test_categorical_mask_and_init():
    torch.manual_seed(0)
    p = PPOPolicy("full", "categorical", mask_above_value=True)
    p.init_near_truthful()
    bids = p.act(V, R, np.random.default_rng(0), "stochastic")[0]
    assert (bids <= V + 1e-9).all()
    assert np.isin(bids, BID_GRID).all()
    det = p.act(V, R, np.random.default_rng(0), "deterministic")[0]
    assert np.abs(det - 0.9 * V).mean() < 0.02 and np.abs(det - 0.9 * V).max() < 0.06


def test_log_probs_match_distribution():
    p = PPOPolicy("full", "beta")
    _, x, lp = p.act(V, R, np.random.default_rng(1), "stochastic", need_logp=True)
    d = p.dist(p.make_obs(V, R))
    np.testing.assert_allclose(lp, d.log_prob(torch.as_tensor(x, dtype=torch.float32)).detach().numpy(), rtol=1e-5)


def test_censor_wrapper():
    v = np.array([0.4, 0.6, 0.9])
    np.testing.assert_array_equal(censor(np.array([0.3, 0.45, 0.7]), v, 0.5), [np.nan, 0.5, 0.7])
    w = CensorWrapper(PPOBidder(PPOPolicy("blind", "beta")), "A2")
    b = w.bid(v, np.full(3, 0.5), np.random.default_rng(0))
    assert np.isnan(b[0]) and (b[1:] >= 0.5).all()


def _cfg(**over):
    cfg = dict(run_id="t", n_bidders=4, value_dist={"kind": "uniform"},
               reserves={"r_low": 0.1, "r_high": 0.5, "p_high": 0.5, "mode": "mixed"},
               population=["learner", "A0", "A0", "A0"],
               ppo={"auctions_per_update": 1024, "updates": 3},
               eval={"every": 3, "dyn_auctions": 20_000, "dyn_eps_draws": 20_000,
                     "final_auctions": 20_000, "final_eps_draws": 20_000})
    return cfg | over


def test_training_is_reproducible_from_seed():
    a = Trainer(_cfg(), seed=3)
    b = Trainer(_cfg(), seed=3)
    for t in (a, b):
        for u in range(2):
            t.update(t.collect(), u / 2)
    for pa, pb in zip(a.policy.parameters(), b.policy.parameters()):
        torch.testing.assert_close(pa, pb)


def test_shared_self_play_collects_one_sample_per_learner_slot():
    t = Trainer(_cfg(population=["learner"] * 4), seed=0)
    obs, act, logp, rew = t.collect()
    assert len(rew) == 4 * 1024 and obs.shape == (4 * 1024, 3)
