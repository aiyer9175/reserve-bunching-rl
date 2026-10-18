"""PPO actor-critic bidders: Beta shading head (primary) and categorical bid-grid head.

The problem is a one-step contextual bandit, so PPO reduces to clipped policy
gradient with a learned baseline V(obs). Sampling uses NumPy generators so that
runs are reproducible from the config seed; log-probabilities use torch.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Categorical

from .fixed import Bidder

BID_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
_X_EPS = 1e-6


def _mlp(n_in, n_out, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(y)))


class PPOPolicy(nn.Module):
    """Actor and critic (separate MLPs).

    obs:     "full"  -> [v, r, 1{r = r_high}]  (A3)
             "blind" -> [v]                    (A2, A2-r0, A3-blind)
    action:  "beta"        -> shading x ~ Beta(a, b), bid = x * v
             "categorical" -> bid ~ Categorical over BID_GRID
    """

    def __init__(self, obs: str = "full", action: str = "beta", hidden: int = 64,
                 r_high: float = 0.5, mask_above_value: bool = False):
        super().__init__()
        self.obs_kind, self.action_kind = obs, action
        self.r_high, self.mask_above_value = r_high, mask_above_value
        n_obs = 3 if obs == "full" else 1
        n_act = 2 if action == "beta" else len(BID_GRID)
        self.actor = _mlp(n_obs, n_act, hidden)
        self.critic = _mlp(n_obs, 1, hidden)
        self.register_buffer("grid", torch.tensor(BID_GRID, dtype=torch.float32))

    # ---- observations -------------------------------------------------
    def make_obs(self, v, r) -> torch.Tensor:
        v = np.asarray(v, dtype=np.float32)
        if self.obs_kind == "blind":
            return torch.from_numpy(v[:, None].copy())
        r = np.asarray(r, dtype=np.float32)
        return torch.from_numpy(np.column_stack([v, r, (np.abs(r - self.r_high) < 1e-9)]).astype(np.float32))

    # ---- action distribution -------------------------------------------
    def dist(self, obs: torch.Tensor):
        out = self.actor(obs)
        if self.action_kind == "beta":
            ab = 1.0 + nn.functional.softplus(out)
            return Beta(ab[:, 0], ab[:, 1])
        if self.mask_above_value:
            v = obs[:, :1]
            out = out.masked_fill(self.grid[None, :] > v + 1e-9, -1e9)
        return Categorical(logits=out)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    # ---- acting ----------------------------------------------------------
    @torch.no_grad()
    def act(self, v, r, rng: np.random.Generator, mode: str = "stochastic", need_logp: bool = False):
        """Returns (bids, raw actions, log-probs or None) as NumPy arrays."""
        v = np.asarray(v, dtype=float)
        obs = self.make_obs(v, r)
        d = self.dist(obs)
        if self.action_kind == "beta":
            a, b = d.concentration1.numpy().astype(float), d.concentration0.numpy().astype(float)
            if mode == "stochastic":
                x = rng.beta(a, b)
            elif mode == "deterministic":
                x = a / (a + b)  # Beta mean (PREREGISTRATION Section 1)
            elif mode == "mode":
                denom = a + b - 2
                x = np.where(denom > 1e-9, (a - 1) / np.maximum(denom, 1e-9), a / (a + b))
            else:
                raise ValueError(mode)
            bad = ~np.isfinite(x)
            if bad.any():
                i = np.flatnonzero(bad)[:5]
                raise FloatingPointError(
                    f"{bad.sum()} non-finite shading factors in mode={mode}: "
                    f"v={v[i]}, alpha={a[i]}, beta={b[i]}")
            x = np.clip(x, _X_EPS, 1 - _X_EPS)
            logp = d.log_prob(torch.as_tensor(x, dtype=torch.float32)).numpy() if need_logp else None
            return x * v, x, logp
        probs = d.probs.numpy().astype(float)
        if mode == "stochastic":
            cdf = np.cumsum(probs, axis=1)
            u = rng.random(len(v))[:, None] * cdf[:, -1:]
            idx = np.minimum((cdf < u).sum(axis=1), len(BID_GRID) - 1)
        else:  # "deterministic" and "mode" coincide: arg-max, ties to the lower bid
            idx = probs.argmax(axis=1)
        logp = d.log_prob(torch.as_tensor(idx)).numpy() if need_logp else None
        return BID_GRID[idx], idx, logp

    # ---- initialization -------------------------------------------------
    def init_near_truthful(self, shading: float = 0.9, concentration: float = 20.0,
                           rng: np.random.Generator | None = None, steps: int = 600):
        """Start at bid ~ shading * v so the learner must discover shading itself."""
        if self.action_kind == "beta":
            last = self.actor[-1]
            nn.init.normal_(last.weight, std=1e-3)
            a, b = shading * concentration, (1 - shading) * concentration
            with torch.no_grad():
                last.bias.copy_(torch.tensor([_inv_softplus(a - 1), _inv_softplus(b - 1)]))
            return
        # Categorical: behaviour-clone a discretized normal around shading * v.
        rng = rng or np.random.default_rng(0)
        opt = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
        for _ in range(steps):
            v = rng.random(2048)
            r = np.where(rng.random(2048) < 0.5, self.r_high, 0.1)
            target = np.exp(-0.5 * ((BID_GRID[None, :] - shading * v[:, None]) / 0.03) ** 2)
            if self.mask_above_value:
                target = np.where(BID_GRID[None, :] > v[:, None] + 1e-9, 0.0, target)
                target[:, 0] += 1e-12
            target = torch.as_tensor(target / target.sum(1, keepdims=True), dtype=torch.float32)
            logp = torch.log_softmax(self.dist(self.make_obs(v, r)).logits, dim=1)
            loss = -(target * logp.clamp_min(-1e4)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


class PPOBidder(Bidder):
    """Bidder interface around a (frozen or training) PPOPolicy."""

    learned = True

    def __init__(self, policy: PPOPolicy, kind: str = "A3"):
        self.policy, self.kind = policy, kind

    def bid(self, v, r, rng, mode="deterministic"):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v.copy()
        return self.policy.act(v.ravel(), np.broadcast_to(r, v.shape).ravel(), rng, mode)[0].reshape(v.shape)
