"""Train PPO bidders.

    python -m rbrl.train configs/m2_br_a0_beta.yaml --seed 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import metrics
from .bidders.fixed import make_fixed_bidder
from .bidders.ppo import PPOBidder, PPOPolicy
from .bidders.wrappers import CensorWrapper, censor
from .distributions import ReserveSchedule, make_value_dist
from .env import AuctionEnv, make_streams
from .evaluate import evaluate_population, to_frame
from .exploitability import exploitability
from .runlog import load_config, run_dir, save_run

PPO_DEFAULTS = dict(
    auctions_per_update=8192, epochs=4, minibatch=1024, clip=0.2, lr=3e-4, lr_decay="linear",
    ent_coef=0.01, ent_schedule="anneal", vf_coef=0.5, max_grad_norm=0.5, updates=2000, hidden=64,
)
EVAL_DEFAULTS = dict(
    every=50, dyn_auctions=200_000, dyn_eps_draws=100_000, final_auctions=1_000_000,
    final_eps_draws=200_000, save_auctions=200_000, converge_eps=1e-4, converge_window=5,
    early_stop=True,
)
LEARNER_DEFAULTS = dict(kind="A3", obs="full", action="beta", mask_above_value=False,
                        wrapper="none", init="truthful")


class Trainer:
    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.ppo = PPO_DEFAULTS | cfg.get("ppo", {})
        self.ev = EVAL_DEFAULTS | cfg.get("eval", {})
        lc = self.lc = LEARNER_DEFAULTS | cfg.get("learner", {})

        N = cfg["n_bidders"]
        dist = make_value_dist(cfg["value_dist"])
        self.train_env = AuctionEnv(N, dist, ReserveSchedule(**cfg["reserves"]), cfg.get("mechanism", "first"))
        er = cfg.get("eval_reserves", {k: v for k, v in cfg["reserves"].items() if k != "mode"})
        self.eval_env = AuctionEnv(N, dist, ReserveSchedule(**er), cfg.get("mechanism", "first"))

        torch.manual_seed(seed)
        self.streams = make_streams(seed, names=("values", "reserves", "bidders", "ties", "policy"))
        self.policy = PPOPolicy(lc["obs"], lc["action"], self.ppo["hidden"],
                                self.eval_env.reserves.r_high, lc["mask_above_value"])
        if lc["init"] == "truthful":
            self.policy.init_near_truthful(rng=np.random.default_rng(seed + 50_000))
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.ppo["lr"])

        # Population: "learner" slots share self.policy; others are fixed benchmark bidders.
        self.slots = cfg["population"]
        self.learner_slots = [j for j, s in enumerate(self.slots) if s == "learner"]
        assert self.learner_slots and self.learner_slots[0] == 0, "slot 0 must be the focal learner"
        learner = PPOBidder(self.policy, lc["kind"])
        eval_learner = CensorWrapper(learner, lc["kind"]) if lc["wrapper"] in ("censor", "censor_train") else learner
        self.eval_pop = [eval_learner if s == "learner" else make_fixed_bidder(s, N, dist) for s in self.slots]
        self.fixed = {j: make_fixed_bidder(s, N, dist) for j, s in enumerate(self.slots) if s != "learner"}
        self.ref_fns = {k: make_fixed_bidder(k, N, dist).bid for k in ("A0", "A1")}
        self.dynamics: list[dict] = []

    # ---- data collection --------------------------------------------------
    def collect(self):
        B = self.ppo["auctions_per_update"]
        values, r = self.train_env.sample_context(B, self.streams)
        bids = np.empty_like(values)
        obs, acts, logps = [], [], []
        for j in range(values.shape[1]):
            if j in self.fixed:
                bids[:, j] = self.fixed[j].bid(values[:, j], r, self.streams["bidders"])
                continue
            raw, a, lp = self.policy.act(values[:, j], r, self.streams["policy"], "stochastic", need_logp=True)
            bids[:, j] = censor(raw, values[:, j], r) if self.lc["wrapper"] == "censor_train" else raw
            obs.append(self.policy.make_obs(values[:, j], r))
            acts.append(a)
            logps.append(lp)
        res = self.train_env.resolve(values, bids, r, self.streams)
        rew = res.utility[:, self.learner_slots].T.reshape(-1)
        act_dtype = torch.float32 if self.lc["action"] == "beta" else torch.long
        return (torch.cat(obs), torch.as_tensor(np.concatenate(acts), dtype=act_dtype),
                torch.as_tensor(np.concatenate(logps), dtype=torch.float32),
                torch.as_tensor(rew, dtype=torch.float32))

    # ---- PPO update -------------------------------------------------------
    def update(self, batch, frac_done: float):
        obs, act, logp_old, rew = batch
        p = self.ppo
        lr = p["lr"] * (1 - frac_done if p["lr_decay"] == "linear" else 1.0)
        for g in self.opt.param_groups:
            g["lr"] = lr
        ent_coef = p["ent_coef"] * (1 - frac_done if p["ent_schedule"] == "anneal" else 1.0)
        with torch.no_grad():
            adv_all = rew - self.policy.value(obs)
        n = len(rew)
        gen = torch.Generator().manual_seed(int(self.streams["policy"].integers(2**31)))
        stats = []
        for _ in range(p["epochs"]):
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, p["minibatch"]):
                mb = perm[s:s + p["minibatch"]]
                d = self.policy.dist(obs[mb])
                ratio = torch.exp(d.log_prob(act[mb]) - logp_old[mb])
                adv = adv_all[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -torch.min(ratio * adv, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * adv).mean()
                vloss = ((self.policy.value(obs[mb]) - rew[mb]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + p["vf_coef"] * vloss - ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), p["max_grad_norm"])
                self.opt.step()
                stats.append((pg.item(), vloss.item(), ent.item()))
        return np.mean(stats, axis=0), float(rew.mean())

    # ---- evaluation -------------------------------------------------------
    def evaluate(self, update: int, final: bool = False, mode: str = "deterministic"):
        ev = self.ev
        seed = 100_000 + self.seed if final else 200_000 + 10_000 * self.seed + update
        with torch.no_grad():
            m, sim = evaluate_population(
                self.eval_pop, self.eval_env, self.ref_fns, mode=mode, seed=seed,
                n_auctions=ev["final_auctions" if final else "dyn_auctions"],
                eps_focal=self.eval_pop[0], eps_opponents=self.eval_pop[1:],
                eps_draws=ev["final_eps_draws" if final else "dyn_eps_draws"],
                bid_columns=self.learner_slots, n_boot=1000 if final else 100, n_sub=200 if final else 20)
        return m, sim

    def summarize(self, update: int, m: dict, eps_stoch: float | None) -> dict:
        b = m["bids"]
        s = b["shares"]["0.01"]
        e = m["economics"]
        return dict(
            update=update, atom=b["atom"], atom_exact=b["atom_exact"],
            P_high=s["P_high"], P_low=s["P_low"], kappa=b["kappa"]["kappa"],
            ks_D=b["tail_ks"]["D"], ks_p=b["tail_ks"]["p"],
            eps_det=m["exploitability"]["eps"], eps_stoch=eps_stoch,
            revenue=e["all"]["revenue"], L2_A0=m["distance"]["A0"]["L2"], L2_A1=m["distance"]["A1"]["L2"],
            sup_A0=m["distance"]["A0"]["sup"], classification=m.get("classification"),
        )

    def eps_stochastic(self, draws: int) -> float:
        p = self.eval_env.reserves.p_high
        with torch.no_grad():
            return exploitability(self.eval_pop[0], self.eval_pop[1:], self.eval_env.value_dist,
                                  self.eval_env.reserves.levels, (1 - p, p),
                                  np.random.default_rng(300_000 + self.seed), mode="stochastic",
                                  n_draws=draws, n_values=200, n_policy_samples=32)["eps"]

    def converged(self) -> bool:
        w = self.ev["converge_window"]
        recent = self.dynamics[-w:]
        return len(recent) == w and all(d["eps_det"] < self.ev["converge_eps"] for d in recent)

    # ---- main loop --------------------------------------------------------
    def run(self, log=print, on_trained=None) -> dict:
        t0 = time.time()
        U = self.ppo["updates"]
        converged_at = None
        for u in range(U + 1):
            if u % self.ev["every"] == 0 or u == U:
                m, _ = self.evaluate(u)
                row = self.summarize(u, m, self.eps_stochastic(self.ev["dyn_eps_draws"] // 2))
                self.dynamics.append(row)
                log(f"[{time.time() - t0:6.0f}s] u={u:5d} A={row['atom']:.3f} Ph={row['P_high']:.3f} "
                    f"Pl={row['P_low']:.3f} k={row['kappa']:.2f} D={row['ks_D']:.3f} "
                    f"eps={row['eps_det']:.1e}/{row['eps_stoch']:.1e} rev={row['revenue']:.4f} "
                    f"L2(A0)={row['L2_A0']:.4f} L2(A1)={row['L2_A1']:.4f} sup(A0)={row['sup_A0']:.3f}")
                if self.converged() and converged_at is None:
                    converged_at = self.dynamics[-self.ev["converge_window"]]["update"]
                    if self.ev["early_stop"]:
                        log(f"converged (window starting at update {converged_at}); stopping")
                        break
            if u == U:
                break
            batch = self.collect()
            self.update(batch, u / U)

        if on_trained is not None:  # e.g. save the checkpoint before the (long) final evaluation
            on_trained(self)
        final = {}
        for mode in ("deterministic", "stochastic") + (("mode",) if self.lc["action"] == "beta" else ()):
            m, sim = self.evaluate(u, final=True, mode=mode)
            if mode == "deterministic":
                det_sim = sim
            final[mode] = m
        rng = np.random.default_rng(400_000 + self.seed)
        rh = self.eval_env.reserves.r_high
        final["modal_bid_h8"] = metrics.modal_bid(
            lambda v, r: self.eval_pop[0].bid(v, r, rng, "deterministic"), rh, 0.53, 0.66)
        final["eps_stoch"] = self.eps_stochastic(self.ev["final_eps_draws"])
        if self.lc["wrapper"] != "none":  # distance of the unclamped policy to b~ (A2 family)
            v = np.arange(0, 1.0001, 0.005)
            raw = PPOBidder(self.policy).bid(v, np.zeros_like(v), rng, "deterministic")
            bt = self.ref_fns["A1"](v, np.zeros_like(v))
            final["btilde_L2"] = float(np.sqrt(np.mean((raw - bt) ** 2)))
        return dict(final=final, dynamics=self.dynamics, converged_at=converged_at,
                    updates_run=u, wallclock_s=time.time() - t0), det_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    if args.updates is not None:
        cfg.setdefault("ppo", {})["updates"] = args.updates
    run_id = f"{cfg['run_id']}_s{args.seed}"
    out = run_dir(run_id)
    logf = open(out / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def save_checkpoint(t):
        (out / "checkpoints").mkdir(exist_ok=True)
        torch.save(t.policy.state_dict(), out / "checkpoints" / "policy.pt")

    tr = Trainer(cfg, args.seed)
    results, det_sim = tr.run(log, on_trained=save_checkpoint)
    save_run(out, cfg | {"seed": args.seed, "config_path": args.config}, results)
    n = tr.ev["save_auctions"]
    sample = {k: v[:n] for k, v in det_sim.items()}
    to_frame(sample, tr.eval_pop, run_id, args.seed, results["updates_run"], "deterministic").to_parquet(
        out / "eval.parquet", compression="zstd")
    f = results["final"]["deterministic"]
    log(f"final: class={f.get('classification')} A={f['bids']['atom']:.4f} "
        f"kappa={f['bids']['kappa']['kappa']:.3f} eps_det={f['exploitability']['eps']:.2e} "
        f"sup(A0)={f['distance']['A0']['sup']:.4f} modal_bid_h8={results['final']['modal_bid_h8']}")


if __name__ == "__main__":
    main()
