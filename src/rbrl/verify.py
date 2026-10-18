"""M1 acceptance: reproduce the SPEC Section 5.3 table (H0) and the second-price control.

    python -m rbrl.verify configs/m1_benchmarks.yaml
"""
from __future__ import annotations

import sys

import numpy as np

from . import benchmarks
from .bidders.fixed import make_fixed_bidder
from .distributions import ReserveSchedule, make_value_dist
from .env import AuctionEnv
from .evaluate import evaluate_population, to_frame
from .runlog import load_config, run_dir, save_run

# SPEC Section 5.3 targets (N = 4, Uniform[0,1], r_low = 0.1, r_high = 0.5).
TARGETS = {
    "A0": {"atom_exact": 0.0, "atom": 0.012, "P_high_0.001": 0.038, "P_high_0.01": 0.128,
           "P_high_0.02": 0.188, "P_low_0.01": 0.040, "ks_D": 0.060, "revenue": 49 / 80, "unsold": 0.0625},
    "A1": {"atom_exact": 0.333, "atom": 0.333, "P_high_0.001": 0.336, "P_high_0.01": 0.360,
           "P_high_0.02": 0.387, "P_low_0.01": 0.040, "ks_D": 0.007, "revenue": 0.5885, "unsold": 0.0625},
}


def _check(name, est, target, se=0.0, floor=0.0015):
    tol = max(3 * se, floor)
    return dict(metric=name, estimate=est, target=target, tol=tol, ok=bool(abs(est - target) <= tol))


def check_h0(kind: str, m: dict) -> list[dict]:
    t, b = TARGETS[kind], m["bids"]
    s = b["shares"]
    n_hi = s["0.01"]["n_high"]
    rows = []
    for key, est in [("atom_exact", b["atom_exact"]), ("atom", b["atom"])]:
        rows.append(_check(key, est, t[key], np.sqrt(t[key] * (1 - t[key]) / n_hi)))
    for d in ("0.001", "0.01", "0.02"):
        p = s[d]["P_high"]
        rows.append(_check(f"P_high_{d}", p, t[f"P_high_{d}"], np.sqrt(p * (1 - p) / n_hi)))
    p = s["0.01"]["P_low"]
    rows.append(_check("P_low_0.01", p, t["P_low_0.01"], np.sqrt(p * (1 - p) / s["0.01"]["n_low"])))
    ks = b["tail_ks"]
    rows.append(_check("ks_D", ks["D"], t["ks_D"], floor=0.01))
    if kind == "A0":
        rows.append(dict(metric="ks_rejects", estimate=ks["p"], target="< 0.05", tol=None, ok=bool(ks["p"] < 0.05)))
    else:
        rows.append(dict(metric="ks_sub_reject_rate", estimate=ks["sub_reject_rate"], target="<= 0.10",
                         tol=None, ok=bool(ks["sub_reject_rate"] <= 0.10)))
    e = m["economics"]["r0.5"]
    rows.append(_check("revenue_r0.5", e["revenue"], t["revenue"], floor=0.002))
    rows.append(_check("unsold_r0.5", e["unsold"], t["unsold"], np.sqrt(0.0625 * 0.9375 / e["n"])))
    return rows


def main(cfg_path: str) -> int:
    cfg = load_config(cfg_path)
    N = cfg["n_bidders"]
    dist = make_value_dist(cfg["value_dist"])
    reserves = ReserveSchedule(**cfg["reserves"])
    env = AuctionEnv(N, dist, reserves)
    out_dir = run_dir(cfg["run_id"])
    ref_fns = {k: make_fixed_bidder(k, N, dist).bid for k in ("A0", "A1")}

    results, checks, frames = {}, {}, []
    for kind in ("A0", "A1"):
        bidder = make_fixed_bidder(kind, N, dist)
        pop = [bidder] * N
        m, sim = evaluate_population(
            pop, env, ref_fns, n_auctions=cfg["n_auctions"], seed=cfg["eval_seed"],
            eps_focal=bidder, eps_opponents=[bidder] * (N - 1),
            n_match=cfg["ks_match_n"], eps_draws=cfg["eps_draws"])
        results[kind] = m
        checks[kind] = check_h0(kind, m)
        if cfg.get("save_bids"):
            frames.append(to_frame(sim, pop, cfg["run_id"], cfg["eval_seed"], 0, "deterministic"))

    # Analytic cross-checks.
    results["analytic"] = dict(
        revenue_A0=benchmarks.equilibrium_revenue(reserves.r_high, N, dist),
        revenue_A1=benchmarks.heuristic_revenue(reserves.r_high, N, dist),
        v_tilde=benchmarks.v_tilde(reserves.r_high, N, dist),
        L2_A0_vs_A1=results["A0"]["distance"]["A1"]["L2"],
        sup_A0_vs_A1=results["A0"]["distance"]["A1"]["sup"],
    )

    # Second-price control: truthful bidders, no shading -> no bunching beyond value mass.
    spa = AuctionEnv(N, dist, reserves, mechanism="second")
    truthful = make_fixed_bidder("truthful", N, dist)
    m_spa, _ = evaluate_population([truthful] * N, spa, ref_fns, n_auctions=cfg["n_auctions"],
                                   seed=cfg["eval_seed"])
    results["SPA_truthful"] = m_spa
    s = m_spa["bids"]["shares"]["0.01"]
    # With v ~ U[0,1], Pr(v in [0.5, 0.51] | v >= 0.5) = 0.02 under either reserve.
    checks["SPA_truthful"] = [
        _check("P_high_0.01", s["P_high"], 0.02, np.sqrt(0.02 * 0.98 / s["n_high"])),
        _check("P_low_0.01", s["P_low"], 0.02, np.sqrt(0.02 * 0.98 / s["n_low"])),
        _check("atom", m_spa["bids"]["atom"], 0.0002, floor=0.0005),
        _check("revenue_r0.5", m_spa["economics"]["r0.5"]["revenue"], 49 / 80, floor=0.002),
    ]

    results["H0_checks"] = checks
    passed = all(row["ok"] for rows in checks.values() for row in rows)
    results["H0_passed"] = passed
    save_run(out_dir, cfg, results)
    if frames:
        import pandas as pd
        pd.concat(frames).to_parquet(out_dir / "eval.parquet", compression="zstd")

    for kind, rows in checks.items():
        print(f"\n== {kind} ==")
        for row in rows:
            est = row["estimate"]
            tol = "" if row["tol"] is None else f" ± {row['tol']:.4f}"
            tgt = row["target"] if isinstance(row["target"], str) else f"{row['target']:.4f}"
            print(f"  {'PASS' if row['ok'] else 'FAIL'}  {row['metric']:<20} {est:.4g}  (target {tgt}{tol})")
    print(f"\nH0 {'PASSED' if passed else 'FAILED'}  ->  {out_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/m1_benchmarks.yaml"))
