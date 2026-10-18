"""Per-seed tables and pre-registered decision rules.

    python -m rbrl.summarize m2
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .runlog import ROOT

EPS_CONVERGED = 1e-4  # PREREGISTRATION Amendment 1


def load_runs(prefix: str) -> dict[str, dict]:
    runs = {}
    for d in sorted((ROOT / "results").glob(f"{prefix}*_s*")):
        f = d / "metrics.json"
        if f.exists():
            runs[d.name] = json.loads(f.read_text())
    return runs


def row(name: str, m: dict) -> dict:
    det = m["final"]["deterministic"]
    b = det["bids"]
    return dict(
        run=name,
        updates=m["updates_run"], conv_at=m["converged_at"],
        eps_det=det["exploitability"]["eps"], eps_stoch=m["final"].get("eps_stoch"),
        sup_A0=det["distance"]["A0"]["sup"], L2_A0=det["distance"]["A0"]["L2"], L2_A1=det["distance"]["A1"]["L2"],
        A=b["atom"], A_exact=b["atom_exact"], P_high=b["shares"]["0.01"]["P_high"],
        P_low=b["shares"]["0.01"]["P_low"], kappa=b["kappa"]["kappa"], ks_D=b["tail_ks"]["D"],
        modal_h8=m["final"]["modal_bid_h8"]["modal_bid"], offset=b["atom_offset"],
        revenue=det["economics"]["all"]["revenue"], cls=det.get("classification"),
    )


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2e}" if v != 0 and (abs(v) < 1e-3) else f"{v:.4f}"
    return str(v)


def print_table(rows: list[dict]):
    cols = list(rows[0])
    w = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(w[c]) for c in cols))


def h1(rows):
    ok = [r["sup_A0"] < 0.02 and r["eps_det"] < EPS_CONVERGED for r in rows]
    return all(ok), f"{sum(ok)}/{len(ok)} seeds with sup < 0.02 and eps_det < 1e-4"


def h8_discrete(rows, r_high=0.5):
    ok = [abs(r["modal_h8"] - (r_high + 0.01)) < 1e-9 for r in rows]
    return sum(ok) >= 4, f"{sum(ok)}/{len(ok)} seeds with modal bid = r_high + 0.01"


def main(prefix: str = "m2"):
    runs = load_runs(prefix)
    groups: dict[str, list[dict]] = {}
    for name, m in runs.items():
        groups.setdefault(name.rsplit("_s", 1)[0], []).append(row(name, m))
    for g, rows in groups.items():
        print(f"\n### {g}")
        print_table(rows)
        if "br_a0" in g:
            passed, msg = h1(rows)
            print(f"H1: {'SUPPORTED' if passed else 'NOT SUPPORTED'} ({msg})")
        if "br_a1" in g and "disc" in g:
            passed, msg = h8_discrete(rows)
            print(f"H8 (discrete): {'SUPPORTED' if passed else 'NOT SUPPORTED'} ({msg})")
        if "br_a1" in g and "beta" in g:
            print(f"H8 (continuous, descriptive): mean A = {np.mean([r['A'] for r in rows]):.3f}, "
                  f"mean offset = {np.mean([r['offset'] for r in rows]):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
