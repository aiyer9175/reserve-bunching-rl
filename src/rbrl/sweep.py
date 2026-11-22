"""Run several configs x seeds in parallel, one process per run (1 torch thread each).

    python -m rbrl.sweep configs/a.yaml configs/b.yaml --seeds 0-4 --jobs 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .runlog import ROOT, load_config

# One thread per run for numpy/scipy BLAS too (torch is pinned in train.py), whatever the shell sets.
SINGLE_THREAD_ENV = os.environ | {k: "1" for k in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    jobs = [(c, s) for c in args.configs for s in parse_seeds(args.seeds)]
    t0 = time.time()

    def run(job):
        cfg, seed = job
        name = f"{load_config(cfg)['run_id']}_s{seed}"
        p = subprocess.run([sys.executable, "-m", "rbrl.train", cfg, "--seed", str(seed)],
                           cwd=ROOT, capture_output=True, text=True, env=SINGLE_THREAD_ENV)
        last = (p.stdout.strip().splitlines() or [""])[-1]
        status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})\n{p.stderr[-2000:]}"
        print(f"[{time.time() - t0:6.0f}s] {name}: {status}  {last}", flush=True)
        return p.returncode

    with ThreadPoolExecutor(args.jobs) as ex:
        codes = list(ex.map(run, jobs))
    print(f"done: {sum(c == 0 for c in codes)}/{len(codes)} ok in {time.time() - t0:.0f}s")
    sys.exit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
