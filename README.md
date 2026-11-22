# reserve-bunching-rl

**Do learning bidders discover the reserve-price heuristic?**

In first-price auctions where the reserve price switches at random between a low and a high level, real bids pile up just above the high reserve, even in auctions where the low reserve applies. Alcobendas & Zeithammer, *Mass of Bids at Reserve* (2025), explain this with a "censoring heuristic": bidders shade as if no reserve existed, then raise any bid below the reserve up to the reserve.

This project simulates those auctions with reinforcement-learning (PPO) bidders. It measures whether the bidders converge to the Bayes–Nash equilibrium, to the heuristic, or to something else. The measurements are the paper's own bunching tests plus atom share, a bin-width scaling exponent κ, tail Kolmogorov–Smirnov (KS) tests and exploitability.

All data is synthetic. No external datasets are needed.

## Documents

| File | Contents |
|---|---|
| [`SPEC.md`](SPEC.md) | Full design: background, environment, bidder types, PPO design, metrics, experimental factors, hypotheses H0–H9, milestones |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Frozen metric definitions, thresholds, decision rules, amendments, frozen M3 PPO settings, M2 outcome record |
| `REPORT.md` | Final write-up (to be written at M5) |

## Status

| Milestone | State | Notes |
|---|---|---|
| **M1**: environment and benchmarks | ✅ Done | Section 5.3 table reproduced (H0) |
| **M2**: PPO sanity checks | ✅ Done | H1 not supported under the strict rule (see the PREREGISTRATION M2 outcome record); H8 (discrete) supported 5/5. PPO tuned (T1–T4) and settings frozen |
| **M3**: primary grid | ⏳ Next | 9 training configurations × 10 seeds, evaluating H2–H7 |
| **M4**: secondary factors and A4/A5 | Not started | |
| **M5**: report | Not started | |

**Frozen M3 PPO settings:** 5,000 updates. The learning rate is 3e-4, held for the first 50% of training, then decayed linearly to 10% of its starting value. The entropy coefficient is 0.01, annealed linearly to 0. Everything else follows SPEC 7.5.

## Setup

Requires Python ≥ 3.10 (3.11 recommended) and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/aiyer9175/reserve-bunching-rl.git
cd reserve-bunching-rl
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"        # M1: environment and benchmarks
uv pip install --python .venv/bin/python -e ".[dev,rl]"     # M2+: adds torch for PPO
```

## Usage

```bash
# Tests
.venv/bin/python -m pytest                 # fast unit tests
.venv/bin/python -m pytest -m slow         # Monte Carlo acceptance tests

# M1 acceptance: reproduce the SPEC 5.3 table (H0)
.venv/bin/python -m rbrl.verify configs/m1_benchmarks.yaml

# Single training run
.venv/bin/python -m rbrl.train configs/m2_br_a0_beta.yaml --seed 0 [--updates N] [--threads 1]

# Sweep: configs × seeds in parallel, one process per run
.venv/bin/python -m rbrl.sweep configs/m2_*.yaml --seeds 0-4 --jobs 8

# Per-seed tables and pre-registered decision rules
.venv/bin/python -m rbrl.summarize m2
```

Each run writes `results/<run_id>_s<seed>/`, which contains:
- `config.yaml`, including the git commit, for provenance
- `metrics.json`
- `train.log`
- `eval.parquet` and `checkpoints/`, which are large and kept out of git

### Running long sweeps on a remote server

M3 is 90 PPO runs, so run it on a many-core machine inside `tmux` so it survives disconnects:

```bash
tmux new -s m3
.venv/bin/python -m rbrl.sweep configs/m3_*.yaml --seeds 0-9 --jobs <n_cores>
# detach: Ctrl-b d    reattach: tmux attach -t m3
```

Set `--jobs` to about the number of physical cores. Each run uses one torch thread.

## Layout

```
configs/                 YAML run configs (m1_*, m2_*, tuning/t1–t4)
results/                 one directory per run (metrics tracked; parquet and checkpoints ignored)
src/rbrl/
  distributions.py       value distributions, reserve schedules (mixed / none)
  env.py                 vectorized first/second-price auctions, per-auction or per-bidder reserves
  benchmarks.py          b*(v; r) closed form and numeric, heuristic, v~, analytic revenue
  bidders/fixed.py       A0 equilibrium, A1 heuristic, truthful
  bidders/ppo.py         PPO actor-critic bidders (Beta shading head, 1-cent bid-grid head)
  bidders/wrappers.py    censor wrapper for the two-stage learners (A2, A2b, A2-r0)
  train.py               PPO training loop (best response and self-play)
  sweep.py               parallel config × seed runner
  summarize.py           per-seed tables and pre-registered decision rules
  metrics.py             bunching shares, atom share, kappa, tail KS, distances, economics, classification
  exploitability.py      best-response utility against actual opponents
  evaluate.py            simulate a frozen population and run the metric battery
  verify.py              M1 acceptance
  runlog.py              run directories and provenance
tests/                   unit tests (env, benchmarks, metrics, PPO)
```

## Related work

- **Alcobendas & Zeithammer (2025), *Mass of Bids at Reserve*:** the empirical anomaly and the censoring heuristic this project tests.
- **Amazon AuctionGym** (https://github.com/amazon-science/auction-gym, Apache-2.0, KDD 2023): simulates first- and second-price ad auctions with learning bidders, but has no reserve prices. This project is standalone and does not depend on it.
