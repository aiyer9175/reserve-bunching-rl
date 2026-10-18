# reserve-bunching-rl

Do learning bidders discover the reserve-price heuristic? A simulation framework for first-price auctions with a randomly switched reserve. It compares learned bidders with the Bayes–Nash equilibrium and with the "censoring heuristic" of Alcobendas & Zeithammer, *Mass of Bids at Reserve* (2025).

- `SPEC.md`: the full design (research questions, environment, bidders, metrics, hypotheses).
- `PREREGISTRATION.md`: fixed metric definitions, thresholds and decision rules.
- `REPORT.md`: results (written at M5).

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"        # M1: environment and benchmarks
uv pip install --python .venv/bin/python -e ".[dev,rl]"     # M2+: adds torch for PPO
```

## Usage

```bash
.venv/bin/python -m pytest                                  # unit tests
.venv/bin/python -m rbrl.verify configs/m1_benchmarks.yaml  # M1 acceptance: SPEC 5.3 table (H0)
```

Each run writes `results/<run_id>/` with `config.yaml` (including the git commit), `metrics.json` and, optionally, `eval.parquet`.

## Layout

```
src/rbrl/
  distributions.py   value distributions, reserve schedules (mixed / none)
  env.py             vectorized first/second-price auctions, per-auction or per-bidder reserves
  benchmarks.py      b*(v; r) closed form and numeric, heuristic, v~, analytic revenue
  bidders/fixed.py   A0 equilibrium, A1 heuristic, truthful
  metrics.py         bunching shares, atom share, kappa, tail KS, distances, economics, classification
  exploitability.py  best-response utility against actual opponents
  evaluate.py        simulate a frozen population and run the metric battery
  verify.py          M1 acceptance
  runlog.py          run directories and provenance
```

## Related work

Amazon AuctionGym (https://github.com/amazon-science/auction-gym, Apache-2.0, KDD 2023) simulates first- and second-price ad auctions with learning bidders, but has no reserve prices. This project is standalone and does not depend on it.
