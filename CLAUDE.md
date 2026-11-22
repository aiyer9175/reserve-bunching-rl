# CLAUDE.md

Working notes for Claude Code sessions on this repo. Read `README.md`, `SPEC.md` and `PREREGISTRATION.md` first. The preregistration is binding: any change to metrics, thresholds, decision rules or frozen PPO settings after M3 starts is an **amendment** and must be logged there with a date and reason.

## Where things stand (2026-11-21)

- **M1 done:** H0 reproduced.
- **M2 done:**
  - H1 not supported under the strict rule. See "M2 outcome record" in PREREGISTRATION; the decision was to proceed and report this.
  - H8 (discrete) supported 5/5.
  - PPO tuned (T1–T4, seeds 100–101, not used for hypothesis tests). Settings are frozen for M3.
- **M3 is next:** primary grid, SPEC Sections 9 and 11. The tasks:
  1. Write the 9 training configs: A3, A3-blind, A2, A3-disc × p_high ∈ {0.5, 0.95}, plus A2-r0 trained once per seed and evaluated under both p_high mixes. All with N = 4, uniform values, shared self-play, and the frozen PPO settings.
  2. Smoke-test each config with `--updates` small.
  3. Run 10 seeds (0–9) per config.
  4. Log the dynamics.
  5. Evaluate H2–H7 using the preregistered rules (`rbrl.summarize`).
  6. Run benchmarks A0 and A1 under the same evaluation.
- Pause and summarize after M3 before starting M4.

## Compute: pd-server-10

- **Repo location:** `~/work/reserve-bunching-rl` (user `anirudh`). The server is shared and owned by Anand.
- **Hardware:** 72 logical CPUs, 755 GB RAM, no GPU (CPU torch).
- **Agreed usage:** start with **4 parallel workers**, i.e. `rbrl.sweep --jobs 4`. Coordinate with Anand before running more. Each run uses one torch thread.
- **Long jobs:** always run inside `tmux`, e.g. `tmux new -s m3`.
- **Environment:** `.venv` uses Python 3.12 with the same versions as the Mac: numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, pyarrow 26.0.0, torch 2.14.1 (CPU).
- **Cross-machine reproducibility:** Mac and server results agree to about 1e-9 but are **not bit-identical** (ARM vs x86 floating point). Run all M3 seeds on the server so the grid is internally consistent.

## Conventions

- **Results:** each run writes `results/<run_id>_s<seed>/`. Commit `config.yaml`, `metrics.json` and `train.log`. `eval.parquet` and `checkpoints/` are gitignored; copy them with rsync if needed.
- **Commits:** commit configs before running them so that `config.yaml` records a clean git commit.
- **Remote:** https://github.com/aiyer9175/reserve-bunching-rl (public).
