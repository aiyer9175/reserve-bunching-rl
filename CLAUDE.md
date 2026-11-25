# CLAUDE.md

Working notes for Claude Code sessions on this repo. Read `README.md`, `SPEC.md` and `PREREGISTRATION.md` first. The preregistration is binding: any change to metrics, thresholds, decision rules or frozen PPO settings after M3 starts is an **amendment** and must be logged there with a date and reason.

## Where things stand (2026-11-24)

- **M1 done:** H0 reproduced.
- **M2 done:**
  - H1 not supported under the strict rule. See "M2 outcome record" in PREREGISTRATION; the decision was to proceed and report this.
  - H8 (discrete) supported 5/5.
  - PPO tuned (T1–T4, seeds 100–101, not used for hypothesis tests). Settings are frozen for M3.
- **M3 runs done; the analysis is not.**
  - All 90 runs finished (9 configs × seeds 0–9; `results/m3_sweep.log`, about 21 h at 4 jobs). They were committed in 2b8a521.
  - **No run met the convergence rule** (ε_det < 1e-4). Every run hit the 5,000-update cap, with a final ε_det of about 1.5e-3 to 4.5e-3. ε_det plateaued by about update 500.
- **Interim M3 results** come from scratch scripts, not `rbrl.summarize`. Confirm them with the official code before reporting:

  | Hypothesis | Interim result |
  |---|---|
  | H2 | Supported. A3 is equilibrium-like in 10/10 seeds at each p_high. A ≈ 0.0002, κ ≈ 1.25, tail KS D 0.10–0.14, revenue at r_high 0.612. |
  | H3 | No transient. The seed-mean A for A3 never exceeds 0.0003. Runs start from truthful initialization. |
  | H4 | Not supported. (i) Time to convergence is censored at 5,000 in every run, so the difference is 0. (ii) The AUC of A, p95 − p50, is −0.022 with CI [−0.048, 0.007]. |
  | H5 | Falsified. A3-blind has A ≈ 0 and P_high ≈ P_low ≈ 0.01 (p50) / 0.03 (p95). |
  | H6a | Looks supported, 10/10 seeds. A2-r0: b̃ L² 0.005, A 0.331, κ 0.045, revenue loss about −4.2%. The loss was measured on the r_high slice of the p=0.5 evaluation; the required r_high-only evaluation is still to run. |
  | H6b | p50 passes 10/10 (A 0.163, κ 0.046, b̃ L² 0.026). p95 fails 0/10 (A 0.040, κ 0.37). |
  | H7 | Supported at both p_high. A3-disc − A3 ≈ +0.11, bootstrap CI about [0.10, 0.13]. The A3-disc t-interval is above 0.087. |

- **Oddities to explain before writing up** (likely visible in A3's bid function vs b* near r_high):
  - A3's A is below equilibrium's (0.0002 vs about 0.012), and κ is about 1.25 vs 0.54. Bids are thinner than equilibrium just above the reserve.
  - A3's tail KS D is about 2× equilibrium's 0.06.
- **M3 wrap-up tasks, in order:**
  1. Implement the H2–H7 decision rules in `rbrl.summarize`, following PREREGISTRATION Section 6 and Amendments 2, 4 and 5. Cross-check against the table above.
  2. Evaluate A2-r0 from its checkpoints at p_high = 0.95 and r_high-only. `evaluate.py` has no checkpoint-reload command yet, so add a small one.
  3. Evaluate A0 and A1 under the same evaluation as M3.
  4. Plot A3's bid functions against b* and b̃ near each reserve, to explain the oddities above.
  5. Commit and write an M3 summary.
- **H4 reporting, pending the user's confirmation:** the recommendation is no amendment. Report H4(i) as uninformative because every run is censored, and report H4(ii) as computed. Do not change any decision rule without the user's approval.
- Pause and summarize after M3 before starting M4. The first M4 factor will probably be F8 (random init), because of H3. Coordinate compute with Anand.

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
- **Pushing:** the server has no GitHub credentials, so `git push` there fails. Commit on the server and leave pushing to the user. From the Mac, push with `git fetch pd-server-10:work/reserve-bunching-rl main && git push origin FETCH_HEAD:refs/heads/main`.
