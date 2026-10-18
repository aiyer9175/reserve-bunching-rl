# Pre-registration

This file fixes the metric definitions, thresholds and decision rules **before any learned bidder is trained**. It follows `SPEC.md` (Rev 3). Where the spec says "≈", "large" or "suggested", this file replaces it with a fixed number.

Any later change must be committed as a dated amendment at the bottom of this file, giving the reason. The original text stays in place. Results produced before an amendment are reported under the rules that held when they were produced.

---

## 1. Evaluation protocol

| Item | Rule |
|---|---|
| Evaluation size | 10⁶ auctions per evaluation, policies frozen. *(Storage of bids: see Amendment 3.)* |
| Evaluation streams | Seed `100_000 + training_seed`. It is never used in training |
| Test evaluation | **Balanced** reserve mix (50/50). Used for every statistical test and for classification: P(δ), A, A_exact, κ, KS, distances |
| Economic evaluation | The run's own p_high. Used for revenue, surplus, unsold share, efficiency, tie frequency and ε |
| Pooling | Bids are pooled over all N bidders in the evaluation sample (symmetric populations). For mixed populations, metrics are computed per bidder type |
| Policy mode | **Deterministic** for classification and hypotheses. Stochastic is reported alongside |
| Deterministic mode (Beta) | x = Beta mean α/(α+β), bid = x·v. The Beta mode (α−1)/(α+β−2) is a diagnostic column only |
| Deterministic mode (categorical) | arg-max grid point; ties go to the lower bid |
| "At convergence" | The final evaluation of the run: at early stop (Section 4) or at the update cap, whichever comes first |
| Dynamics | A full metric evaluation every 50 updates, with 2×10⁵ auctions (smaller than the final evaluation, for speed) |

## 2. Metric definitions

Notation: r_H = r_high, r_L = r_low. "Bids" means submitted bids. Abstentions (v < r, or a wrapper's "no bid") are excluded from every conditional share.

| Symbol | Definition |
|---|---|
| P_high(δ) | Pr(bid ∈ [r_H, r_H + δ] \| reserve = r_H, bid ≥ r_H) |
| P_low(δ) | Pr(bid ∈ [r_H, r_H + δ] \| reserve = r_L, bid ≥ r_H) |
| δ grid | {0.001, 0.0025, 0.005, 0.01, 0.02}; primary δ = 0.01 |
| Two-proportion test | Pooled-variance z-test of P_high = P_low; difference reported with a 95% Wald CI (unpooled) |
| **A** (atom share, primary) | Pr(\|bid − r_H\| < τ \| reserve = r_H, bid ≥ r_H), **τ = 1e-4** |
| A_exact (diagnostic) | Same, with tolerance 1e-9 |
| κ | OLS slope of log P_high(δ) on log δ over the δ grid. 95% CI from 1,000 multinomial bootstrap resamples of the conditioning sample |
| Tail KS | `scipy.stats.ks_2samp` on X = {bids > r_H + 0.01 \| r_H}, Y = {bids > r_H + 0.01 \| r_L}. Report D, p, n_X, n_Y |
| Subsample KS | 200 draws of n = 5,000 per group without replacement; rejection rate at α = 0.05 |
| Bid-function distance | Deterministic bid on v ∈ [r, 1], step 0.005, for each r ∈ {r_L, r_H}. **L²** = root-mean-square difference over both grids pooled (equal weight per reserve). **Sup** = maximum absolute difference over both grids |
| b̃ distance (A2 family) | L² distance of the unclamped π(v) to b̃(v) on v ∈ [0, 1], step 0.005 |
| ε_det (primary exploitability) | Section 8.6 of the spec. The focal bidder and all learned opponents play deterministic policies. H_r is estimated from 2×10⁵ Monte Carlo draws per r. Best response is searched over the 1,001-point grid {0, 0.001, …, 1}, ties included. The expectation over v uses 400 value quantiles, weighted by the run's reserve mix |
| ε_stoch (diagnostic) | Same as ε_det, with all learned bidders playing stochastic policies |
| Offset (H8, continuous) | Median of (bid − r_H) among bids in [r_H, r_H + 0.01] under reserve r_H |

## 3. Run classification

Applied per seed, on the balanced deterministic evaluation at convergence. Let L²(A0) and L²(A1) be the bid-function distances to the equilibrium and the heuristic.

- **Heuristic-like:** L²(A1) + 0.01 ≤ L²(A0) **and** A > 0.05.
- **Equilibrium-like:** L²(A0) + 0.01 ≤ L²(A1) **and** κ > 0.3 **and** the tail KS rejects (full-sample p < 0.05 **and** D > 0.02).

*(For discrete-grid bidders, these rules are superseded by Amendment 4.)*
- **Neither:** anything else.

The D > 0.02 effect-size floor stops the very large evaluation sample from counting trivial differences as rejections. The equilibrium benchmark has D ≈ 0.06.

## 4. Convergence and stopping

- **Converged:** ε_det < 1e-3 at every dynamics evaluation over a window of 200 updates (5 consecutive evaluations). *(Threshold superseded by Amendment 1.)*
- Training stops at the first update where this holds, or at the update cap (default 2,000).
- **Time to convergence** = the first update of that window. Runs that never converge are recorded as censored at the cap.

## 5. Seeds

| Group | Seeds |
|---|---|
| Primary grid (Section 9 of the spec: A3, A3-blind, A2, A3-disc × p_high ∈ {0.5, 0.95}, plus A2-r0) | 0–9 |
| Sanity checks (H1, H8) and secondary factors | 0–4 |
| Benchmark verification (H0) | One fixed evaluation seed (12345), plus the analytic values |

No seed is dropped or replaced after the fact. Runs that crash are rerun with the same seed. A run that finishes but diverges (NaN loss) is reported as diverged, not rerun.

## 6. Hypotheses: decision rules

Seed-level statistics are summarized as the mean with a 95% t-interval across seeds. "Seed-mean" means the mean across seeds.

| ID | Supported if (all conditions) | Otherwise |
|---|---|---|
| **H0** | Every Section 5.3 value for A0 and A1 is matched within max(3·MC standard error, 0.0015). Tail KS: D for A0 within 0.01 of 0.060 and rejecting; for A1, the subsample rejection rate ≤ 0.10. Revenue within 0.002 of 49/80 (A0) and 0.5885 (A1) | Falsified → fix code before M2 |
| **H1** | For best response to fixed A0 (continuous and discrete, 5 seeds each), every seed has sup distance < 0.02 and ε_det < 1e-3 *(threshold superseded by Amendment 1)* | Falsified → tune PPO before M3 |
| **H2** | In ≥ 8/10 seeds of A3 (each p_high separately): A < 0.05 **and** κ > 0.3 **and** the tail KS rejects (Section 3 rule) | Falsified. If A > 0.05 in ≥ 3/10 seeds, report this as "persistent bunching" |
| **H3** | For A3: the seed-mean A curve peaks at ≥ 0.05 at some dynamics evaluation **and** the final seed-mean A is < 0.5 × peak **and** < 0.05 | No transient (peak < 0.05), or no decay (see H2) |
| **H4** | The p_high = 0.95 minus p_high = 0.5 difference in (i) time to convergence and (ii) area under the A curve (trapezoid, over update steps) both have a 95% bootstrap CI (10,000 resamples over seeds) above 0 | No difference |
| **H5** | For A3-blind, in ≥ 8/10 seeds: P_low(0.01) ≥ 0.15 **and** P_high(0.01) ≥ 0.15 **and** P_high/P_low < 1.5 | The spike disappears (P_high/P_low ≥ 1.5 or P_low < 0.15) |
| **H6a** | For A2-r0, in ≥ 8/10 seeds: b̃ distance < 0.02, \|A − 0.333\| < 0.03, the κ CI contains 0 *(superseded by Amendment 2)*, and the revenue loss vs A0 is within 1 percentage point of −3.9% (r_high-only evaluation) | Falsified |
| **H6b** | For A2, in ≥ 8/10 seeds: A > 0.05, the κ CI contains 0 **or** κ < 0.15, and b̃ distance ≥ 0.02. Report the revenue gap to A1 (size and sign) | Falsified |
| **H7** | Seed-mean A of A3-disc minus seed-mean A of A3 (same p_high) has a 95% bootstrap CI above 0 *(superseded by Amendment 5)* | No difference |
| **H8 (discrete)** | Best response to fixed A1: at r_H, for v ∈ [0.53, 0.66], the modal deterministic bid is r_H + 0.01 in ≥ 4/5 seeds | Modal bid elsewhere |
| **H8 (continuous)** | Descriptive only: report A, A_exact and the offset. A continuous bidder can't land on r_H + 0.01, so this version is not tested | — |
| **H9** | Seed-mean A of A5 < 0.05, in each p_high | A ≥ 0.05 |

The tail KS is not part of H6. For wrapper bidders (A2, A2b, A2-r0) equal tails hold by design, so the test is reported only as a code check.

## 7. Multiplicity

Each hypothesis has one pre-set decision rule over a fixed set of seeds. No p-value correction is applied across hypotheses. Every metric for every configuration is reported in full, whatever the hypothesis outcome.

## 8. What is not pre-registered

PPO hyperparameter tuning (spec Section 7.5) happens only in M2, on the H1/H8 sanity runs. The hyperparameters are then frozen for M3. Any tuning after M3 starts is an amendment.

---

## Amendments

### Amendment 1 (2026-10-17): convergence threshold 1e-3 → 1e-4

**Changes:** Section 4 (convergence) and H1 use ε_det < **1e-4** instead of 1e-3.

**Reason:** the M1 benchmark run (`results/m1_benchmarks`, commit c743be6) showed that a population playing the heuristic has ε_det = 6.6e-4 at p_high = 0.5. That matches a hand calculation: at r_high, a bidder with v ∈ (r_high, 2/3) gains by bidding just above r. At p_high = 0.95 it is about 1.24e-3. The old threshold would therefore count a heuristic population as converged at p_high = 0.5. Equilibrium play has a Monte Carlo floor of about 1.1e-5. A threshold of 1e-4 sits 9× above that floor and about 6.6× below the heuristic.

**Timing:** made after M1 and before any learned bidder was trained.

### Amendment 2 (2026-10-17): H6a κ criterion

**Changes:** in H6a, "the κ CI contains 0" becomes **κ < 0.15**, the rule H6b already uses.

**Reason:** the exact heuristic itself has κ = 0.045, 95% CI [0.044, 0.045], at 10⁶ auctions (M1). The smooth part of its bid distribution above the atom makes P_high(δ) rise slightly with δ. Requiring the CI to contain 0 would reject the benchmark that H6a is meant to recognize. Equilibrium has κ = 0.536, so 0.15 separates the two clearly.

**Timing:** made after M1 and before any learned bidder was trained.

### Amendment 3 (2026-10-17): stored evaluation bids

**Changes:** learned runs store a **2×10⁵-auction sample** of the final evaluation in `eval.parquet`. All metrics are still computed on the full 10⁶-auction evaluation, which is reproducible from the checkpoint and the evaluation seed.

**Reason:** a full 10⁶-auction evaluation takes about 95 MB as Parquet, or about 4 GB across the primary grid. This changes storage only; no metric or decision rule changes.

**Timing:** made after M1 and before any learned bidder was trained.

### Amendment 4 (2026-10-18): classification of discrete-grid bidders

**Changes:** for bidders on the 1-cent grid (A3-disc and any other categorical policy):
- **Heuristic-like:** L²(A1) + 0.01 ≤ L²(A0) **and** A > 0.2.
- **Equilibrium-like:** L²(A0) + 0.01 ≤ L²(A1) **and** A < 0.2 **and** the tail KS rejects (p < 0.05 **and** D > 0.02). No κ condition.
- **Neither:** anything else.

**Reason:**
- On a 1-cent grid, P_high(δ) for δ < 0.01 counts only bids exactly at r_high, so κ is not a measure of bin-width scaling. It was 0 in every untuned M2 discrete run.
- Rounding the benchmarks to the grid (10⁶ auctions, seed 12345) gives atom shares of A = 0.087 for equilibrium and 0.347 for the heuristic. The equilibrium bid is flat at the reserve: b* < r + 0.005 for v < r + 0.041, so all those values round to exactly r. The rounded equilibrium also has κ = 0.32, at the old 0.3 cutoff.
- A cutoff of 0.2 lies between the two grid benchmarks.

**Timing:** made after the M2 sanity runs (best response to fixed A0 and A1 opponents) and before any M3 self-play run.

### Amendment 5 (2026-10-18): H7 relative to grid rounding

**Changes:** H7 is supported only if **both** hold:
- (i) seed-mean A of A3-disc minus seed-mean A of A3 (same p_high) has a 95% bootstrap CI above 0; **and**
- (ii) seed-mean A of A3-disc has a 95% t-interval above 0.087, the atom of the equilibrium rounded to the grid.

**Reason:** by Amendment 4's calculation, a discrete bidder playing the rounded equilibrium already has A ≈ 0.087. The original H7 would count that mechanical rounding atom as "raising the atom share".

**Timing:** as Amendment 4.

### Frozen PPO settings for M3 (2026-10-18; Section 8)

Chosen on the M2 tuning runs (`results/tune_*`, seeds 100–101, which are not used for any hypothesis test):
- **updates:** 5,000
- **lr:** 3e-4, held for the first 50% of training, then decayed linearly to 10% of its starting value
- **entropy coefficient:** 0.01, annealed linearly to 0 over training
- **unchanged from SPEC 7.5:** auctions per update (8,192), epochs, minibatch, clip, value coefficient, gradient clip, network

Under self-play with shared parameters, each update has 4 × 8,192 = 32,768 samples. That equals the per-update data of the best tuning variant (T2), which had 32,768 auctions with one learner.

### M2 outcome record (2026-10-18)

- **H1 (pre-registered strict rule): not supported.**
  - **Exploitability passes for the continuous learner after tuning:** ε_det of 5.5e-5 and 5.6e-5 (T2, tuning seeds).
  - **The sup distance fails by construction.** At r_low, the policy at v = r is still Beta(1, 1): no bid there can win, so the only learning signal is the entropy bonus. Its deterministic bid is 0.05, against b* = 0.1.
  - **The discrete learner does not reach ε_det < 1e-4.** Its best was 1.13e-4. The 1-cent rounding floor is 2.4e-5.
  - **Decision:** proceed to M3 and report this. The H1 rule is not amended.
- **H8 (discrete): supported**, 5/5 seeds (`results/m2_br_a1_disc_*`).

