# Spec: Do Learning Bidders Discover the Reserve-Price Heuristic?

**Audience:** a coding agent starting from an empty repository.
**Goal:** build a small, clean simulation framework and run the experiments below. Read the whole spec before writing code; Sections 1–2 explain *why* each design choice matters, and many choices only make sense with that context.

Suggested repo name: `reserve-bunching-rl`

### Revision log

**Rev 2 (2026-10-16).** Four fixes from spec review. Each changed passage is marked **[Rev 2]**.
1. **H6's tail result holds by design.** A2's wrapper only raises bids up to r, so above ṽ its bids cannot depend on r, and the tail KS cannot reject. That result is now treated as a consistency check, not as evidence (Sections 6, 10).
2. **A2 does not learn b̃.** Training A2 on auctions with reserves and no wrapper makes it close to A3-blind, so its candidate bid already bends toward r_high. A new variant, **A2-r0**, learns its candidate bid with no reserve (r = 0) and only gets the clamp at bid time. A2-r0 is now the treatment compared with the analytic −3.9% (Sections 6, 9, 10).
3. **Exploitability uses the deterministic policy.** With a random policy, exploration noise alone can keep ε above 1e-3, especially with fixed entropy (F7). The convergence rule and H1 now use ε of the deterministic profile. Stochastic ε is reported only as a diagnostic (Sections 7.6, 8.6, 10).
4. **Deterministic mode has one definition: the Beta mean.** The mean and the mode give different bid functions. The mean is used for all classification. The mode is reported as a secondary diagnostic only (Sections 7.2, 8.2).

**Rev 4 (2026-10-17).** Two fixes after the M1 benchmark run, marked **[Rev 4]**. Both are recorded as amendments in `PREREGISTRATION.md`.
6. **Convergence threshold 1e-3 → 1e-4.** A population playing the heuristic has ε_det = 6.6e-4 at p_high = 0.5, so the old threshold would count it as converged. Equilibrium play has a Monte Carlo floor of about 1e-5 (Sections 8.6, 10).
7. **H6a uses κ < 0.15 instead of "κ CI contains 0".** The exact heuristic has κ = 0.045 with a narrow CI that excludes 0 (Section 10, via `PREREGISTRATION.md`).

**Rev 3 (2026-10-17).** One fix, marked **[Rev 3]**.
5. **The atom is measured with a tolerance of τ = 1e-4, for every bidder.** A neural network won't output exactly r/v over a range of values, so with the old 1e-9 tolerance a continuous policy would score about 0 even if it bunched heavily. The near-atom share A (tolerance τ = 1e-4) is now the atom measure everywhere: for classification, hypotheses, tables and figures. It gives the same answer as the exact check for the discrete-grid, wrapper and benchmark bidders. The exact 1e-9 share is reported as a diagnostic column only (Sections 5.3, 7.2, 8.2).

---

## 0. TL;DR for the agent

1. Build a vectorized first-price sealed-bid auction environment with a **reserve price that is randomly switched per auction** between a low and a high level.
2. Implement two **analytic benchmark bidders** (Bayes–Nash equilibrium and a "censoring heuristic") and verify that a set of statistical tests reproduces the predicted numbers in Section 5.
3. Train **PPO bidders** (and a few variants) in this environment and measure whether they end up behaving like the equilibrium, like the heuristic, or something else, using the same tests.
4. Produce figures and a results table (Section 12).

No external data is needed. Everything is synthetic (Section 3).

---

## 1. Background: the paper this builds on

**Paper:** Alcobendas & Zeithammer, *"Mass of Bids at Reserve"* (working paper, Dec 2025; Yahoo Research / UCLA Anderson).

### 1.1 Setting
- Online display advertising. When a user loads a web page, the **publisher** (seller) sells the ad slot (one **impression**) through an **ad exchange** running a **first-price sealed-bid auction**: highest bid wins and pays its own bid.
- The exchange sets a **reserve price** *r*: bids below *r* are rejected. If no bid ≥ *r*, the impression is unsold.
- Bidders are automated bidding agents. Each has a private **valuation** *v* for the impression and submits a bid *b*. Payoff to the winner is *v − b* (quasi-linear); losers get 0.
- Because the auction is first-price, rational bidders **shade**: bid below value.

### 1.2 The natural experiment in the data
The exchange normally uses a revenue-optimized **"production" reserve** (e.g. $0.71 CPM), but in a random ~5% of auctions it switches to a much lower **"monitoring" reserve** ($0.15) to observe bids it would otherwise not see. Because the switch is random, auctions in the two reserve conditions are otherwise comparable. This lets the authors compare bidding behaviour under two reserves holding everything else fixed.

### 1.3 The anomaly
In production auctions, a large share of bids sit **exactly at the reserve** (26.6% of all bids; 11.7% for Yahoo's own bidder; up to ~80% for one bidder). Standard theory says equilibrium bidding in first-price auctions has **no point mass at the reserve**: if competitors bunched at *r*, a bidder with *v > r* would bid *r + ε* to break the tie at negligible cost, so the bunching unravels.

### 1.4 The proposed mechanism: a "censoring heuristic"
Bidders appear to:
1. Pre-compute a **candidate bid** b̃(v) that maximizes Pr(win | b)·(v − b), where Pr(win | b) is an *average* win-probability curve that **ignores the reserve**.
2. When a request arrives with reserve *r*: bid nothing if v < r; bid **exactly r** if b̃(v) < r ≤ v; otherwise bid b̃(v).

Formally:
```
β̃(v | r) = no bid      if v < r
          = r           if r ≤ v ≤ ṽ,   where ṽ = b̃⁻¹(r)
          = b̃(v)        if v > ṽ
```
The middle branch creates the point mass at the reserve. A possible origin: bidders kept valuation models from the earlier second-price era and bolted on a reserve-blind shading step after the industry switched to first-price auctions (~2019).

### 1.5 The paper's two indirect tests (we reuse both)
Let r_prod = production reserve, r_0 = monitoring reserve.

**Test 1, disappearing mass point.** Among bids ≥ r_prod, compare the share falling in [r_prod, r_prod + $0.01]:
- P₀ = Pr(bid ∈ [r_prod, r_prod+0.01] | reserve = r_0, bid ≥ r_prod)
- P_prod = Pr(bid ∈ [r_prod, r_prod+0.01] | reserve = r_prod, bid ≥ r_prod)

The heuristic predicts P_prod ≫ P₀. (A clustering of *valuations* at r_prod would predict P_prod ≈ P₀.) Tested with a two-proportion test.

**Test 2, tail comparison.** Compare the distribution of bids strictly above r_prod + $0.01 under the two reserve conditions with a two-sample Kolmogorov–Smirnov test. The heuristic predicts **identical tails**, because above ṽ the bid b̃(v) does not depend on *r*. Equilibrium predicts **different tails**, because the reserve enters the bid function at every valuation.

### 1.6 Empirical results (context only; do not fit to these)
- 11 of 12 large bidders show the disappearing mass point; 6 of 12 also pass the tail test.
- Regular DSPs (direct bidders) fit the heuristic better (4 of 5 pass the tail test) than Rebroadcasters (intermediaries that run their own sub-auctions; 2 of 7 pass).
- A structural counterfactual estimates that switching all bidders from the heuristic to Bayes–Nash equilibrium would raise publisher revenue ~5% (~9% if reserves are re-optimized).

### 1.7 The gap this project fills
The paper establishes *that* real bidders bunch and quantifies the cost, but never asks **how such behaviour arises or whether learning removes it**. This project asks: *if bidders learn to bid from experience (reinforcement learning), do they discover the heuristic, converge to equilibrium, or end up somewhere else, and what determines which?*

---

## 2. Research questions

- **RQ1.** Do PPO bidders that observe (v, r) converge to the Bayes–Nash equilibrium, or do they exhibit reserve bunching (persistent or transient)?
- **RQ2.** Which ingredients produce heuristic-like behaviour: not observing the reserve, a two-stage "learn-then-censor" pipeline, coarse/discrete bid grids, low-capacity policies, exploration noise, or rarity of the low-reserve condition?
- **RQ3.** Do the paper's two tests correctly classify bidders whose true strategy we know? (We will find they need a refinement; see Section 5.3.)
- **RQ4.** How much revenue and bidder surplus differ between learned behaviour, the heuristic, and equilibrium?

---

## 3. Scope and data

**In scope (v1):**
- Symmetric bidders, independent private values, single-item first-price sealed-bid auctions.
- One reserve level applied to all bidders in a given auction, drawn per auction from {r_low, r_high}.
- Synthetic valuations only.

**Out of scope for v1 (design so it can be added later; see Section 14):**
- Two asymmetric bidder types with type-specific reserves (the paper's Regular DSP vs Rebroadcaster structure).
- Intermediaries that run sub-auctions.
- Reserve-setting by a learning seller.

**Data:** none external. All valuations are drawn from specified distributions. The paper's numbers appear only as context and as optional calibration targets (Section 9, factor F6). Do not attempt to obtain or reconstruct Yahoo data. This repository should not reference or depend on any other project.

**Data produced** (for evaluation, stored per run as Parquet):
`run_id, seed, update_step, bidder_kind, policy_mode (stochastic|deterministic), v, r, bid, valid (bid ≥ r), won, price_paid, utility`
plus a run-level `config.yaml` and a `metrics.json`.

---

## 4. Environment specification

| Item | Baseline | Notes |
|---|---|---|
| Mechanism | First-price sealed-bid; highest valid bid wins and pays its bid | Implement second-price as a switchable control |
| Reserve | Per auction: r_high = 0.50 with prob. p_high, else r_low = 0.10 | p_high ∈ {0.5, 0.95}. 0.50 is the revenue-optimal reserve for Uniform[0,1] (Myerson) |
| Valid bid | bid ≥ r. Invalid bids are rejected (utility 0) | A bidder with v < r effectively abstains |
| Bidders | N = 4, symmetric | N ∈ {2, 4, 6} as a factor |
| Valuations | i.i.d. Uniform[0, 1], drawn fresh each auction | Alternative distribution in factor F6 |
| Ties | Broken uniformly at random among tied highest valid bids | Matters: the heuristic produces ties at r |
| Bid space | Continuous in [0, 1] (primary); discrete grid {0.00, 0.01, …, 1.00} (variant) | Grid includes r_low and r_high exactly |
| Feedback to learners | Own realized utility only (bandit feedback) | Optional: also reveal winning price |
| Episode | One auction = one step (γ = 0). This is a contextual bandit | — |

**[Rev 2]** The environment must also support a **no-reserve mode** (r = 0 for every auction), used to train A2-r0 (Section 6).

**Implementation:** fully vectorized over a batch of auctions (NumPy or PyTorch). Expect ≥10⁶ auctions/second on CPU for fixed bidders. Seed every random stream.

**Unit tests:** payment and allocation correctness, tie-breaking frequencies, rejection of bids below reserve, and seeds reproducing identical outputs.

---

## 5. Analytic benchmarks and predicted test values

### 5.1 Bayes–Nash equilibrium (A0)
For N symmetric bidders with value CDF F and reserve r, the equilibrium bid for v ≥ r is (Riley–Samuelson 1981):

```
b*(v; r) = v − ∫_r^v [F(z)/F(v)]^(N−1) dz
```
For Uniform[0,1]:
```
b*(v; r) = ((N−1)/N)·v + r^N / (N·v^(N−1)),   v ≥ r
```
Implement the general integral form numerically too, and test it against the closed form.

**Key property (important for Section 5.3):** b*(r; r) = r and b*'(r) = 0. The equilibrium bid function is *flat* at the reserve, so equilibrium bids are dense just above r without having an atom at r.

### 5.2 Censoring heuristic (A1)
- Candidate bid = equilibrium bid **without a reserve**: b̃(v) = ((N−1)/N)·v for uniform values.
- Submitted bid = max(b̃(v), r) if v ≥ r, else no bid.
- Bunching threshold ṽ = r·N/(N−1). For N = 4 and r = 0.5, ṽ = 2/3.

### 5.3 Predicted metric values (N = 4, Uniform[0,1], r_low = 0.10, r_high = 0.50)
These were checked by Monte Carlo with 4M draws. The agent should reproduce them as an acceptance test.

| Metric (defined in Section 8) | Equilibrium A0 | Heuristic A1 |
|---|---|---|
| Exact atom at r_high: Pr(bid = r_high \| r_high, bid ≥ r_high) | **0** | **0.333** |
| Near-atom share A, τ = 1e-4 (Section 8.2) **[Rev 3]** | ≈ 0.012 (analytic approximation; confirm by Monte Carlo) | 0.333 |
| P_high(δ = 0.001) | 0.038 | 0.336 |
| P_high(δ = 0.01), the paper's bin width | **0.128** | **0.360** |
| P_high(δ = 0.02) | 0.188 | 0.387 |
| P_low(δ = 0.01) | 0.040 | 0.040 |
| Tail KS above r_high + 0.01 (n ≈ 65k per group) | D ≈ 0.060, p ≈ 1e-115 → **reject** | D ≈ 0.007, p ≈ 0.10 → **don't reject** |
| Publisher revenue per auction at r = 0.5 | 0.6125 (= 49/80) | ≈ 0.5885 (≈ −3.9%) |
| Share of unsold auctions at r = 0.5 | r^N = 0.0625 | 0.0625 |

**Important methodological finding to carry through the project.** With the paper's 1-cent bin, the equilibrium **also** gives P_high ≫ P_low (0.128 vs 0.040). That is because b*'(r) = 0 concentrates equilibrium bids just above the reserve. So Test 1 as literally defined does not by itself separate the heuristic from equilibrium in this model. Two refinements do:
1. **Atom share:** Pr(bid = r). The heuristic gives a positive atom; equilibrium gives 0. **[Rev 3]** It is measured with the near-atom share A (tolerance τ = 1e-4, Section 8.2). Equilibrium then gives ≈ 0.012, from bids just above r, well below the 0.05 cutoff.
2. **Bin-width scaling:** as δ → 0, the heuristic's P_high(δ) stays at the atom size (~0.333). Equilibrium's goes to 0 like √δ, since b* − r ≈ ½·b*''(r)·(v − r)² near r.

Report the paper's 1-cent version **and** both refinements everywhere. (In the paper's real data, the direct evidence from Yahoo's own bidder — an exactly flat bid at r over a range of values — is consistent with a true atom. This finding concerns how sharply the indirect test discriminates, not whether the paper's conclusion holds.)

---

## 6. Bidder types (treatments)

All learned bidders use the same PPO core (Section 7) unless stated.

| ID | Name | Observes | Policy / rule | Purpose |
|---|---|---|---|---|
| **A0** | Equilibrium | v, r | b*(v; r) | Benchmark |
| **A1** | Heuristic | v, then r | max(b̃(v), r) | Benchmark. Checks that the tests detect bunching |
| **A2** | Two-stage learner | v during learning; r only at bid time | PPO learns a reserve-blind policy π(v) from auctions with randomly drawn reserves; a wrapper submits max(π(v), r) when v ≥ r. Variant A2b: also apply the wrapper during training | Mimics the industry story: learn shading, then bolt on the reserve |
| **A2-r0** **[Rev 2]** | Two-stage learner, reserve-free training | v during learning (all auctions have r = 0); r only at bid time | PPO learns π(v) in self-play with no reserve, then is frozen. At evaluation the wrapper submits max(π(v), r) when v ≥ r | The literal "learn shading in a world without reserves, then clamp" story. Its π(v) should approach b̃(v) = ((N−1)/N)·v, so at evaluation it should approach A1 |
| **A3** | Full PPO | v, r | PPO policy π(b \| v, r) | Main RQ1 treatment |
| **A3-blind** | Reserve-blind PPO | v only | PPO policy π(b \| v), no wrapper | Isolates the role of observing r |
| **A3-disc** | Full PPO, discrete grid | v, r | Categorical over a 101-point bid grid | Can the learner place an exact atom at r? |
| **A4** | No-regret learner (Hedge / multiplicative weights) | context (v-bin, r) | Hedge over the bid grid per context, full-information or bandit (EXP3) version | Learning-theory benchmark against PPO |
| **A5** | Low-capacity PPO | v, r | Policy mean linear in (v, r): μ = w₀ + w₁v + w₂r, then clipped | Does limited capacity alone produce heuristic-like behaviour? |

**[Rev 2] How to read A2 and A2-r0:**
- **A2 is not the same as A1.** A2 trains on auctions that have reserves and has no wrapper during training, so it faces the same problem as A3-blind. Its learned π(v) will already bend toward r_high (see H5). Treat A2 as "A3-blind plus a clamp", and don't expect its π(v) to match b̃(v). Report how far π(v) is from b̃(v) on the evaluation grid.
- **The tail result holds by design for both.** The wrapper only raises bids up to r, so above ṽ the bid is π(v) for both reserves. The tail KS cannot reject (up to sampling error), so for A2, A2b and A2-r0 a non-rejecting tail KS is a check that the code is correct, not evidence about learning.
- **Training opponents for A2-r0:** shared self-play under r = 0. Because training uses no reserve, p_high does not affect A2-r0's training. Train it once per seed and evaluate it under both p_high mixes.

**Opponent configurations** (factor F5):
- **Self-play, shared parameters:** all N bidders use one policy (symmetric). This is the primary configuration.
- **Self-play, independent:** N separate policies.
- **Best response to fixed A0:** 1 learner + (N−1) A0 opponents. This is a sanity check: the best response to equilibrium opponents is the equilibrium bid.
- **Best response to fixed A1:** 1 learner + (N−1) A1 opponents. Predicted best response: bid just above r_high instead of at it. On a 1-cent grid that is r_high + 0.01, which falls inside the paper's [r, r+0.01] bin. That is another illustration of why the exact-atom refinement matters.
- **Mixed populations:** e.g. 2 A3 + 2 A1.

---

## 7. PPO design

The problem is a **one-step contextual bandit**, so PPO here reduces to clipped policy-gradient updates with a learned baseline. No discounting and no GAE.

### 7.1 Observation
`[v, r, 1{r = r_high}]` (A3), or `[v]` (A2, A2-r0, A3-blind). Values are already in [0, 1].

### 7.2 Action parameterization (primary)
- The policy outputs a **Beta(α, β)** distribution over the shading factor x ∈ (0, 1). Bid = x·v.
  - Bids never exceed value, which is rational in a first-price auction.
  - α, β = 1 + softplus(network outputs).
- Continuous actions make an exact atom at r a measure-zero event. A learned "atom" will show up as a sharp concentration near r. Evaluate in two modes:
  - **stochastic:** sample from the policy
  - **deterministic [Rev 2]:** x = **Beta mean** α/(α+β), bid = x·v. This is the only deterministic mode used for classification, exploitability, bid-function distances, and hypothesis tests.
  - **mode (diagnostic only) [Rev 2]:** x = (α−1)/(α+β−2), always defined because α, β > 1. Report the atom share A and the bid-function distances under the mode as a secondary column, but don't classify runs with it. If mean and mode results disagree on a run, flag the run in the report.

  In deterministic mode, a policy that maps a range of v to x = r/v produces an atom. **[Rev 3]** In practice a network only gets close to x = r/v, so atoms from continuous policies are measured with the near-atom share (tolerance τ = 1e-4, Section 8.2).

- **[Rev 2] Why the mean:** it is always well defined, changes smoothly with the network outputs, and is the point that PPO's own gradient updates move. The mode becomes ill-conditioned as α or β approaches 1. The choice is fixed here and in `PREREGISTRATION.md` before any learned run.

### 7.3 Action parameterization (variant A3-disc)
- Categorical over the bid grid {0.00, …, 1.00}.
- Optional mask on bids > v (run with and without).
- **[Rev 2]** Deterministic mode for categorical policies = the arg-max grid point (ties broken toward the lower bid).

### 7.4 Networks
- Separate actor and critic MLPs, 2 hidden layers of 64 units, tanh.
- The critic estimates V(v, r) = expected utility.

### 7.5 Starting hyperparameters (tune lightly and report)

| Hyperparameter | Value |
|---|---|
| Auctions per update | 8,192 (× N samples under parameter sharing) |
| PPO epochs per update | 4 |
| Minibatch size | 1,024 |
| Clip ε | 0.2 |
| Optimizer | Adam, learning rate 3e-4, linear decay |
| Advantage | r_t − V(s); normalized per minibatch |
| Entropy coefficient | 0.01, linearly annealed to 0 (factor F7: fixed vs annealed) |
| Value loss coefficient | 0.5 |
| Gradient clip | 0.5 |
| Number of updates | 2,000 (~16M auctions). Make configurable and stop early on convergence (Section 8.6) |
| Initialization | Shading factor ≈ 0.9 (near-truthful), so the learner must discover shading. Variant: random init |

### 7.6 Pitfalls
- **Non-stationarity in self-play:** monitor exploitability over time. If it oscillates, try a smaller learning rate, more auctions per update, or opponent snapshots (fictitious-play style: sample opponents from past policy checkpoints).
- **Exploration noise blurs bunching:** always report both deterministic and stochastic evaluations.
- **[Rev 2] Exploration noise inflates exploitability:** a random policy pays a utility cost for its own noise. With fixed entropy 0.01 (F7), that cost alone can keep ε above the convergence threshold forever. This is why the convergence rule uses the deterministic profile (Section 8.6).
- **Reward scale:** utilities are in [0, 1], so no reward normalization is needed beyond advantage normalization.

---

## 8. Measurements

All metrics are computed on a held-out **evaluation set**: ≥10⁶ auctions, with policies frozen, under both reserve conditions in the proportions of the run's p_high. Also compute a balanced evaluation (50/50) for the tests. Evaluate every 50 updates to trace dynamics.

### 8.1 Paper-style bunching shares
For bin width δ:
```
P_high(δ) = Pr(bid ∈ [r_high, r_high + δ] | reserve = r_high, bid ≥ r_high)
P_low(δ)  = Pr(bid ∈ [r_high, r_high + δ] | reserve = r_low,  bid ≥ r_high)
```
- δ ∈ {0.001, 0.0025, 0.005, 0.01, 0.02}. The primary δ is 0.01, matching the paper.
- Two-proportion z-test of H₀: P_high = P_low.
- Report the difference and a 95% CI.

### 8.2 Atom share **[Rev 3]**
**Near-atom share (primary):**
`A = Pr(|bid − r_high| < τ | reserve = r_high, bid ≥ r_high)`, with τ = 1e-4.

**Exact-atom share (diagnostic only):**
`A_exact = Pr(|bid − r_high| < 1e-9 | reserve = r_high, bid ≥ r_high)`

- Wherever this spec says "atom", "atom share" or "exact atom" in a classification rule, hypothesis, table or figure, it means A.
- The same τ applies to every bidder. For the discrete grid (spacing 0.01), the wrapper bidders, and A0/A1, A and A_exact give the same answer up to A0's ≈ 0.012 near r.
- **Why τ = 1e-4:**
  - A network can't output exactly r/v over a range of values, so A_exact is about 0 for continuous policies even when they bunch.
  - 1e-4 is 100 times finer than the paper's 1-cent bin. That keeps A well apart from P_high(0.01).
  - A0 scores only ≈ 0.012, well below the 0.05 cutoff.
- **Caveat:** A cannot tell a bid at r from a bid at r + ε with ε < τ. Economically these are the same: both tie or nearly tie at the reserve. For the continuous version of H8, also report where the learned bids sit relative to r (median of bid − r_high among bids within 0.01 of r_high).
- Continuous policies: compute in deterministic mode (Beta mean, Section 7.2). **[Rev 2]** Also report it under the Beta mode as a diagnostic column.
- Discrete policies: compute in both modes.
- Report A and A_exact side by side; classify with A only.

### 8.3 Bin-width scaling exponent
- Fit log P_high(δ) = a + κ·log δ over the δ grid.
- An atom gives **κ ≈ 0**. Equilibrium-like concentration gives **κ ≈ 0.5**.
- Report κ with a bootstrap CI.

### 8.4 Tail test: two-sample Kolmogorov–Smirnov
Let X = {bids > r_high + 0.01 in r_high auctions} and Y = {bids > r_high + 0.01 in r_low auctions}. With empirical CDFs F̂_X(t) = (1/n_X)·#{x ≤ t} and F̂_Y(t) likewise, the statistic is

```
D = sup_t | F̂_X(t) − F̂_Y(t) |
```
That is, the **largest vertical gap between the two empirical CDFs**, in [0, 1]. Under H₀ (same distribution), √(n_X n_Y/(n_X+n_Y))·D converges to the Kolmogorov distribution.

- Compute with `scipy.stats.ks_2samp(X, Y)`. Report D, the p-value, n_X, and n_Y.
- **Caveats:**
  - With large n, tiny D values are "significant." Always report D as an effect size.
  - Also run the test on **paper-sized subsamples** (n = 5,000 per group, 200 bootstrap draws). Report the rejection rate at α = 0.05.
  - Discrete bids create ties, which makes KS conservative. Note this for A3-disc.
  - **[Rev 2]** For wrapper-based bidders (A2, A2b, A2-r0), equal tails hold by design (Section 6). Report the test, but mark it as a code check in the tables.
- Heuristic prediction: don't reject (D small). Equilibrium prediction: reject. The tails differ in this symmetric model because the reserve enters b* at every v.

### 8.5 Distance to benchmark bid functions
- On a grid v ∈ [r, 1] (step 0.005) for each r, compute the deterministic learned bid b_π(v, r).
- Report the sup-norm and L² distances to b* (A0) and to the heuristic (A1).
- **[Rev 2]** For A2, A2b and A2-r0, also report the distance from the unclamped π(v) to b̃(v) = ((N−1)/N)·v over v ∈ [0, 1].
- Classify each run as "equilibrium-like," "heuristic-like," or "neither" using a pre-registered threshold. Suggested: closer to one by ≥ 0.01 in L² and passing that benchmark's signature (atom share A > 0.05 for heuristic; κ > 0.3 and tail KS rejecting for equilibrium).

### 8.6 Exploitability (distance from equilibrium play)
For a focal bidder against the *actual* opponent policies:
1. Estimate the CDF H_r(b) of the highest valid opposing bid by Monte Carlo (≥10⁵ draws per r).
2. For each v on a grid, compute best-response utility U*(v, r) = max_b (v − b)·Pr(win | b), over the 1,001-point bid grid, including tie probabilities.
3. Compare it with the policy's utility U_π(v, r).
4. Exploitability ε = E_{v,r}[U* − U_π] under the evaluation mix.

**[Rev 2] Policy mode:**
- **Primary ε (ε_det):** focal bidder and all learned opponents play their deterministic policies (Section 7.2). This is the ε used for the convergence rule, H1, and all classification.
- **Diagnostic ε (ε_stoch):** focal bidder and opponents play their stochastic policies. Report it next to ε_det in the dynamics figure. The gap ε_stoch − ε_det measures what exploration noise costs.
- Fixed benchmark opponents (A0, A1) are deterministic in both cases.

Convergence criterion: **ε_det** < **1e-4** **[Rev 4]** and stable for 200 updates. (A heuristic population has ε_det ≈ 6.6e-4 at p_high = 0.5, so 1e-3 would not separate it from equilibrium.)

### 8.7 Economic outcomes
- Publisher revenue per auction
- Bidder surplus per auction
- Share unsold
- Allocative efficiency: share of auctions where the highest-value bidder wins
- Tie frequency

Report everything relative to A0 and A1.

### 8.8 Statistical protocol
- **Seeds:** ≥5 per configuration (10 for the primary configurations).
- **Reporting:** report mean ± 95% CI across seeds. Report a per-seed classification table.
- **Pre-registration:** commit metric definitions and classification thresholds before running the grid, in `PREREGISTRATION.md`. **[Rev 2]** This includes the deterministic-mode choice (Beta mean; categorical arg-max) and the ε_det convergence rule.

---

## 9. Experimental factors (what we vary)

| Factor | Levels | Primary? |
|---|---|---|
| F1: learner observes r | yes (A3) / no (A3-blind) / at bid time only, trained with reserves (A2) / at bid time only, trained without reserves (A2-r0) **[Rev 2]** | ✓ |
| F2: action space | continuous Beta / discrete 1-cent grid | ✓ |
| F3: share of high-reserve auctions p_high | 0.5 / 0.95 (the paper's real mix) | ✓ |
| F4: number of bidders N | 2 / 4 / 6 | secondary |
| F5: opponents | shared self-play / independent self-play / fixed A0 / fixed A1 / mixed | secondary (fixed A0 is a required sanity check) |
| F6: value distribution | Uniform[0,1] / "ad-like": with prob 0.5 v = 0, else lognormal scaled so r_high sits near the 60th–70th percentile of positive values; reserves recomputed (r_high revenue-optimal by grid search, r_low = 0.2·r_high) | secondary |
| F7: entropy schedule | annealed to 0 / fixed 0.01 | secondary |
| F8: initialization | near-truthful / random | secondary |
| F9: mechanism | first-price / second-price (control: no shading, so no bunching expected from shading) | control |

**Primary grid:** A3, A3-blind, A2, A3-disc × p_high ∈ {0.5, 0.95}, with N = 4, uniform values, shared self-play: 8 configurations × 10 seeds. **[Rev 2]** Plus A2-r0 × 10 seeds, trained once per seed (no reserve in training) and evaluated under both p_high mixes, for 9 training configurations in total. Run benchmarks A0 and A1 under the same evaluation. Then add secondary factors one at a time.

---

## 10. Hypotheses (with what would falsify each)

| ID | Hypothesis | Supported if | Falsified if |
|---|---|---|---|
| **H0** | The test battery reproduces Section 5.3 for A0 and A1 | All values within Monte Carlo error | Any mismatch. Fix the code before proceeding |
| **H1** | PPO's best response to fixed A0 opponents recovers b* | Sup distance < 0.02 (deterministic mode) and **ε_det** < **1e-4** **[Rev 2, Rev 4]** | Persistent gap. Tune PPO before proceeding |
| **H2** | Full PPO in self-play (A3) converges to equilibrium-like play: atom share A → equilibrium level (≈ 0.012, below 0.05), κ ≈ 0.5, tail KS rejects | All three hold at convergence in ≥ 8/10 seeds | A persistent atom at r. **This would be a headline result**: learning does not remove bunching |
| **H3** | A3 shows a *transient* bunching phase early in training that decays | Atom share rises then falls over updates | No transient, or bunching that never decays (then see H2) |
| **H4** | A rare low-reserve condition (p_high = 0.95) slows A3's convergence and prolongs bunching | Time-to-convergence and area under the atom-share curve are larger at 0.95 than 0.5 | No difference |
| **H5** | **A reserve-blind learner (A3-blind) bunches at r_high under both reserves**, because without seeing r the best pooled bid for values just above r_high is r_high. So its spike does **not** disappear | P_high ≈ P_low and both are large | The spike disappears. Implication if supported: the paper's disappearing spike implies real bidders **do** observe r but use it only as a clamp |
| **H6** **[Rev 2]** | **(a)** The reserve-free two-stage learner (A2-r0) learns π(v) ≈ b̃(v) and so reproduces the heuristic's atom and κ ≈ 0, losing publisher revenue relative to A0 close to the analytic −3.9%. **(b)** The reserve-trained two-stage learner (A2) also shows an atom and κ ≈ 0, but its π(v) is pulled toward r_high (as in H5), so its atom size and revenue loss differ from A1's | (a) L² distance from π to b̃ < 0.02, atom within 0.03 of 0.333, κ CI includes 0, revenue loss within 1 percentage point of −3.9%. (b) Atom > 0.05, κ ≈ 0, and π measurably different from b̃ (L² ≥ 0.02); report the size and sign of the revenue gap to A1 | (a) π(v) does not approach b̃, or the atom or revenue is far off. (b) No atom, or π ≈ b̃ (A2 then behaves like A2-r0). **The tail KS is not part of H6:** for wrapper bidders it does not reject by design and is reported only as a code check |
| **H7** | A discrete bid grid (A3-disc) raises the atom share A relative to continuous A3 | Higher atom at convergence | No difference |
| **H8** | A best-responder to A1 opponents bids at r_high + one grid step, so it "passes" the paper's 1-cent bin test without having an atom at r | Learned mode at r_high + 0.01 | Learned mode elsewhere |
| **H9** | Low capacity (A5) alone does not produce an atom | Atom ≈ 0 | Atom > 0.05 |

---

## 11. Milestones and acceptance criteria

1. **M1, environment and benchmarks.**
   - Env (including no-reserve mode **[Rev 2]**), A0, A1, metrics.
   - Unit tests pass.
   - Section 5.3 table reproduced (H0).
   - Second-price control: truthful bidding shows no bunching beyond valuation mass.
2. **M2, PPO sanity.**
   - Best response to A0 (H1), continuous and discrete.
   - Best response to A1 (H8).
3. **M3, primary grid.**
   - 9 training configurations × 10 seeds (Section 9) **[Rev 2]**.
   - Dynamics logged.
   - H2–H7 evaluated.
4. **M4, secondary factors and A4/A5.**
5. **M5, report** (Section 12).

Pause after M1 and after M2 and summarize results before continuing, since later stages depend on these checks passing.

---

## 12. Outputs

### Figures (each mirrors a figure in the paper where noted)
1. **Bid functions:** v vs bid, coloured by reserve condition, learned policy overlaid on b* and the heuristic (cf. paper Fig. 1).
2. **Bid histograms** with 1-cent bins, high vs low reserve, vertical line at r_high (cf. Fig. 2).
3. **Tail CDFs** above r_high + 0.01 for the two reserve conditions (cf. Fig. 3).
4. **δ-sweep:** log P_high(δ) vs log δ for A0, A1, and learned bidders, with reference slopes 0 and 0.5.
5. **Training dynamics:** atom share A **[Rev 3]**, P_high(0.01), tail-KS D, ε_det and ε_stoch **[Rev 2]**, and revenue vs update step (mean ± CI across seeds).
6. **Revenue / surplus / efficiency** bar chart across bidder types.

### Tables
- **A "Table 3"-style table:** one row per bidder type × configuration, with columns P_low, P_high, two-proportion test, atom share A (mean; mode as a diagnostic), A_exact (diagnostic) **[Rev 3]**, κ, KS D, KS p, subsample rejection rate, and classification. **[Rev 2]** Mark KS columns for wrapper bidders as "by design".
- **Economic outcomes table.**

### Write-up
`REPORT.md` (≤ 6 pages): setup, benchmark verification, results per hypothesis, the bin-width finding, limitations, next steps.

---

## 13. Repository layout

```
reserve-bunching-rl/
  README.md
  SPEC.md                 # this document
  PREREGISTRATION.md
  REPORT.md
  pyproject.toml          # numpy, scipy, torch, pandas, pyarrow, matplotlib, pyyaml, pytest, tqdm
  configs/                # one YAML per experiment; all factors explicit
  src/rbrl/
    env.py                # vectorized FPA/SPA with per-auction reserve and a no-reserve mode; supports per-bidder reserves for later
    benchmarks.py         # b* (closed form + numeric integral), heuristic, thresholds
    bidders/
      fixed.py            # A0, A1, truthful
      ppo.py              # actor-critic, Beta and categorical heads
      hedge.py            # A4
      wrappers.py         # censor wrapper for A2 / A2b / A2-r0
    metrics.py            # 8.1–8.7
    exploitability.py
    train.py              # python -m rbrl.train configs/xxx.yaml
    evaluate.py
    plots.py
  tests/
  results/<run_id>/       # config.yaml, metrics.json, eval.parquet, checkpoints/, figures/
```

Conventions:
- Every run is reproducible from its config and seed.
- Log git commit hash with each run.
- CPU-only must work; GPU optional.

---

## 14. Phase 2 (design for, don't build yet)

- **Asymmetric types.** Two bidder types with different value distributions and type-specific reserves r₁ < r₂ (the paper's Regular DSP / Rebroadcaster structure). The equilibrium is in Kotowski (2018), *Theoretical Economics*. No closed form exists; benchmark against a numerical solver, or use exploitability alone.
- **Intermediary bidders.** A "Rebroadcaster" runs its own first-price sub-auction among k internal bidders and submits one bid upstream, with alternative upstream bidding rules: reserve-blind clamp, pass-through minus commission, or reserve-aware shading. Question: which rules reproduce the paper's finding that intermediaries fail the tail test more often?
- **Estimator robustness.** Apply the paper's structural estimation (invert bids to values assuming the heuristic) to simulated data where true values are known, and measure bias in recovered valuations and counterfactual revenue.
- **Learning seller.** The exchange learns the reserve (e.g. a bandit over r) while bidders learn.

Keep `env.py` general enough (per-bidder reserves, bidder groups) that these can be added without a rewrite.

---

## 15. Notes on Amazon AuctionGym

Repository: https://github.com/amazon-science/auction-gym (Apache-2.0; KDD 2023).

**What it has:**
- `src/AuctionAllocation.py` with `FirstPrice` and `SecondPrice` allocation
- `src/Bidder.py` with `TruthfulBidder`, `EmpiricalShadedBidder`, `ValueLearningBidder`, `PolicyLearningBidder`, and `DoublyRobustBidder`. These learn a bid-shading factor; PolicyLearning and DoublyRobust use off-policy estimators.
- Embedding- and CTR-based value models, with JSON configs.

**What it lacks for this project:**
- No reserve price support
- Values are tied to CTR models and item embeddings, which we don't need

**Recommendation:** build the standalone environment above, which is simpler and fully controlled. Cite AuctionGym in the README as related work. Optionally port `PolicyLearningBidder` as an additional learned baseline, retaining the Apache-2.0 notice for any copied code. Don't take a dependency on its pinned requirements (old versions).

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Impression | One ad showing; the item auctioned |
| Reserve (r) | Minimum acceptable bid |
| Production / monitoring reserve | High, revenue-optimized reserve used most of the time / low reserve used at random to observe bids (here: r_high / r_low) |
| Bid shading | Bidding below value in a first-price auction |
| Candidate bid b̃(v) | The heuristic's reserve-blind bid |
| ṽ | Value above which the heuristic stops bunching (b̃(ṽ) = r) |
| Bunching / mass point / atom | Positive probability on one exact bid (here, the reserve) |
| Bayes–Nash equilibrium | Strategy profile in which each bidder's strategy is a best response given beliefs about others' values |
| Exploitability ε | Average utility a bidder could gain by best-responding instead of following its policy. ε_det: with deterministic policies (primary); ε_stoch: with stochastic policies (diagnostic) |
| KS statistic D | Largest vertical gap between two empirical CDFs (Section 8.4) |
