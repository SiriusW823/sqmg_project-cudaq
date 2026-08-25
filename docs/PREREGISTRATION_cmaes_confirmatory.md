# Pre-registration — CMA-ES versus the QPSO family at full budget

**Written and committed before any confirmatory data was collected.**
The git commit of this file timestamps the analysis plan.

---

## 1. Background

An eight-algorithm comparison at M=64 and the full 9,664-evaluation budget
(n = 5 paired seeds, `SQMG/experiments/07_eight_algorithms_M64/`) produced this
ranking:

| Rank | Algorithm | Median V×U | SD |
|---|---|---|---|
| 1 | QPSO | 0.9750 | 0.0096 |
| 2 | CMA-ES | 0.9710 | 0.0055 |
| 3 | RR-QPSO | 0.9640 | 0.0169 |
| 4 | Differential Evolution | 0.9160 | 0.0187 |

Friedman χ² = 28.114, p = 0.00009 establishes that the algorithms differ, but
n = 5 makes the pairwise tests structurally incapable of significance: the
smallest attainable paired-Wilcoxon p is 0.0625, and 0.4375 after Holm
correction across seven comparisons.

The observation that motivates this study is that **CMA-ES — a standard evolution
strategy with no quantum-inspired component — matches RR-QPSO** (Δ = −0.0010,
RR winning 1/5) while showing roughly one third the spread (SD 0.0055 vs 0.0169).
If that holds, the quantum-inspired machinery is not what produces the
performance.

## 2. Hypotheses (pre-specified, confirmatory)

Let `F(a, s)` be the best-so-far V×U reached by algorithm `a` under seed `s` at
exactly 9,664 evaluations.

- **H1 (primary, equivalence):** CMA-ES and RR-QPSO are equivalent within
  ±0.02 V×U. Tested by TOST on the paired differences.
- **H2 (secondary, superiority):** CMA-ES ≠ RR-QPSO, two-sided paired Wilcoxon.
- **H3 (secondary, variance):** CMA-ES has lower run-to-run variance than
  RR-QPSO. Tested by Fligner–Killeen on the two groups.

H1 is the primary endpoint because the interesting claim is *equivalence*, not
superiority — "a conventional method does just as well" is what would undercut
the quantum-inspired framing, and equivalence must be tested directly rather
than inferred from a non-significant difference test.

## 3. Design

| Item | Value |
|---|---|
| Arms | `cmaes`, `qpso`, `rr_qpso` (ablate=none) |
| Seeds | **10–19** (n = 10) — disjoint from the pilot's 0–4 |
| Particles M | 64 |
| Budget | 9,664 evaluations (T = 150) |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for both QPSO variants |
| Objective | V × U |
| Blocking | Paired: seed `s` fixes both the optimizer RNG and the shot RNG |
| Nodes | DGX101, DGX102 excluded (drained / ~14× slower) |

**Why all three arms are re-run rather than reusing existing data.**
Paired blocking requires every arm to share the same seeds. Existing QPSO and
RR-QPSO data at 9,664 evaluations uses seeds 0–9, and CMA-ES was selected for
this test *because it performed well in the pilot*, so its pilot estimate is
inflated by selection. Running all three arms on fresh seeds 10–19 keeps the
pairing intact and makes the test genuinely out of sample. The cost is 30 runs
instead of 10; the alternative would repeat the error the confirmatory ablation
was designed to avoid.

## 4. Analysis plan

1. **Primary:** TOST on `Δ = F(cmaes) − F(rr_qpso)` with equivalence bounds
   ±0.02, α = 0.05.
2. **Secondary:** two-sided paired Wilcoxon on the same differences; Fligner–Killeen
   for variance.
3. **Multiplicity:** Holm–Bonferroni across H2 and H3. H1 is the primary endpoint
   and is not corrected.
4. **Effect size:** Cliff's δ and a 95% bootstrap CI (20,000 resamples) on the
   median difference, reported regardless of significance.
5. **Context:** QPSO is included as a third arm so the three-way ranking can be
   reported with a Friedman test, but no pairwise claim involving QPSO is
   pre-registered.
6. Runs failing to reach 9,664 evaluations are excluded pairwise, and the number
   excluded is reported.

## 5. Decision rules

- TOST p < 0.05 → **CMA-ES and RR-QPSO are equivalent** within ±0.02.
- TOST p ≥ 0.05 and Wilcoxon p_holm < 0.05 → one is better; report which.
- Both non-significant → inconclusive at this sample size; report the CI and stop.

**No additional seeds will be added and no alternative endpoint substituted to
reach significance.**

## 6. What each outcome would mean

**Equivalence confirmed.** A standard evolution strategy with no quantum-inspired
component matches RR-QPSO at the paper's own budget. Combined with the finding
that RR-QPSO is itself equivalent to plain QPSO, the reasonable reading is that
performance on this problem comes from using a population-based optimizer at all,
not from the quantum-inspired mechanisms.

**CMA-ES better.** Stronger version of the same conclusion.

**RR-QPSO better.** The first positive evidence for the method in this study; it
would need to be weighed against the null results from the α-alignment,
long-horizon, and ablation experiments.

## 7. Interim monitoring

Progress will be checked during the run for infrastructure reasons only
(stragglers, node failures). **No hypothesis test will be run on partial data.**
