# Pre-registration — population methods versus the reference Bayesian optimizer

**Written and committed before the Ax/BoTorch environment was working and before
any data was collected.** The git commit of this file timestamps the plan.

---

## 1. Background

The strongest positive result in this project is that population-based methods
separate completely from Bayesian optimization: at the full budget both QPSO and
RR-QPSO win 10/10 paired seeds with Cliff's δ = +1.000 and p_holm = 0.0039.

That result rests on **our own** BO implementation, which differs from the one
the published comparison used (Ax/BoTorch `Models.GPEI`) in three ways, each
favouring the reference:

| | Reference | `optimizers/bayesopt.py` |
|---|---|---|
| Kernel | ARD Matérn 5/2 (per-dimension length scales) | isotropic Matérn |
| GP training points | all observations | capped at 400 |
| Sobol initialization | 5 trials | 128 |

At D = 134 an isotropic kernel asserts that all 134 parameters share one length
scale. Our BO plateaus at 0.6925 where the published figure is 0.902. A
controlled test showed the training cap explains at most 22% of the gap
(Δ = +0.0125, 5/10, p = 0.156), leaving the kernel as the prime suspect.

**Until this is resolved the population-versus-BO margin can only be reported as
directional.** This study exists to make it quantitative — in whichever
direction the data goes.

## 2. Hypotheses (pre-specified)

Let `F(a, s)` be the best-so-far V×U at exactly 1,984 evaluations for algorithm
`a` under seed `s`.

- **H1 (primary):** `F(qpso, s) > F(ax_bo, s)`. One-sided paired Wilcoxon,
  α = 0.05.
- **H2 (secondary):** `F(rr_qpso, s) > F(ax_bo, s)`. Same test.
- **H3 (secondary):** the reference BO outperforms our hand-written BO,
  `F(ax_bo, s) > F(bo, s)`. This measures how much of the earlier gap was our
  implementation's weakness rather than a property of Bayesian optimization.

H3 is the diagnostic that matters for the paper's integrity. If it is strongly
supported, the earlier BO numbers must be reported as an artefact of our
implementation and the eight-optimizer table corrected.

## 3. Design

| Item | Value |
|---|---|
| Arms | `ax_bo`, `qpso`, `rr_qpso`, `bo` (our implementation, re-run) |
| Objective | `vu` (unconstrained) |
| Seeds | **200–209** (n = 10) — disjoint from every prior experiment |
| Particles M | 64 for the population arms; BO variants are sequential |
| Budget | 2,000 evaluations, enforced identically for all four arms |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for both QPSO variants |
| Blocking | Paired: seed fixes both optimizer and shot RNG |

**Ax configuration**, matched item by item to upstream `constrained_bo.py`:
`GenerationStep(SOBOL, num_trials=5)` then `GenerationStep(GPEI, num_trials=-1)`,
`torch_dtype=float64`, parameters `x1..x134` ranged [0,1], one trial at a time.

Two deliberate deviations, both recorded in `optimizers/axbo.py`:

1. **`torch_device` is CPU.** `ax-platform` 0.4.3 pulls torch 2.14 with CUDA 13,
   which conflicts with the cluster's CUDA 12.2 driver and with cuda-quantum
   0.7.1. Ax therefore runs in a separate environment with CPU torch, calling the
   GPU molecule evaluator by subprocess. The upstream code has this exact
   fallback (`"cuda" if torch.cuda.is_available() else "cpu"`), so it is a
   supported configuration, not an invention. GP fitting is slower; results are
   unchanged.
2. **`random_seed` is the experiment's seed, not the fixed 42 upstream uses.**
   A paired design needs each seed to give an independent run; a fixed seed would
   make all ten runs identical.

Budget is 2,000 rather than 9,664 because Ax GPEI on CPU refits a GP with up to
2,000 points in 134 dimensions at every step; the full budget is not affordable.
2,000 is the budget at which our BO already reaches 0.6650 and the population
methods 0.88–0.89, so the gap is well established there.

## 4. Analysis plan

1. **H1–H3:** one-sided paired Wilcoxon, Holm–Bonferroni across the three.
2. **Effect size:** Cliff's δ and 95% bootstrap CI (20,000 resamples) on the
   median difference, for each comparison, significant or not.
3. **Configuration check (mandatory, precedes any test):** confirm from the run
   logs that Ax used GPEI with an ARD kernel, that no training cap was applied,
   and that exactly 5 Sobol trials preceded the GP steps. If the configuration
   does not match, the batch is void and no hypothesis is tested — the same class
   of check that caught the voided HBA/HBD batch.
4. Runs not reaching 1,984 evaluations are excluded pairwise and counted.

## 5. Decision rules

- **H3 supported** → our BO was materially weaker than the reference. The
  eight-optimizer table and every BO comparison are re-reported using `ax_bo`,
  and the earlier BO figures are labelled an implementation artefact.
- **H3 not supported** → the plateau is a property of Bayesian optimization on
  this problem, not of our code. The existing threat-to-validity paragraph is
  resolved and the margin becomes quantitative.
- **H1/H2** determine whether the population advantage survives against a
  properly configured BO, which is the claim the paper rests on.

Whatever the outcome, **the result is reported.** If the population advantage
shrinks or disappears against the reference implementation, that is the finding,
and the paper's principal positive claim changes accordingly.

## 6. Interim monitoring

Infrastructure and the §4.3 configuration check only. No hypothesis test on
partial data.
