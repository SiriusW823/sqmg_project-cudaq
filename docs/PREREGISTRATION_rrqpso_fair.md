# Pre-registration — RR-QPSO versus QPSO, on a fair harness

**Written and committed before any run of `rr_qpso2` was submitted.** The git
commit of this file timestamps the plan.

---

## 1. Why this study exists

This project has reported, across several experiments, that RR-QPSO is
equivalent to plain QPSO and at n = 20 significantly worse. `docs/FAIRNESS_AUDIT.md`
shows that comparison carried four confounds, **all favouring plain QPSO**:

| # | Confound | Size |
|---|---|---|
| 1 | Sobol scramble hard-coded to `seed=0`, so every RR-QPSO run in the project started from the **same** population while QPSO got one per seed | effective n = 1 for anything initialization-dependent |
| 2 | `T = max_evals // M − 2` plus an OBL batch left RR-QPSO **104 fewer search evaluations** at the 1,000 budget (832 vs 936) | **+0.021 to +0.026 V×U**, measured from the convergence curves |
| 3 | Different α schedules: linear for QPSO, cosine + Gaussian perturbation + stagnation boost for RR-QPSO | not quantified |
| 4 | Different codebases; the RR-QPSO side carried four mechanisms the manuscript never describes | not quantified |

Confound 2 alone is **two to five times** every RR-versus-QPSO difference the
project has reported. The correct statement is not that RR-QPSO has no
advantage — it is that **the two have never been fairly compared.**

All four are now fixed (commit `6bb272c`). This study is the fair comparison.

## 2. What changed in the code

`optimizers/qpso.py::QPSO` now implements the manuscript's §III as three flags:

| Flag | Manuscript | Default in `qpso` | Default in `rr_qpso2` |
|---|---|---|---|
| `sobol_init` | §III-B | off | on |
| `rank_refined` | §III-C, Eq. (9) | off | on |
| `fitness_guided` | §III-D, Eq. (10)(11) | off | on |

`rr_qpso2` **is** `QPSO` with the three flags on. The two arms therefore share
one α schedule, one budget accounting (`while True` to `BudgetExhausted`, no OBL
batch), one RNG seeding scheme and one initialization path. The Sobol scramble
seed now comes from `base._sobol`, which derives it from the experiment seed.

Verification before submission (`SQMG/scripts/verify_rrqpso2.sh`, 12 checks, all
passing): flags-off is **bit-identical** to the previous QPSO so existing results
still reproduce; four seeds give four distinct Sobol starts; both arms consume
**exactly** the same number of evaluations; Eq. (9) and Eq. (10)/(11) match hand
calculation, including that the indicator functions disable an elite whose
complementary metric is below τ.

The legacy `rr_qpso` (the `AESOQPSOOptimizer` wrapper) remains registered so the
existing experiments stay reproducible. It is **not** an arm in this study.

## 3. Hypotheses (pre-specified)

Let `F(a, s)` be best-so-far V×U at exactly 2,000 evaluations for algorithm `a`
under seed `s`.

- **H1 (primary, confirmatory):** `F(rr_qpso2, s) > F(qpso, s)`.
  One-sided paired Wilcoxon, α = 0.05.
- **H2 (secondary):** the same at 500 evaluations, testing whether any advantage
  is budget-dependent — the manuscript's mechanism story (broader early
  coverage) predicts an *early* advantage.

**Component ablation (secondary, exploratory).** Three arms each removing one
component from `rr_qpso2`, to locate any effect H1 finds. Reported with Holm
correction across the three, and labelled exploratory regardless of outcome —
this study is not powered for them.

## 4. Design

| Item | Value |
|---|---|
| Arms | `qpso`, `rr_qpso2`, and 3 ablation arms (`sobol_init=0`, `rank_refined=0`, `fitness_guided=0`) |
| Objective | `vu` (unconstrained) |
| Seeds | **300–329** (n = 30) — disjoint from every prior experiment |
| Particles M | 64 |
| Budget | 2,000 evaluations, identical for every arm |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for **all** arms, same linear form |
| ρ, w_RR, w_V, w_U, τ | 0.015, 0.70, 0.15, 0.15, 0.5 — the manuscript's values |
| Blocking | Paired: seed fixes the optimizer RNG, the Sobol scramble and the shot RNG |

n = 30 rather than 10. The audit's power analysis for the constrained-objective
H1 put 80% power at n ≈ 43 for d_z ≈ 0.43; n = 30 gives 80% power for
d_z ≈ 0.53. **This study is therefore powered to detect a large effect only.** A
null result at n = 30 does not establish equivalence, and §6 says what may and
may not be concluded from one.

## 5. Analysis plan

1. **Harness check (mandatory, precedes any test).** From the CSVs, confirm that
   (a) the two arms consumed the same number of evaluations, and (b) the first
   M rows differ across seeds **for both arms**. If either fails, the batch is
   void and no hypothesis is tested. This is the check that would have caught
   the original problem, and it is now mandatory rather than optional.
2. **H1, H2:** one-sided paired Wilcoxon. H1 and H2 are separate questions at
   separate budgets and are **not** corrected against each other; H2 is labelled
   secondary and interpreted only if H1 is supported.
3. **Effect size:** Cliff's δ, Cohen's d_z, and a 95% bootstrap CI (20,000
   resamples) on the median difference — reported whether or not significant.
4. **Ablation:** Holm across the three arms, exploratory.
5. Runs not reaching 2,000 evaluations are excluded pairwise and counted.

## 6. Decision rules

- **H1 supported** → RR-QPSO has a real advantage over plain QPSO on this
  benchmark that the previous harness was hiding. The project's null result is
  retracted and the manuscript's central claim is supported, at this budget,
  with this effect size.
- **H1 not supported** → report as a null **with its confidence interval and the
  detectable effect size**. This does *not* establish that RR-QPSO is equivalent
  or worse; n = 30 only rules out large effects. The honest conclusion would be
  "no large advantage; smaller effects untested".
- **Stopping rule.** Whatever the outcome, this line stops here. No adding
  seeds, no switching to the constrained objective, no changing the endpoint or
  budget. A follow-up requires a new pre-registration with n fixed in advance
  from this study's observed effect size.

## 7. What would make this study wrong

Stated in advance, so it is not invented afterwards:

- **The fix could itself be the confound.** Giving RR-QPSO a seed-dependent
  Sobol start changes its initialization distribution, not just its variance. If
  H1 flips to supported, the honest reading includes "RR-QPSO's advantage
  depends on initialization" as a live alternative to "RR-QPSO works".
- **Only the unconstrained objective is tested.** The manuscript's multi-objective
  claim is out of scope.
- **M = 64 only.** No interaction between particle count and the RR mechanism.
- **1,000 shots, not 5,000.** Absolute values are not comparable to the
  manuscript; the paired comparison is unaffected.
- **`rr_qpso2` is not `rr_qpso`.** It implements the manuscript's §III and
  deliberately drops four mechanisms the legacy runner had. If H1 is supported
  here but the legacy arm lost, the difference is those four mechanisms — an
  interesting result, and a different one.

Whatever the outcome, **it is reported.** The previous null was retracted for
being unfair, not for being inconvenient; the same standard applies to a
positive result.
