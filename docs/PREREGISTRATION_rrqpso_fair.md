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

---

## 8. Amendment — 2026-10-12, before any data from the corrected batch

Committed before the corrected batch is submitted. The commit timestamp is the
record.

### 8.1 The first batch is void

The first execution (150 runs, `results_rrfair/`, 2026-09-17 to 09-21) ran
with every hyperparameter override silently dropped. `RRQPSO2.__init__` took
`(*args, **kwargs)`, so `run_experiment.py`'s signature-based filter saw no
named parameters (fixed in `7c09dcb`, which also makes an undeliverable override
fatal). The logs show `⚠ rr_qpso2 不接受 alpha 參數，忽略覆寫` for every
`rr_qpso2` arm. Consequences:

- the three ablation arms ran the same configuration as `rr2` — their 30 values
  at 500 evaluations are identical;
- `rr_qpso2` used α ∈ [0.5, 1.0] while `qpso` used [0.3, 1.2].

**§5.1 did not catch this.** Both of its conditions passed (30 distinct starts
per arm, equal evaluation counts). It checks the harness, not the
configuration. §8.4 adds the missing check.

### 8.2 What had been seen when this was written

The void batch was analysed with the §5 script on 2026-10-12:
H1 @2,000 Δ = −0.0045, 95% CI [−0.0165, +0.0185], 14/16, p = 0.55;
H2 @500 Δ = +0.033, CI [+0.002, +0.061], 21/9, p = 0.013. The `qpso` arm of the
void batch ran exactly the configuration it will run again, so its 30 values for
seeds 300–329 are known.

This does not touch any analysis choice: hypotheses, endpoints, tests, decision
rules and the stopping rule were fixed in `894ad76` and are unchanged. The seeds
are also kept at 300–329 — switching seeds now would itself be a choice made
after seeing data.

### 8.3 Corrected batch — design identical to §4 except

| Item | Value |
|---|---|
| Code | `539ac61` or later; includes `7c09dcb` plus the harness robustness fixes `4fbb29b` and `c6be0e5`. `SQMG/scripts/verify_rrqpso2.sh` (six sections, including the new override checks) passed on 2026-10-12 |
| Output | `results_rrfair2/` — the void batch stays untouched in `results_rrfair/` |
| Submission order | all `qpso` and `rr_qpso2` jobs (H1/H2) before the three ablation arms, so the primary comparison completes first |
| Everything else | as §4: 5 arms, seeds 300–329, M = 64, 2,000 evaluations, 1,000 shots, α ∈ [0.3, 1.2] for all arms, ρ/w/τ at the manuscript's values |

### 8.4 Added mandatory check — §5.1(c), configuration

Before any test, from each run's console log: the logged override line must
contain `alpha_max: 1.2` and `alpha_min: 0.3` for every arm, and the arm's own
flag (`sobol_init`, `rank_refined` or `fitness_guided` = False) for each
ablation arm. In addition the four `rr_qpso2`-family arms must **not** be
identical at 500 evaluations. If any condition fails, the batch is void.

### 8.5 Deadline and interim looks

The batch overlaps a manuscript deadline (2026-10-20), which is exactly when an
early look is tempting. Rules:

- No H1/H2 statistic, and no per-arm best-so-far value, is computed until all
  60 `qpso` and `rr_qpso2` runs have reached 2,000 evaluations and §5.1(a–c)
  pass. Monitoring is limited to job counts, evaluation progress and the
  §5.1 checks (`SQMG/scripts/rrfair2_status.sh` prints no fitness values).
- If H1 is not complete when the manuscript must be finalised, the manuscript
  describes this study as pre-registered and in progress, with no partial
  result.
- Ablation arms are analysed once, after all 150 runs complete.

The stopping rule of §6 is unchanged.

### 8.6 Analysis code — 2026-10-13, before any statistic from the corrected batch

`benchmark/rrfair2_analyze.py` is the §5 analysis that was run on the void batch
(§8.2), moved into the repository and split into the two stages §8.5 already
fixes: `--stage h1` (§5.1 checks on `qpso` and `rr_qpso2`, then H1 @2,000 and
H2 @500) and `--stage ablation` (all five arms, only after all 150 runs). In the
`h1` stage the ablation arms are neither summarised nor tested; they enter only
the §8.4 "not identical at 500" check, which compares values for equality and
prints none. Tests, endpoints, effect sizes, bootstrap seed and decision rules
are unchanged.

State when this was committed (from `rrfair2_status.sh`, which prints no
fitness): `qpso` 30/30 and `rr_qpso2` 30/30 runs at 2,000 evaluations, §5.1(c)
configuration check 0 errors in all five arms; ablation arms 10/90 complete.

### 8.7 Ablation arms held — 2026-10-13 18:45

After H1 was analysed (§8.6), the 70 ablation jobs still pending were put on
`scontrol hold` so the equal-hardware study (`PREREGISTRATION_anytime_vs_bo.md`)
can run first; the 10 ablation runs already executing were left alone. This
changes only *when* the ablation arms run. Their configuration, seeds, budget and
the single analysis after all 150 runs (`--stage ablation`) are unchanged. Held
job IDs are listed on the cluster in `results_rrfair2/held_ablation_jobs.txt`;
`SQMG/scripts/rrfair2_release_ablation.sh` releases them.
