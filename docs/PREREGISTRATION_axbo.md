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

---

## 7. Amendment — 2026-09-05, before any confirmatory data

Recorded after the smoke test (`results_axsmoke`, seed 999, 12 evaluations) and
before the confirmatory batch was submitted. Two corrections.

### 7.1 The reference kernel is ARD RBF, not ARD Matérn 5/2

§1's table describes the reference kernel as "ARD Matérn 5/2". The smoke test
log reports what Ax actually built:

```
[ax_bo][組態稽核] kernel=RBFKernel/RBFKernel  lengthscale 維度=134  D=134  ARD=是
[ax_bo][組態稽核] GP 訓練點數=5  已完成評估=6  訓練上限=無
[ax_bo][組態稽核] 第 0 次評估起改用 generation step：Sobol
[ax_bo][組態稽核] 第 5 次評估起改用 generation step：GPEI（先前 Sobol 共提出 5 個 trial）
```

Matérn 5/2 was BoTorch's `SingleTaskGP` default for years; BoTorch 0.12 — the
version `ax-platform` 0.4.3 pins — replaced it with a dimension-scaled ARD RBF.
So "Matérn 5/2" was my description of an older default, not of the code the
comparison actually runs.

This does not change the study. What §1 identifies as the suspected mechanism is
**isotropic versus per-dimension length scales**, and that contrast is intact:
134 length scales here against one in `optimizers/bayesopt.py`. §4.3's operative
criterion was always "an ARD kernel", and the log satisfies it. The correction is
recorded because the background section stated a fact that turned out to be
wrong, and it should not be quietly repaired later.

The other two §4.3 conditions are met and now logged rather than inferred:
exactly 5 Sobol trials precede the first GP step, and no training cap is applied.

One reading note for the exit line, which in the smoke test was
`{'Sobol': 5, 'GPEI': 8}` for a 12-evaluation budget: it counts trials
*proposed*, and the last proposal is discarded when the budget runs out mid-step.
Proposals therefore exceed completed evaluations by exactly one. The evaluation
count of record is the CSV row count, not this line.

### 7.2 `ax_bo` runs may not be resumed

`BaseOptimizer`'s resume restores the CSV and the evaluation counter, but not
Ax's internal state — completed trials, GP training data, and the position in
the generation strategy all live inside `AxClient`. A resumed run would restart
from 5 Sobol trials with part of the budget already spent, producing neither a
2,000-evaluation GPEI run nor any describable algorithm.

`optimizers/axbo.py` therefore raises rather than resuming, and the `ax_bo` arm
is submitted without `RESUME=1`. A run cut short by the wall clock is excluded
pairwise under §4.4 and counted, exactly as that section already specifies. The
population and `bo` arms keep resume, which is correct for them.

---

## 8. Amendment — 2026-09-05, budget reduced from 2,000 to 1,000

The first batch (jobs 87535–87574) was cancelled and is void. This section
records why the budget changed, and what had been seen at the time, because the
change was made after partial data existed.

### 8.1 The 2,000-evaluation budget was not computable

Measured on the running jobs, `ax_bo`'s cost per evaluation on CPU grows as
**2.41·n^0.79 seconds**, where n is the number of points the GP is conditioned on:

| n | 1–50 | 50–100 | 100–150 | 150–200 | 200–300 |
|---|---|---|---|---|---|
| s/eval (CPU) | 41 | 77 | 111 | 162 | 192 |

Only 8.6 s of that is the molecule simulation. Extrapolated to 2,000 evaluations
this is **≈308 hours per seed**; the 48-hour wall clock reaches roughly 700. Ten
seeds were never going to finish.

§3 justified the 2,000 budget as affordable. That judgement was wrong, and the
smoke test did not catch it: 12 evaluations sit in the flat part of the curve
where the GP overhead is 37 s and reads as a constant. The cost curve should have
been measured before 40 jobs were submitted.

### 8.2 The CPU restriction was removed first

§3's deviation 1 (CPU torch) was not necessary — it described what pip installs
when left alone, not a constraint. Pinning `torch==2.5.1+cu121` matches the
cluster's CUDA 12.2 driver, and `ax-bo` and `cudaq-v071` are separate
environments in separate processes. Upstream runs on GPU, so this **removes**
a deviation from the reference rather than adding one. §3 deviation 1 no longer
applies; only deviation 2 (per-seed `random_seed`) stands.

A pilot run of the real pipeline on a V100 (`results_axpilot`, seed 900) gives
**1.85·n^0.60 s**:

| n | 1–50 | 50–100 | 100–150 | 150–200 | 200–300 |
|---|---|---|---|---|---|
| s/eval (CPU) | 41 | 77 | 111 | 162 | 192 |
| s/eval (GPU) | 22 | 34 | 39 | 47 | 63 |

That is ≈66 h per seed at 2,000 — better, still over the wall clock.

### 8.3 The reference itself uses 205 evaluations

`constrained_bo.py` takes `--num_iterations` and loops
`range(args.num_iterations + 5)`; upstream's own analysis notebook
(`analysis_figures/01_bo_analysis.ipynb`) sets `num_iterations = 200` and slices
`iloc[5:num_iterations+5]`. **The published BO comparison ran 205 evaluations.**

Our pre-registered 2,000 was already 10× the reference's own budget. This makes
the reduction principled rather than a concession to the wall clock.

### 8.4 Amended design

| Item | Was | Now |
|---|---|---|
| Budget (all four arms) | 2,000 | **1,000** |
| Seeds | 200–209 | **210–219** |
| Data directory | `results_axbo/` | `results_axbo1k/` |
| Ax torch device | CPU | **GPU (`cuda:3`)** |

Everything else — arms, objective, M, shots, α schedule, pairing, and the whole
of §2 and §4 — is unchanged. The budget applies identically to all four arms, so
it cannot favour any of them.

**Primary endpoint:** best-so-far V×U at exactly **960** evaluations — see §8.6.
**Secondary endpoint:** best-so-far V×U at exactly 205 evaluations, matching the
published setting. This costs nothing extra — every run logs best-so-far per
evaluation, so the 205 prefix is read from the same CSVs — and it is reported
whether or not it agrees with the primary.

Projected cost at 1,000: ≈23 h per `ax_bo` seed, roughly 2× headroom against the
48-hour limit. The other three arms run 1,000 evaluations in well under an hour.

### 8.5 What had been seen when this was decided

Full disclosure, since the budget was changed with partial data in hand. From
the cancelled batch: `ax_bo` best-so-far 0.8530 at n≈225; `qpso` 0.8780,
`rr_qpso` 0.9100, `bo` 0.6760, each at whatever evaluation count that run had
reached (1,030–2,000, not matched across arms). No hypothesis-relevant paired
comparison was computed, and no test was run.

Seeds 210–219 are disjoint from those runs, so the confirmatory data is drawn
from a search this decision never touched. The cancelled runs are retained in
`results_axbo_void/` and `results_axbo_2k_superseded/` for the cost analysis
above, and are **not** eligible as confirmatory data under any endpoint.

### 8.6 Correction: the primary endpoint is 960, not 1,000

§8.4 first said 1,000. That number is not reachable by every arm, and the error
was mine in writing the amendment.

`RRQPSO` delegates to `AESOQPSOOptimizer` with
`T = max_evals // M - 2`, so it always spends `(T + 2)·M` — the largest multiple
of the population size that does not exceed the budget. With M = 64 that is:

| Budget | RR-QPSO actually uses | Unused |
|---|---|---|
| 9,664 | 9,664 | 0 |
| 2,000 | 1,984 | 16 |
| 1,000 | **960** | 40 |

Plain `QPSO` instead loops until `BudgetExhausted` and lands on the budget
exactly, as do `bo` and `ax_bo`. So at a 1,000 budget RR-QPSO gets 960
evaluations and the other three get 1,000 — a 4% asymmetry, against our own
method.

Reading every arm at **960** removes it: QPSO's, BO's and Ax's extra 40
evaluations are discarded, and all four arms are compared on exactly the same
number. This is the convention §2 already used, where the 2,000-evaluation
budget was read at 1,984 for the same reason.

The correction is structural — 960 follows from M and the budget alone, is fixed
before any arm's outcome at that endpoint was computed, and could not have been
chosen to favour anything. All nine completed RR-QPSO runs ended cleanly
(`完成 評估 960/1000`, no errors); the stop is by design, not a failure.
