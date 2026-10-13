# Pre-registration — RR-QPSO versus the reference BO on equal hardware and equal wall-clock time

**Committed before any confirmatory run.** The git commit of this file timestamps
the plan. Code: `a26e4f1` (time budgets, checkpoints, fresh shot noise, paired
launcher, test scoring), verified by `benchmark/verify_anytime.py`.

---

## 1. Why this study exists

Every equal-*evaluation* comparison in this project favours the reference BO
(Ax/BoTorch GPEI, the configuration of Chen et al.): it is more sample-efficient
(`RESULTS.md` §7h). The question this study asks is different, and is the one a
user with a fixed GPU allocation faces:

> **On the same hardware, given the same wall-clock time, which optimizer
> finds better QMG parameters?**

### 1.1 What had been seen (exploratory pilot, disclosed in full)

Re-reading existing runs by their logged `elapsed_s` — exp. 11 (`ax_bo`, n = 10,
1,000 evaluations) against exp. 2 + 7 (`rr_qpso`/`qpso`, n = 20, 9,664
evaluations); every run on a 4×V100 allocation, 1,000 shots, one fixed shot seed
per run; cross-experiment and **unpaired**. Best-so-far V×U, median over runs:

| Wall-clock | ax_bo | rr_qpso | qpso |
|---|---|---|---|
| 4 h | **0.851** | 0.728 | 0.749 |
| 8 h | **0.878** | 0.825 | 0.842 |
| 12 h | 0.909 | 0.885 | 0.913 |
| 16 h | 0.911 | 0.947 | **0.961** |
| 24 h | 0.924 | 0.964 | **0.970** |

At 4 h BO leads (Cliff's δ = −0.70, Mann–Whitney p = 0.002); at 24 h the
population methods lead (δ ≈ +0.41/+0.43, p = 0.075/0.061). The mechanism is
visible in the logs: GPEI averages 163 s per evaluation (its GP cost grows with
history; 764 evaluations by 24 h) against ~7 s for the population methods (9,664
by ~19 h). Script: `SQMG/submission_qcnc2027/exploratory_wallclock/`.

**The hypotheses below were derived from this pilot.** That is why the
confirmatory study uses new seeds, a changed harness (fresh shot noise,
independent test scoring, same-node pairing) and fixes both directions of the
prediction in advance.

## 2. Hypotheses

Let `S_h(a, s)` be the **test score** (§4.7) of algorithm `a`'s incumbent at
wall-clock checkpoint `h` hours, seed `s`.

- **H1 (primary):** `S_24(rr_qpso2, s) > S_24(ax_bo, s)` — the population method
  reaches better parameters within one day.
- **H2 (primary, the other half of the prediction):**
  `S_4(ax_bo, s) > S_4(rr_qpso2, s)` — BO is ahead early.

One-sided paired Wilcoxon signed-rank in the stated directions, Holm across
H1 and H2, α = 0.05.

H2 is pre-registered so that BO's early advantage is reported as a finding, not
left out. A claim of "RR-QPSO beats BO" without the qualifier "given about a day"
is not permitted by §6 whatever the result.

**Secondary, descriptive, no claim attached:**

- `S_h` at h = 2, 8, 12, 16, 20 (anytime curve), and the first checkpoint at which
  the median paired difference turns positive (crossover).
- Evaluations completed by each arm at each checkpoint.
- Fraction of seeds whose incumbent has `S_h ≥ 0.95`, per checkpoint.
- **E3, attribution** — a `qpso` arm (§3). If `qpso` matches or beats `rr_qpso2`
  at 24 h, any advantage is attributed to the QPSO family, not to the
  rank-refined design.

## 3. Arms

| Arm | Implementation | Configuration |
|---|---|---|
| `rr_qpso2` | `optimizers/qpso.py::RRQPSO2` | M = 64; Sobol init, rank-refined, fitness-guided all on (class defaults); ρ = 0.015, w = 0.70/0.15/0.15, τ = 0.5; α ∈ [0.3, 1.2] via override |
| `ax_bo` | `optimizers/axbo.py` | upstream `constrained_bo.py` configuration: 5 Sobol trials, then GPEI, ARD kernel, every observation in the GP, `max_parallelism = 1` |
| `qpso` (secondary) | `optimizers/qpso.py::QPSO` | M = 64; α ∈ [0.3, 1.2]; flags off |

Batch-parallel BO is **not** tested. The claim is therefore about the BO
configuration of the reference paper, not about BO in general (§9).

## 4. Protocol

### 4.1 Problem
9 heavy atoms, D = 134, chemistry constraint applied, CUDA-Q 0.7.1 /
cuStateVec, V100.

### 4.2 Shots
**1,000 per evaluation during optimization** — the setting of the pilot and of
all eleven earlier experiments, so the pilot's predictions apply. Evaluation cost
sets where the crossover falls; with Chen et al.'s 10,000 shots the simulation
would dominate BO's GP cost for longer and the crossover would move later. This
is a stated limitation, not a tuned choice.

### 4.3 Fresh shot noise
Evaluation `k` of a run with seed `s` uses shot seed `s × 1,000,000 + k`
(`evaluator.shot_seed_for`); no seed repeats within or across runs of
different seeds. The two arms of a pair share `s`, so their k-th evaluations
share a seed — common random numbers, not reuse.

### 4.4 Hardware and pairing
Each pair is **one exclusive node** (`benchmark/paired_anytime.slurm`,
`--exclusive --gres=gpu:8 --no-requeue`), both arms started together:

- `rr_qpso2`: GPUs 0–3, four persistent evaluation workers;
- `ax_bo`: worker on GPU 4, GP on GPU 5 (GPUs 6–7 idle) — the same use of a
  4-GPU share as in the pilot. Sequential BO cannot use more workers; that is
  the property under test.

Node speed and load therefore cancel within each pair. Workers are recycled
every **18 minutes** in both arms (`QMG_RECYCLE_MINUTES=18`); batch-count
recycling would have charged BO one respawn per 10 evaluations.

### 4.5 Budget
**24 h of wall-clock per arm** (`--time_budget_h 24`), measured from the start of
the optimizer, including all of its own computation (GP fitting, acquisition).
`MAX_EVALS = 12,288` (= 64 × 192) sets the α-schedule horizon of the population
arms at the pilot's throughput (~7 s/evaluation × 4 workers ≈ 24 h); an arm that
reaches it earlier keeps its final incumbent for the remaining checkpoints. Job
limit 26 h. A run may not resume (the code refuses).

### 4.6 Incumbent
At each checkpoint h ∈ {2, 4, 8, 12, 16, 20, 24} h: the parameter vector with the
highest **observed** V×U among evaluations completed by h. A batch that is still
running at h does not count. Same rule for every arm.

### 4.7 Test score — the endpoint
`benchmark/test_score.py`: each incumbent re-evaluated on **K = 16 test seeds**,
`2,000,000,000 + j`, j = 0..15 — a range disjoint from every optimization seed —
at **5,000 shots** (the protocol of Chen et al. Fig. 3a). `S_h` = mean V×U over
the 16. Identical seeds for every arm and checkpoint. Test evaluations are not
charged to any budget.

### 4.8 Seeds
**n = 24, seeds 500–523**, disjoint from every earlier experiment.

## 5. Analysis plan

1. **Mandatory checks**, per pair, before any test:
   (a) both arms ran on the same host (`host` in each `_summary.json`);
   (b) configuration: `ax_bo` log carries the audit lines for ARD with 134 length
   scales and exactly 5 Sobol trials; `rr_qpso2` log shows the override
   `{'alpha_max': 1.2, 'alpha_min': 0.3}` and no rejected parameter;
   (c) both logs show `fresh shot noise` and strictly increasing per-batch seed
   ranges;
   (d) both logs show the 24 h time budget, and all seven checkpoint files exist
   for both arms;
   (e) no arm crashed before 24 h for a reason other than the budget.
   A pair failing any check is excluded and counted. **If more than 4 of 24 pairs
   are excluded, the batch is void** and no hypothesis is tested.
2. H1, H2: one-sided exact paired Wilcoxon, Holm. Effect sizes Cliff's δ and
   d_z, 95% bootstrap CI on the median paired difference (20,000 resamples),
   reported whether or not significant.
3. Secondary quantities: same statistics, no multiplicity correction, no claims.
4. E3 (`qpso`) runs in separate half-node jobs, so its comparisons are
   cross-node and descriptive only.
5. Every reported number is regenerated from the CSVs, checkpoint files and
   `test_scores/*.csv` by one script.

**Power.** The pilot's unpaired δ ≈ +0.4 at 24 h corresponds to d ≈ 0.7; n = 24
gives roughly 75–80% power for a one-sided test, ignoring the gain from pairing.
At 4 h the pilot effect (δ ≈ −0.70) is large. A null H1 does not establish
equivalence.

## 6. Decision rules — fixed in advance

| Outcome | What may be written |
|---|---|
| H1 and H2 supported | "On the same 8-GPU node, the reference BO finds better parameters during the first hours; given 24 h, RR-QPSO finds better ones (median S_24 difference, CI)." Both halves, always together. |
| H1 only | RR-QPSO is better at 24 h; BO's early lead is not established at this n. |
| H2 only | BO is better early; no advantage of RR-QPSO within 24 h is established. The "higher ceiling" claim is withdrawn. |
| Neither | Null on both, reported with CIs. |
| E3: `qpso` ≥ `rr_qpso2` at 24 h | Every RR-QPSO statement above is rewritten as a statement about the QPSO family. |

**Stopping rule.** No added seeds, no change of budget, checkpoints, K, shots or
endpoint after data. A follow-up needs a new pre-registration.

## 7. Smoke test (before the confirmatory batch; not confirmatory)

One pair, seed 999, `TIME_BUDGET_H=0.25`, checkpoints 0.05, 0.1, 0.25 h, output
`results_anytime_smoke/`, then `test_score.py` with K = 2. Purpose: exercise every
check in §5.1 on real hardware. Its outcome is recorded in an amendment before the
first confirmatory job starts and is excluded from all analysis.

## 8. Compute and schedule

24 pairs × ≤ 26 h on exclusive nodes; five usable `nchc` nodes (DGX102 excluded)
→ five waves ≈ 5–5.5 days. The batch is queued behind `rrfair2` (expected to
finish ≈ 2026-10-15), so results are expected ≈ 2026-10-20–21 — after the QCNC
2027 deadline. The secondary `qpso` arm (24 half-node jobs) is queued after the
pairs and may be cancelled without affecting H1/H2. Test scoring: ~3 GPU-hours
per pair-checkpoint set, ≈ 3 h on the full partition.

## 9. What would make this study wrong

- **One problem instance** (N = 9, unconstrained V×U).
- **Shots.** At 10,000 shots per evaluation the crossover would move later.
- **No batch BO.** A batch-parallel GPEI could recover the late regime. The claim
  is about the reference paper's configuration.
- **Incumbent rule.** Observed-best under fresh noise selects partly on luck;
  identical for both arms and removed from the endpoint by test scoring, but BO
  might gain from a model-based recommendation that is not used here.
- **Ax version.** `ax-platform 0.4.3` / BoTorch 0.12 (ARD RBF), not the exact
  upstream environment.
- **The harness differs** from every earlier experiment (fresh noise, time
  budget); results are not pooled with them.

---

## 10. Amendment — 2026-10-12, smoke tests, before the confirmatory batch

Three smoke pairs, seed 999, 0.25 h budget, partition `large` (DGX107–109 idle),
outputs `results_anytime_smoke{,2,3}/`. Excluded from every analysis.

1. **Job 111058 (code `70c56f8`) found a bug.** `CHECKPOINT_HOURS=0.05,0.1,0.25`
   reached both arms as `0.05`: `sbatch --export` splits on commas. Fixed in
   `0380a19` (colon separator; the launcher now refuses a comma). Every other
   check passed.
2. **Job 111059 (`0380a19`) passed every §5.1 check.** Same host (DGX108); fresh
   shot noise with contiguous, increasing seed ranges in both arms; the α override
   `{'alpha_max': 1.2, 'alpha_min': 0.3}` present; Ax audit lines for ARD with 134
   length scales and exactly 5 Sobol trials; the 0.25 h budget honoured — the
   0.25 h checkpoint counts 256 of 320 RR-QPSO evaluations and 38 of 39 BO
   evaluations, the straddling batch excluded as §4.6 specifies.
   `test_score.py` (job 111062, K = 2) ran on its checkpoints; test scores sit
   below the observed values (e.g. BO 0.495 observed → 0.420 test at 0.25 h),
   as expected from selection on noisy draws and from 5,000-shot scoring.
3. **Throughput differs from the pilot.** On an exclusive node the population arm
   ran ≈ 1,280 evaluations per hour after a ~5-minute worker start-up — about
   2.5× the pilot, whose runs shared nodes two jobs at a time. BO's early rate
   matched the pilot (38 evaluations in the first ~0.25 h vs a pilot median of 38
   at 0.22 h).

   Consequence: §4.5's `MAX_EVALS = 12,288` was chosen so that the population
   arms' α schedule would span ≈ 24 h *at the pilot's rate*; at the measured rate
   it would end near hour 10 and leave the arm idle for the rest of the budget,
   and on a slow node it would never finish annealing. **Changed, before any
   confirmatory data:** both population arms anneal α linearly over the
   wall-clock budget (`--alpha_by_time`, commit `5840f17`), and
   `MAX_EVALS = 999,936` so that time is the only budget. Job 111060 (`5840f17`)
   ran with the new schedule (`α 排程 : 依牆鐘` in the log) and passed the same
   checks; `verify_anytime.py` (29 checks) confirms monotone annealing from
   ≈ α_max to α_min and bit-identity when no budget is set.
4. **Expectation, updated; hypotheses unchanged.** With the population arm ~2.5×
   faster than in the pilot, the pilot's crossover (~12–14 h) is probably an
   overestimate, so H2 (BO ahead at 4 h) is less certain than §1.1 suggests.
   Annealing over 24 h keeps α high early, which works the other way. No
   hypothesis, endpoint, test, seed set or decision rule is changed.

§4.5 is superseded as to `MAX_EVALS` and the α horizon; everything else stands.

## 11. Analysis code, committed before data (2026-10-12)

`benchmark/anytime_analyze.py` implements §5 exactly: the per-pair checks
(a)–(e) with the exclusion and void rules, H1/H2 with Holm, the effect sizes
and bootstrap CIs (seed 20261012), and the descriptive table. It was exercised
only on synthetic files. The confirmatory pairs are jobs **111063–111086**,
submitted 2026-10-12 20:54, queued behind `rrfair2`.

## 12. Schedule change — 2026-10-13 18:45, before any pair has started

The 70 pending ablation jobs of the `rrfair2` batch were held
(`PREREGISTRATION_rrqpso_fair.md` §8.7) so that the 24 pairs (jobs
111063–111086) start as soon as the 10 running ablation runs free their nodes,
expected from about 21:00 the same evening. Nothing in this study's design,
hypotheses, analysis or exclusion rules changes. The `qpso` arm (E3) and the
held ablation jobs are submitted/released only after all 24 pairs have started,
so they cannot take a node a pair is waiting for.

Context, stated so it is on record before any pair data exists: the corrected
`rrfair2` batch found no advantage of `rr_qpso2` over `qpso` at 2,000
evaluations (H1 not supported, Δ median −0.012, 95% CI [−0.023, +0.008]). The
arms of this study are kept as registered; a positive H1 here is therefore read
as a property of the QPSO family as instantiated by `rr_qpso2`, not of the
rank-refined mechanism.
