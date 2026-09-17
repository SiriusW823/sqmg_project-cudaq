# Fairness audit — is RR-QPSO's deficit an artefact of the harness?

**2026-09-17.** The Ax study found that this project's BO baseline was
under-configured in three ways that all favoured the reference, which overturned
the headline result. That audit was applied to a comparator. This document
applies the same audit to **our own method**, because the same failure mode is
just as available in the other direction.

Every finding below is verified against the CSVs on disk, not inferred from
reading the code. Reproduce with `tools/audit_rrqpso_fairness.py`,
`tools/audit_rrqpso_magnitude.py`, `tools/audit_rrqpso_scope.py`.

**Conclusion up front: yes. Four confounds, all favouring plain QPSO. Two are
serious enough that the RR-QPSO-versus-QPSO comparison, as run, cannot answer
the question it was built to answer.** The comparison against the reference BO
is not affected and still stands.

---

## 1. Every RR-QPSO run in the project starts from the same population

`optimizers/qpso.py:147`

```python
sampler = qmc.Sobol(d=self.D, scramble=True, seed=0)   # 固定 seed=0
```

The Sobol scramble seed is hard-coded to 0 and never derived from the experiment
seed. Verified from the first 64 rows of every CSV — the fingerprint is the
(validity, uniqueness) pair of each initial particle:

| Experiment | arm | runs | distinct starting populations |
|---|---|---|---|
| `02_longhorizon` | rr_qpso | 10 | **1** |
| `09_h1_replication` | qpso | 45 | 45 |
| | rr_qpso | 45 | **1** |
| `10_diversity` | qpso | 30 | 30 |
| | rr_qpso | 30 | **1** |
| `12_cmaes_confirmatory` | qpso | 10 | 10 |
| | rr_qpso | 10 | **1** |
| `11_reference_bo` | qpso | 10 | 10 |
| | rr_qpso | 10 | **1** |

**QPSO gets an independent initial population per seed. RR-QPSO gets the same
one every time, in every experiment ever run.**

The seed still varies RR-QPSO's update RNG, so the runs are not identical. But
for anything that depends on where the search starts, RR-QPSO's effective sample
size is **1**, not 10 or 45. The paired design assumes the seed indexes an
independent replicate for both arms; that assumption is false for one arm.

The direction of the resulting bias is not knowable in advance — it depends
entirely on whether that one fixed basin is good or bad. What *is* knowable is
that RR-QPSO's every reported result is conditional on a single draw of its
initialization, and no amount of added seeds fixes that.

This is also why the manuscript's Sobol-initialization claim could never have
been seed-averaged.

## 2. RR-QPSO is given fewer evaluations to search with

Two separate deductions, both in `optimizers/qpso.py`:

- `T = max_evals // M - (2 if obl else 1)` — RR-QPSO stops at the largest
  multiple of M within budget; plain QPSO loops to `BudgetExhausted` and lands on
  the budget exactly.
- OBL spends an extra full batch in Phase 0 that plain QPSO does not spend.

At the 1,000-evaluation budget with M = 64:

| | total evaluations | initialization | left for search |
|---|---|---|---|
| QPSO | 1,000 | 64 | **936** |
| RR-QPSO | 960 | 128 (Phase 0 + OBL) | **832** |

**RR-QPSO searches with 104 fewer evaluations — 11% less.**

### What 104 evaluations are worth

Measured from the convergence curves rather than assumed:

| arm | best V×U @832 | @960 | gain | per 100 evals |
|---|---|---|---|---|
| QPSO | 0.7415 | 0.7740 | +0.0325 | **+0.0254** |
| RR-QPSO | 0.7525 | 0.7790 | +0.0265 | **+0.0207** |

So the handicap is worth roughly **+0.021 to +0.026 in V×U**. Set that against
the gaps it has to explain:

| Gap | Size | Handicap relative to it |
|---|---|---|
| RR-QPSO − QPSO @960 | **+0.0050** (RR ahead) | **4–5× larger than the gap** |
| RR-QPSO − QPSO, long-horizon n = 20 | **−0.0090** | **2–3× larger than the gap** |
| RR-QPSO − reference BO @960 | −0.1500 | 14% of the gap |

**At short budgets the handicap is several times larger than every RR-QPSO
versus QPSO difference this project has reported.** At the 9,664 budget the
totals are equal (151 × 64 divides exactly) and the curve is near plateau, so
only the OBL batch applies there and its marginal value is small — the
long-horizon result is less affected, though confounds 1, 3 and 4 still apply.

The gap to the reference BO is an order of magnitude beyond what this explains,
and plain QPSO — which carries none of these handicaps — loses to the reference
BO by a similar margin. **The BO conclusion is unaffected.**

### A knock-on effect inside the ablation

`--ablate obl` removes OBL, which by the formula above also *returns* a batch of
64 evaluations to that arm. At the ablation's ~2,000-evaluation budget, 64
evaluations are worth roughly +0.016 by the table above, against a measured
"+0.0250 when OBL is removed". **Most of OBL's apparent cost is the budget it
hands back.** The conclusion (no component reaches significance) is unchanged,
but that effect size is inflated.

## 3. The two arms use different α schedules

| | schedule |
|---|---|
| `optimizers/qpso.py:72` (QPSO) | linear: `α_max − (α_max − α_min)·(t / T_est)` |
| `qpso_optimizer_ae.py::_get_alpha` (RR-QPSO) | cosine + Gaussian perturbation + stagnation boost |

Experiment 1 aligned the *endpoints* to [0.3, 1.2], which removed the largest
confound, but the trajectories between those endpoints still differ. Whatever
that difference is worth, it is not the rank-refined mean-best update.

## 4. The two arms are different codebases

```
qpso     → optimizers/qpso.py::QPSO            (v12 framework, ~40 lines)
rr_qpso  → qpso_optimizer_ae.py::AESOQPSOOptimizer  (legacy runner, ~800 lines)
```

`AESOQPSOOptimizer` enables by default four mechanisms that the manuscript's
method section (§III) never describes:

| Mechanism | Default |
|---|---|
| Stagnation reinitialization | `stagnation_limit=12`, `reinit_fraction=0.25` |
| Cauchy heavy-tailed mutation | `mutation_prob=0.15` |
| Paired exploration step | `pair_interval=4` |
| Opposition-based learning | `obl=True` |

So "QPSO versus RR-QPSO" is not a single-variable ablation of the rank-refined
mean-best. It is one implementation against another, differing in the RR term
**and** four undocumented mechanisms **and** the α schedule **and** the search
budget **and** the initialization protocol.

The within-`--ablate` experiments (`03_ablation_pilot`,
`04_ablation_confirmatory`) do not have this problem: both arms run the same
code, so those results remain the cleanest component evidence in the project,
subject only to the OBL budget effect in §2.

---

## What this changes

**It does not rescue RR-QPSO.** No result here shows the method works; the
confounds could equally be masking a deficit as manufacturing one.

**It does retract a claim.** This project has been reporting that RR-QPSO is
equivalent to, and at n = 20 significantly worse than, plain QPSO. That
comparison carries four confounds, all pointing the same way, and at short
budgets the largest of them is several times the measured effect. The accurate
statement is:

> **RR-QPSO and plain QPSO have not been fairly compared.** The question the
> project set out to answer is still open.

This is the same error found in the BO baseline, committed in the opposite
direction, and it was found by the same audit. That is the argument for running
the audit on your own method and not only on the comparator.

## What a fair test requires

1. Seed the Sobol scramble from the experiment seed — a one-line change, and the
   single most important one.
2. Run both arms through one code path, with the rank-refined term as a flag,
   so the ablation is genuinely single-variable.
3. Use one α schedule for both arms.
4. Match the **search** budget, not just the total — either charge both arms for
   initialization equally, or run with OBL off.
5. Fresh seeds, pre-registered, with the endpoint and decision rule fixed before
   data.

Items 1–4 are all fixes to the harness. Until they are made, adding seeds only
measures the harness more precisely.
