# Errata and corrections — RR-QPSO manuscript (arXiv:2607.10284v1)

**Status:** three issues, found 2026-09-11 while checking the manuscript's
optimizer comparison against the multi-seed data collected since submission.
Two are factual errors in the figures and text. One is a provenance gap that
only the authors can close.

Every number below is traceable to a file on the cluster. Where a claim rests on
the *absence* of evidence, that is said explicitly.

---

## Summary

| # | Issue | Severity | Affects |
|---|---|---|---|
| 1 | The RR-QPSO vs QPSO comparison is single-seed, and the reported gap is half of one seed-to-seed standard deviation | High | Fig. 2, §IV-A, Conclusion |
| 2 | Fig. 2's "QPSO" and "QPSO + Sobol Init." bars are swapped; the text's conclusion about Sobol is backwards | High | Fig. 2, §IV-A |
| 3 | The BO baseline V×U = 0.902 has no matching run in the codebase | Critical | Abstract, Fig. 2, Table I, §IV-B, §IV-C, Conclusion |

Separately, new data (§4 below) shows the BO baseline itself was
under-configured, which changes the manuscript's principal claim.

---

## 1. The RR-QPSO advantage over QPSO is not distinguishable from noise

### What the manuscript says

Fig. 2 reports RR-QPSO at V×U = 93.0% against QPSO + Sobol at 91.4%, a gap of
**+1.6 percentage points**, from a single run each. §IV-A already hedges this
("in this single-seed run", "benchmark-level evidence rather than a complete
statistical ranking") and the Conclusion lists multi-seed evaluation as future
work.

### What the data shows

That future work is now done. Ten paired seeds (210–219), both optimizers, same
budget, same shot count, seed fixing both the optimizer and the shot RNG:

| seed | QPSO | RR-QPSO | difference |
|---|---|---|---|
| 210 | 0.8350 | 0.7400 | **−0.0950** |
| 211 | 0.7630 | 0.8290 | **+0.0660** |
| 212 | 0.8210 | 0.7120 | **−0.1090** |
| 213 | 0.7420 | 0.7840 | +0.0420 |
| 214 | 0.7770 | 0.7590 | −0.0180 |
| 215 | 0.7310 | 0.7740 | +0.0430 |
| 216 | 0.7890 | 0.7740 | −0.0150 |
| 217 | 0.7930 | 0.8080 | +0.0150 |
| 218 | 0.7670 | 0.8170 | +0.0500 |
| 219 | 0.7710 | 0.8010 | +0.0300 |

Median difference **+0.0225**, RR-QPSO wins **6/10**, two-sided Wilcoxon
**p = 0.71**. At the 205-evaluation endpoint: +0.0120, 6/10, p = 0.75.

The seed-to-seed standard deviation is **0.0322** for QPSO and 0.0361 for
RR-QPSO. **The manuscript's +0.016 gap is 0.50 standard deviations.** A single
draw cannot separate an effect that size from chance: seed 211 alone would have
produced a +0.066 result, seed 212 alone a −0.109 result.

Note this does **not** show RR-QPSO is worse. The sign of the median difference
agrees with the manuscript. What fails is the certainty, not the direction.

One caveat and why it does not rescue the claim: the new runs use 1,000 shots
where the manuscript used 5,000, so absolute V×U levels are not comparable.
The variance argument is unaffected — a separate check found V×U to be
bit-identical across 24 repeats and 20 shot seeds for fixed parameters, so
essentially all of the between-seed spread comes from search stochasticity,
which more shots do not reduce.

### Suggested correction

Report the ten-seed result with its interval and p-value, and state that the
single-seed gap in the original Fig. 2 is within run-to-run variation. The
honest claim is that RR-QPSO is **not distinguishable from** QPSO on this
benchmark, not that it improves on it.

---

## 2. Fig. 2's QPSO bars are swapped, and the Sobol conclusion is reversed

### What the manuscript says

> QPSO without Sobol initialization gives a similar product of 90.5%, mainly due
> to higher uniqueness. Adding Sobol initialization increases the product to
> 91.4% in this single-seed run, suggesting that low-discrepancy initialization
> improves initial coverage but is not sufficient by itself.

### What the run logs say

Read from the log headers, not the directory names:

| Run | Init strategy (from log) | Final V×U |
|---|---|---|
| `results_qpso_nosobol/console.log` | no Sobol line — random init | **0.9144** |
| `results_qpso_pure/console.log` | `[v10.3] 初始化策略: Sobol scrambled (seed=0, 確定性)` | **0.9046** |

Both have `[v10.3] OBL Phase 0: ✗ 關閉`, M = 64, T = 150, 9,664 evaluations,
5,000 shots — identical except for initialization.

So the run **without** Sobol scored 91.4% and the run **with** Sobol scored
90.5%. Fig. 2 assigns those two numbers to the opposite labels, and the text
then draws the opposite conclusion: in this single-seed run, Sobol
initialization **lowered** V×U by 0.0098.

(Both differences are, by issue 1's argument, within noise. The point here is
not that Sobol hurts — it is that the figure and the sentence state the reverse
of what the run produced.)

### Suggested correction

Swap the two bars, and replace the sentence. Given issue 1, the accurate
statement is that Sobol initialization made no distinguishable difference in a
single run, and that its effect was not separately measured across seeds.

---

## 3. The BO baseline of V×U = 0.902 has no traceable run

### What the manuscript says

§IV-A: "The BO baseline is re-run under the same implementation and evaluation
protocol rather than taken directly from the literature." Fig. 2 reports
V = 94.2%, U = 95.7%, V×U = 90.2%. The abstract, Table I's framing, and the
Conclusion all quote 0.902 as the comparison point for 0.930 and 0.942.

### What the codebase contains

Every Bayesian-optimization result on the cluster, best V×U over all seeds:

| Directory | Budget | Best V×U |
|---|---|---|
| `results_bofull` | 9,664 | **0.7150** |
| `results_prod` | 9,664 | 0.7120 |
| `results_bouncap` | 2,000 | 0.7210 |
| `results_prod_oldpath_0731_2253` | 5,376 | 0.5810 |

The only 0.90-range BO figure anywhere in the repository is a hard-coded
constant:

```
qpso_optimizer_ae.py:80   - BO baseline: V=0.955, U=0.925, V×U=0.8834 (N=9, 5000 shots)
qpso_optimizer_ae.py:642  BO 基準（Chen 2025）: 0.8834（V=0.955, U=0.925, N=9, 5000 shots）
run_qpso_qmg_cudaq.py:1394 "✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
```

That is the **literature** value from Chen et al. (ref. [22]), not a re-run —
and it is 0.8834 with V = 95.5 / U = 92.5, which matches neither the
manuscript's 0.902 nor its V = 94.2 / U = 95.7.

**I could not locate any run, log, or result file that produces V×U = 0.902 with
V = 94.2% and U = 95.7%.** This is an absence of evidence, not proof that no
such run happened — the run may predate the current directory layout or have
been cleaned up. But as things stand the number cannot be reproduced from the
repository, while the manuscript asserts it was re-run in-house.

### Why this is the critical one

0.902 is the denominator of the manuscript's headline claim ("an absolute
improvement of 0.040, or 4.0 percentage points, over BO"). It also anchors the
multi-objective comparison in §IV-C (BO at V×U = 43.8%), whose provenance should
be checked the same way.

### Action required

Locate the run that produced 0.902, or re-run the BO baseline and report the
number obtained. If the figure came from the literature, §IV-A's sentence must
be corrected to say so.

---

## 4. New evidence: the BO baseline was under-configured

This is not an erratum — it is a new result that bears on the manuscript's main
claim, obtained under a pre-registration committed before the data
(`docs/PREREGISTRATION_axbo.md`).

The manuscript positions RR-QPSO as a population-based alternative that
outperforms BO. Our own BO implementation (`optimizers/bayesopt.py`) differs
from the reference used by the original QMG study (Ax/BoTorch `Models.GPEI`) in
three ways, each favouring the reference: an isotropic kernel instead of ARD, a
400-point cap on GP training data, and 128 Sobol initial trials instead of 5.
At D = 134 an isotropic kernel asserts that all 134 parameters share one length
scale.

Porting the reference configuration faithfully and running it against both
population methods on ten paired seeds at a matched budget:

| Arm | median V×U @ 960 evals | @ 205 evals |
|---|---|---|
| **Ax/BoTorch GPEI (reference BO)** | **0.9290** | **0.7990** |
| RR-QPSO | 0.7790 | 0.4540 |
| QPSO | 0.7740 | 0.4705 |
| Our BO (`bayesopt.py`) | 0.6360 | 0.5415 |

Pre-registered tests, Holm-corrected:

- **H3, reference BO > our BO: supported.** Median +0.2960, 95% CI
  [+0.248, +0.308], Cliff's δ = +1.000 (10/10), p_holm = 0.0029. Our BO
  implementation was materially weaker, and every BO comparison resting on it
  must be re-reported.
- **H1/H2, population > reference BO: not supported, and reversed.** δ = −0.840
  (QPSO) and −0.860 (RR-QPSO) at 960; δ = −1.000 for both at 205.

205 evaluations is the budget the original QMG code actually uses
(`constrained_bo.py` loops `range(num_iterations + 5)`;
`analysis_figures/01_bo_analysis.ipynb` sets `num_iterations = 200`), so the
reference BO wins decisively even at its own budget.

**Limitations, stated plainly.** The reference BO's lead shrinks with budget
(+0.33 at 205, +0.155 at 960). The manuscript's comparison is at 9,664
evaluations, and Ax GPEI cannot be run there: measured GP cost is 1.85·n^0.60
seconds per evaluation on a V100, which extrapolates to roughly 760 hours for a
single seed. So we cannot test the crossover directly. What we can say is that
the reference BO reaches 0.929 in 960 evaluations, a level the population
methods need roughly ten times more evaluations to approach — but that
cross-experiment comparison is not budget-matched and not pre-registered, so it
is descriptive only.

---

## Provenance

| Claim | Source |
|---|---|
| Ten-seed QPSO / RR-QPSO comparison | `results_axbo1k/`, jobs 87579–87618 |
| Statistical analysis | `results_axbo1k/stats/analysis.txt` |
| Single-seed QPSO runs | `results_qpso_nosobol/console.log`, `results_qpso_pure/console.log` |
| BO result inventory | `results_bofull/`, `results_prod/`, `results_bouncap/` |
| Hard-coded BO constant | `qpso_optimizer_ae.py:80,642`, `run_qpso_qmg_cudaq.py:1394` |
| Ax/BoTorch study design | `docs/PREREGISTRATION_axbo.md` (committed before data) |
| Upstream BO budget | `~/qmg_upstream/analysis_figures/01_bo_analysis.ipynb` |
