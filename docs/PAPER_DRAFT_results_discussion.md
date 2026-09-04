# Draft — Results and Discussion

Working draft for the manuscript, assembled from `docs/RESULTS.md`. Every figure
in the tables is regenerated from the raw CSVs into `figures_method/stats*.json`;
none is transcribed by hand.

**Suggested title.** *Population-based optimization outperforms Bayesian
optimization for quantum molecular generation, while quantum-inspired refinements
do not: a pre-registered evaluation.*

---

# Results

## 1. Experimental protocol

All experiments use the 9-heavy-atom QMG benchmark: a 20-qubit dynamic circuit
controlled by `θ ∈ R^134`, evaluated with CUDA-Q 0.7.1 (cuStateVec) on NVIDIA
V100 GPUs at 1,000 shots per evaluation, scored by the validity–uniqueness
product `V × U`.

Three properties of the protocol are load-bearing and are stated before any
result.

**Budget parity is enforced by construction.** A common base class counts every
objective-function call and raises once the cap is reached, so a sequential
Gaussian-process method and a population method of 64 particles consume exactly
the same number of circuit evaluations. Parity is a property of the code, not a
convention followed by each implementation.

**All comparisons are paired.** A seed fixes both the optimizer's random state
and the shot random state, so a difference is computed within a seed and
between-seed difficulty cancels. Every test is a paired signed-rank test.

**Confirmatory hypotheses were pre-registered.** Five analysis plans — hypotheses,
endpoint, multiplicity correction, decision rule and stopping rule — were
committed to version control before the corresponding data existed. Commit hashes
are given with each result. Two consequences of that discipline appear below: one
batch was voided by its own mandatory validity check, and one line of inquiry was
closed by its own stopping rule rather than extended.

Ten experiments comprising approximately 620 runs are reported. Effect sizes
(Cliff's δ, Cohen's d_z) and bootstrap confidence intervals are given for every
comparison, significant or not.

## 2. Population-based optimization decisively outperforms Bayesian optimization

Eight optimizers were compared at M = 64 and the full 9,664-evaluation budget
(T = 150), n = 5 paired seeds.

**Table 1.** Eight-optimizer comparison at full budget.

| Rank | Optimizer | Median V×U | SD | Mean rank |
|---|---|---|---|---|
| 1 | QPSO | 0.9750 | 0.0096 | 1.40 |
| 2 | CMA-ES | 0.9710 | 0.0055 | 2.00 |
| 3 | RR-QPSO | 0.9640 | 0.0169 | 2.80 |
| 4 | Differential Evolution | 0.9160 | 0.0187 | 3.80 |
| 5 | SPSA | 0.7190 | 0.0143 | 5.00 |
| 6 | Batch BO | 0.6950 | 0.0168 | 6.00 |
| 7 | Sobol random search | 0.5700 | 0.0318 | 7.00 |
| — | Bayesian optimization † | 0.6925 | 0.0139 | — |

Friedman χ² = 28.114, p = 0.00009. † BO was capped at 2,000 evaluations in this
batch; its full-budget figure comes from the dedicated n = 10 study below.

Sobol random search finishing last is the intended sanity check on the harness.
At n = 5 the smallest attainable paired p is 0.0625, and 0.4375 after correction
across seven comparisons, so pairwise significance is structurally unreachable
here; the omnibus test is unaffected. The three optimizers carrying the study's
conclusions were therefore re-tested at n = 10.

**Table 2.** Population methods versus Bayesian optimization, n = 10 paired seeds.

| Budget | BO | QPSO | RR-QPSO | Δ (QPSO − BO) | Cliff's δ |
|---|---|---|---|---|---|
| 500 | 0.5915 | 0.5715 | 0.5925 | −0.0180 | −0.30 |
| 1,000 | 0.6260 | 0.6565 | 0.6685 | +0.0335 | +0.52 |
| 1,984 | 0.6625 | 0.7490 | 0.7380 | +0.0975 | +0.80 |
| 3,000 | 0.6715 | 0.8095 | 0.7835 | +0.1455 | +0.98 |
| 5,000 | 0.6830 | 0.8990 | 0.8645 | +0.2120 | +1.00 |
| 7,000 | 0.6920 | 0.9555 | 0.9410 | +0.2650 | +1.00 |
| 9,664 | 0.6925 | 0.9720 | 0.9655 | +0.2815 | +1.00 |

At the full budget both population methods win **10/10** paired seeds against BO
(p_holm = 0.0039; Cohen's d_z = +13.6 for QPSO, +11.3 for RR-QPSO). Cliff's
δ = +1.000 denotes complete separation: every population run exceeds every BO run.

The budget dependence is itself informative. BO leads below ~500 evaluations —
its sample efficiency is real — then plateaus at 0.6925 while the population
methods continue to 0.97. Batch BO does not improve on sequential BO
(Δ = −0.0070, 3/10, p = 0.139), so the plateau is not a parallelism artefact.

We tested whether the plateau was an artefact of our own implementation. Our BO
caps the GP training set at 400 points for tractability; removing the cap
entirely (n = 10, single-variable comparison) raised the median only from 0.6650
to 0.6765 (Δ = +0.0125, 5/10, p = 0.156), against a gap of +0.217. A sanity check
confirmed the two configurations are identical below 400 evaluations, where the
cap does not bind. The cap explains at most 22% of the gap. Section 8 treats the
remaining threat to this comparison.

## 3. RR-QPSO is equivalent to, and at pooled sample size worse than, plain QPSO

At M = 64 and the full budget with the contraction–expansion schedule aligned to
[0.3, 1.2] for both arms (n = 10):

| | Median | Mean | SD |
|---|---|---|---|
| QPSO | 0.9720 | 0.9689 | 0.0187 |
| RR-QPSO | 0.9655 | 0.9630 | 0.0193 |

Δ = −0.0080, 95% CI [−0.0160, +0.0040], 3/10 wins, p = 0.1211, Cliff's δ = −0.21.
Two one-sided tests reject a difference larger than ±0.02 (**p = 0.0011**), so
this is a positive equivalence result rather than a failure to detect a
difference. Run-to-run variability is likewise indistinguishable (SD ratio 1.03,
Fligner–Killeen p = 0.589), contrary to the stability claim made for the method.

At intermediate budgets RR-QPSO is significantly *worse*: at 4,000 and 5,000
evaluations it loses 0/10 with Holm-corrected p = 0.018 and Cliff's δ ≈ −0.77.

Pooling every full-budget run under identical settings (n = 20) gives
Δ = −0.0090 with a 95% CI of [−0.0160, −0.0040] that **excludes zero**, 4/20 wins,
p = 0.0152, Cliff's δ = −0.175. The equivalence conclusion drawn at n = 10 does
not survive the larger sample; the direction is established, though the magnitude
is small — about 2% of the 0.40 spread separating the best and worst optimizers.

An α-schedule alignment experiment preceded these: RR-QPSO had used α ∈ [0.3, 1.2]
against a baseline at the literature-standard [0.5, 1.0], confounding the
mechanism's effect with the schedule's. Once aligned, plain QPSO led at every
budget. All results reported here use the aligned schedule.

## 4. No individual component survives out-of-sample confirmation

Each RR-QPSO component was disabled in turn (M = 64, 1,984 evaluations, n = 10).

**Table 3.** Component ablation, discovery sample.

| Component removed | Δ median | Wins | 95% CI | d_z |
|---|---|---|---|---|
| Opposition-based learning | +0.0250 | 7/10 | [−0.0250, +0.0485] | 0.513 |
| V–U decoupling | +0.0170 | 7/10 | [−0.0070, +0.0335] | 0.191 |
| AE-weighted mean-best | +0.0130 | 8/10 | [−0.0055, +0.0270] | 0.538 |
| Sobol initialization | +0.0100 | 7/10 | [−0.0110, +0.0250] | 0.344 |
| Mode-collapse recovery | +0.0060 | 6/10 | [−0.0140, +0.0270] | 0.176 |

Friedman across all seven configurations: p = 0.2969. Every point estimate favours
retaining the component and every confidence interval crosses zero; a sign-flip
permutation test on the aggregate gives p = 0.0814. The sample is underpowered
rather than null: 80% power requires n ≈ 30 for the two largest effects and
n ≈ 215–253 for the two smallest.

The two reachable components were then tested on **30 held-out seeds**, disjoint
from the discovery sample, under a pre-registered plan.

**Table 4.** Confirmatory ablation.

| Component | Δ median | Wins | p_holm | d_z (discovery → confirmatory) | Shrinkage |
|---|---|---|---|---|---|
| Opposition-based learning | +0.0025 | 15/30 | 0.131 | 0.513 → 0.342 | ×0.67 |
| AE-weighted mean-best | +0.0110 | 18/30 | 0.075 | 0.538 → 0.297 | ×0.55 |

Neither is supported. Effect sizes shrank to 0.55–0.67× on held-out seeds, the
expected consequence of selecting the two largest effects from an underpowered
sample. Had the discovery sample simply been extended to n = 30 on the same
seeds, AE-weighted mean-best would very likely have reached significance.

## 5. Single-run comparisons are unreliable at the relevant scale

Five fixed parameter vectors were each re-evaluated with 24 independent shot
seeds. All five showed the same directional bias: the value obtained under the
default shot seed sat near the maximum of its 24 draws (z = +2.9 to +3.8),
inflating the reported figure by **+0.011 to +0.027 in V×U**.

This is intrinsic to reporting a maximum over noisy evaluations — the selected
candidate is chosen partly for genuine quality and partly for a favourable draw,
and a single run cannot separate the two. The inflation exceeds the differences
that single-run optimizer comparisons in this problem typically report.

## 6. On a constrained objective the ordering changes, but not in RR-QPSO's favour

A multi-objective variant folds hydrogen-bond acceptor and donor counts into the
fitness. On this landscape (M = 64, 1,984 evaluations, n = 10):

| | Median | SD | Mean rank |
|---|---|---|---|
| RR-QPSO | 0.8263 | 0.0439 | 1.20 |
| QPSO | 0.8152 | 0.0747 | 1.90 |
| CMA-ES | 0.6870 | 0.0502 | 2.90 |

Friedman p = 0.00068. The property term is genuinely satisfied: the recovered
property-closeness of the best solution has a median of 0.9370.

**RR-QPSO outperforms CMA-ES 10/10** (Δ = +0.1346, Cliff's δ = **+0.940**,
p_holm = 0.0020), a pre-registered hypothesis and the only one supported in this
study. CMA-ES falls by 0.22 relative to its unconstrained performance while
RR-QPSO falls by 0.09.

RR-QPSO versus plain QPSO on the same objective gave Δ = +0.0186, 8/10 wins,
p = 0.0801 — reported as null against the pre-registered α, with a power
calculation indicating n ≈ 43. A pre-registered replication on **45 held-out
seeds** returned Δ = **−0.0267**, 15/45 wins, d_z = **−0.312**, with a 95% CI of
[−0.0374, −0.0095]. The effect did not shrink; it changed sign. The interaction
test that would have made any difference objective-specific was also null
(p = 0.4922).

The first execution of this experiment was **voided**: the evaluator truncated the
worker's result tuple, so the property term silently evaluated to zero and the
fitness reduced to a constant rescaling of the unconstrained objective. The
pre-registration's mandatory requirement to report property values caught it —
the recovered closeness was exactly 0.0000 in all 30 runs. The plan was re-run
unchanged on corrected code.

## 7. The mechanisms change search behaviour without improving any endpoint

Instrumenting 55 existing paired runs establishes what the mechanisms do. All
three fire regularly: the V–U decoupling term sits at its cap for 50–77% of
evaluations, mode-collapse recovery triggers ~179 times per long run, and
stagnation reinitialization ~3 times. Their consistent effect is broader
exploration.

**Table 5.** Search behaviour, both objectives.

| | Cells covered in (V,U) space (20×20) | σ_V | σ_U | Evaluations at U < 0.20 |
|---|---|---|---|---|
| RR-QPSO | 323.5 / 400 | 0.1995 | 0.2480 | 6.09% |
| QPSO | 296.0 / 400 | 0.1893 | 0.2282 | 4.60% |

RR-QPSO covers ~10% more of the reachable space and visits more low-quality
regions — the cost side of the same trade, and the reason the mode-collapse guard
exists. Replicated on the constrained objective (272 vs 247 cells).

Because `max V×U` rewards a single parameter vector while a generator delivers a
*set of molecules*, a pre-registered study (n = 30 held-out seeds) measured `D`,
the count of distinct valid SMILES produced across a whole run.

**Table 6.** Set-level diversity.

| | Distinct molecules D | Coverage C | max V×U |
|---|---|---|---|
| RR-QPSO | 308,068 | 271.5 | 0.8600 |
| QPSO | 269,920 | 243.5 | 0.8490 |
| Δ (median) | +23,522 (19/30) | +23.5 | −0.0145 |

RR-QPSO produces more molecules, but not significantly after correction
(p_holm = 0.0667). The second, pre-specified hypothesis — that it does so beyond
what its extra exploration alone explains — fails decisively. Regressing `D` on
coverage across all 60 runs gives `D = 2,148·C − 280,926` (R² = 0.386), so
RR-QPSO's additional 23.5 cells predict ≈ +50,500 molecules; it delivered
+23,522. Controlling for exploration, it produces **fewer** molecules than
expected (Δ = −27,990, 7/30 wins, p = 0.9993). Its exploration is not better
directed; it is less productive per unit of breadth.

One observation is nonetheless usable. Across all seeds QPSO found 3,127,284
distinct molecules and RR-QPSO 3,404,331, of which only 1,535,738 are common:
1,591,546 were found by QPSO alone and 1,868,593 by RR-QPSO alone. The two
methods explore genuinely different regions rather than one subsuming the other,
and their union (≈5.0 M) substantially exceeds either alone.

## 8. Threats to validity

**The Bayesian optimization baseline may be weaker than the reference
implementation.** The published comparison used Ax/BoTorch `GPEI`; ours is a
hand-written Gaussian process differing in three respects that all favour the
reference: an isotropic Matérn kernel rather than an ARD kernel, a 400-point
training cap rather than none, and CPU rather than GPU execution. At D = 134 an
isotropic kernel asserts that all 134 parameters share one length scale, and is
the most likely reason our BO plateaus at 0.69 where the reference reports 0.90.
The training cap was ruled out experimentally (Section 2); the kernel was not.
**The population-versus-BO margin should therefore be read as directional, not
quantitative**, pending a re-run against Ax/BoTorch.

**Shot counts are not comparable across studies.** Uniqueness is
`distinct / valid`; as shots increase the denominator grows roughly linearly
while the numerator saturates, so V×U falls as shots rise. This work uses 1,000
shots against an upstream default of 10,000. Absolute values are not comparable
to previously published figures; all comparisons here are within a fixed shot
count.

**Ceiling effects.** At 1,000 shots both QPSO variants approach the attainable
maximum (≈ 0.97), which may compress the difference between them. A 5,000-shot
replication was not run.

**Seed-set heterogeneity.** Seeds 10–19 proved ~7× more variable than seeds 0–9
under identical settings, because several runs enter a poor basin early and never
leave. Medians are therefore not comparable across seed sets, and paired blocking
is weaker than assumed: seed difficulty correlates only ~0.1 across algorithms,
so a single outlier can dominate a mean at n = 10.

**Single problem instance.** All results concern the 9-heavy-atom benchmark at
M = 64 under two objectives.

---

# Discussion

## What the evidence supports

The results separate cleanly by the comparison being made, and the separation
matters more than any single number.

**The choice of optimizer family is decisive.** Population-based methods separate
completely from Bayesian optimization on this problem — every run of the former
exceeds every run of the latter at the full budget. Bayesian optimization's
sample efficiency is real but confined to the first few hundred evaluations,
after which it plateaus while population methods continue to improve. For a
problem whose evaluations are independent and parallelisable across GPUs, and
whose budget is thousands of evaluations rather than dozens, the sequential
model-based approach is working against both the structure of the problem and the
structure of the hardware.

**Refinements within that family are not.** Ten experiments, ~620 runs, five
pre-registered confirmatory studies, two objectives and two classes of endpoint
find no advantage for the quantum-inspired mechanisms over plain QPSO. This is
not an absence of evidence: at the pooled sample size RR-QPSO is significantly
*worse* on the primary endpoint, and equivalence within ±0.02 was positively
demonstrated at n = 10.

**A conventional evolution strategy is competitive, except under constraints.**
CMA-ES matches RR-QPSO on the unconstrained objective at a third of the variance,
which by itself would undercut any claim that the quantum-inspired framing is
what produces the performance. On the constrained objective, however, CMA-ES
collapses while both QPSO variants hold up — the one pre-registered hypothesis
this study supports. A single multivariate Gaussian is drawn toward the mean of
several separated optima and lands in the void between them; a particle swarm can
occupy several basins at once. If this generalises, it is a practically useful
guide to optimizer selection for property-constrained generation, and it is
independent of the quantum-inspired question.

## Why the original claim did not replicate

Three factors compound.

The reported differences were **smaller than the measurement noise**. Selection
bias from shot noise inflates a single run's reported maximum by +0.011 to +0.027,
which exceeds the margins typically claimed. The noise analysis also has
predictive content: of the three differences in the original comparison, the two
inside the noise band failed to replicate and the one above it did.

A **confounded hyperparameter** accounted for part of the apparent gap. The method
and its baseline used different contraction–expansion schedules; aligning them
reversed the ordering.

**Effect sizes from underpowered samples are systematically inflated.** This study
measured that twice: shrinkage to 0.55–0.67× in the confirmatory ablation, and an
outright sign reversal in the constrained-objective replication, where d_z went
from +0.427 at n = 10 to −0.312 at n = 45. In neither case would appending seeds
to the original batch have produced a readable answer — the discovery sample's
positive bias would have mixed with the new data. Running independent
confirmatory studies is what made the results interpretable.

## What the mechanisms actually do

The mechanisms are not inert, and the study is more informative for having
measured them rather than only their downstream effect. They fire as designed and
produce ~10% broader coverage of the reachable objective space, at the cost of
visiting more low-quality regions.

That breadth does not pay on either endpoint tested. On `max V×U` it is a
consistent net loss, most pronounced at intermediate budgets where the exploration
delays convergence (Cliff's δ ≈ −0.77 at 4,000–5,000 evaluations). On the
set-level endpoint the breadth is *less* productive per unit than plain QPSO's:
each additional unit of coverage is worth ~2,148 molecules on average, but
RR-QPSO's extra coverage delivered under half of what that rate predicts.

The honest reading is that the mechanisms redistribute search effort without
improving its yield. That is a more specific and more useful negative result than
"the method does not work", because it identifies *where* the design fails: not
in whether the mechanisms activate, but in the productivity of the search
behaviour they induce.

## Implications for evaluation practice

The methodology mattered more than any individual result, and three elements did
most of the work.

**Budget parity enforced in code.** Comparing a sequential method against a
population method is only meaningful if both consume the same number of objective
calls, and that is too easy to get wrong when each implementation counts for
itself.

**Paired blocking with a shared seed.** Fixing both the optimizer's and the
sampler's random state within a seed removes between-seed difficulty, which in
this problem is large — some seeds are hard for every algorithm.

**Pre-registration with a stopping rule.** Four attempts were made to establish an
advantage, each planned before its data existed. Two returned null, one reversed,
and one was voided by its own validity check. Without a stopping rule declared in
advance, the natural response to each failure is to try another endpoint, another
budget, or another objective — and eventually one of them produces p < 0.05. The
rule was declared and honoured.

We note that the field routinely compares optimizers on this problem class using
single runs. Our measurements suggest that practice cannot resolve the
differences typically claimed.

## Limitations and future work

The Bayesian optimization comparison — this study's strongest positive result —
rests on an implementation weaker than the published reference in ways that all
favour the reference. Re-running it against Ax/BoTorch with an ARD kernel and no
training cap is the single most valuable follow-up, and would convert a
directional claim into a quantitative one.

The CMA-ES advantage under constraints was established at n = 10 on one
constrained objective. Whether it reflects a general property of single-Gaussian
models on multimodal constrained landscapes, or something specific to this
property term, is untested.

Ceiling effects at 1,000 shots may compress differences between the QPSO variants;
a 5,000-shot replication would settle this at roughly five times the cost.

Finally, the near-disjoint molecule sets produced by the two methods suggest that
ensembling distinct optimizers may cover chemical space more effectively than
improving any single one — a direction this study did not pursue.
