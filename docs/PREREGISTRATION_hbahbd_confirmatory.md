# Pre-registration — RR-QPSO on the constrained (HBA/HBD) objective

**Written and committed before any data on this objective was analysed.**
The git commit of this file timestamps the analysis plan.

---

## 1. Background and motivation

Across five experiments on the unconditional V×U objective, RR-QPSO shows no
advantage over plain QPSO. Pooling the two full-budget batches (n = 20, identical
settings) gives Δ = −0.0090 with a 95% CI of [−0.0160, −0.0040] that excludes
zero: RR-QPSO is *significantly worse*, though by a small margin (Cliff's
δ = −0.175).

All of that evidence comes from **one objective function**. The multi-objective
HBA/HBD variant folds a target-property term into the fitness:

```
F_MO = (V × U) · [ (1 − λ) + λ · C_prop ],   λ = 0.40
C_prop = exp( −0.5 [ ((H̄_HBA − 4)/σ)² + ((H̄_HBD − 3)/σ)² ] ),   σ = 1
```

This is a **more rugged, more multimodal landscape**: the property term
introduces additional local optima that the unconstrained product does not have.
RR-QPSO's mechanisms — opposition-based learning, mode-collapse detection and
reinitialization, rank-refined mean-best — are exploration devices. On the
unconstrained objective they appear to cost budget without returning value
(RR-QPSO is significantly worse at 4,000–5,000 evaluations, Cliff's δ ≈ −0.77).
On a rugged landscape, exploration is exactly what should pay.

This hypothesis is **specified in advance from the mechanism**, not selected by
inspecting HBA/HBD data — none has been analysed for these arms. The published
work already claims a large RR-QPSO advantage on this objective (V×U 79.0 vs
43.8 for BO), from a single run.

## 2. Hypotheses (pre-specified, confirmatory)

Let `F(a, s)` be the best-so-far multi-objective fitness `F_MO` reached by
algorithm `a` under seed `s` at exactly 1,984 evaluations.

- **H1 (primary):** RR-QPSO outperforms plain QPSO on the constrained objective:
  `Δ = F(rr_qpso) − F(qpso) > 0`. One-sided paired Wilcoxon.
- **H2 (secondary):** RR-QPSO outperforms CMA-ES on the constrained objective.
  One-sided paired Wilcoxon.
- **H3 (secondary, interaction):** the RR-QPSO − QPSO difference is more
  favourable on the constrained objective than on the unconditional one.
  Tested by comparing `Δ_constrained` against `Δ_unconditional` on the same
  seeds (Wilcoxon on the per-seed difference-of-differences).

H1 is one-sided because the direction is specified by the mechanism in advance.
H3 is the hypothesis that actually matters: a difference confined to one
objective is only interesting if it *differs* from the other objective.

## 3. Design

| Item | Value |
|---|---|
| Arms | `qpso`, `rr_qpso` (ablate=none), `cmaes` |
| Objective | `hbahbd` (HBA target 4, HBD target 3, λ = 0.40) |
| Seeds | **0–9** (n = 10) |
| Particles M | 64 |
| Budget | 2,000 evaluations (endpoint read at 1,984) |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for both QPSO variants |
| Blocking | Paired: seed `s` fixes both the optimizer RNG and the shot RNG |
| Nodes | DGX102, DGX106 excluded |

Seeds 0–9 are used deliberately: the unconditional results for these arms are
already available on exactly these seeds, so H3 can be tested within-seed rather
than across independent samples.

**Budget note.** 2,000 evaluations rather than 9,664, to keep the cost near
10 hours. This is the budget region where RR-QPSO was *worst* on the
unconditional objective (Δ = −0.0120 at 2,000, and −0.0310 at 4,000), so it is
not a region selected to flatter the method.

## 4. Analysis plan

1. **Primary:** one-sided paired Wilcoxon on `Δ_H1`, α = 0.05.
2. **Multiplicity:** Holm–Bonferroni across H1 and H2. H3 is reported separately
   as the interaction test and is not corrected against H1/H2.
3. **Effect size:** Cliff's δ and a 95% bootstrap CI (20,000 resamples) on the
   median difference, reported regardless of significance.
4. **Secondary reporting:** mean HBA and HBD of the best solution per arm, to
   confirm that the property target is actually being approached and that any
   fitness difference is not achieved by ignoring the constraint.
5. Runs failing to reach 1,984 evaluations are excluded pairwise, and the number
   excluded is reported.

## 5. Decision rules

- `p_holm < 0.05` for H1 → RR-QPSO outperforms QPSO on this objective.
- H1 significant **and** H3 significant → the advantage is specific to the
  constrained landscape. This is the outcome that would support the method.
- H1 significant but H3 not → weaker: the direction differs but the objectives
  cannot be distinguished at this sample size.
- H1 not significant → reported as null. **No additional seeds, no alternative
  endpoint, no switch to a different budget.**

## 6. What each outcome would mean

**H1 and H3 both supported.** RR-QPSO's exploration mechanisms pay off where the
landscape is rugged and cost budget where it is not. That is a coherent,
mechanistically grounded, and useful claim — and it would be the first positive
result for the method in this study. It would need to be stated with its scope:
an advantage on constrained objectives, not in general.

**H1 not supported.** Six experiments across two objectives find no advantage.
At that point the reasonable conclusion is that the quantum-inspired refinements
do not improve on plain QPSO for quantum molecular generation, and the
contribution of this work lies in the population-vs-Bayesian comparison and in
the methodology.

## 7. Interim monitoring

Progress will be checked for infrastructure reasons only. **No hypothesis test
will be run on partial data.**
