# Pre-registration — adequately powered replication of RR-QPSO vs QPSO on the constrained objective

**Written and committed before any data on the held-out seeds was collected.**
The git commit of this file timestamps the analysis plan.

---

## 1. Background

Experiment 8 (pre-registration `8ddc9d7`, n = 10, seeds 0–9) tested whether
RR-QPSO outperforms plain QPSO on the constrained HBA/HBD objective. Result:

| | Value |
|---|---|
| Δ (RR-QPSO − QPSO), median | **+0.0186** |
| 95% bootstrap CI | [−0.0017, +0.0416] |
| RR-QPSO wins | 8/10 |
| One-sided paired Wilcoxon | p = 0.0801 |
| Cohen's d_z | +0.427 |

Reported as null against the pre-registered α = 0.05, and no seeds were added.
The direction, the win count and the effect size all point the same way; the
sample does not settle it. At d_z = 0.427, n = 10 has roughly **21% power** —
the experiment was very likely to fail regardless of whether the effect is real.

This study repeats the same test at a sample size chosen in advance to detect it.

## 2. Hypothesis (single, primary, confirmatory)

Let `F(a, s)` be the best-so-far constrained fitness `F_MO` reached by algorithm
`a` under seed `s` at exactly 1,984 evaluations.

**H1:** `Δ(s) = F(rr_qpso, s) − F(qpso, s) > 0`, one-sided paired Wilcoxon
signed-rank, α = 0.05.

One hypothesis only. No secondary tests, so no multiplicity correction and no
opportunity to substitute an endpoint that happens to work.

## 3. Sample size and what it can detect

n = 45 paired seeds.

| True d_z | Power at n = 45 |
|---|---|
| 0.427 (observed in Experiment 8) | ~92% |
| 0.35 | ~80% |
| 0.30 | ~66% |
| 0.20 | ~35% |

**Stated plainly: if the true effect is half the observed one, this study will
probably fail too.** Effect sizes from underpowered studies that land near
p = 0.08 are systematically inflated — the confirmatory ablation in this project
measured exactly that, with d_z shrinking to 0.55–0.67× on held-out seeds. If the
same shrinkage applies here, the true d_z is nearer 0.25 and n = 45 is not
enough. n = 45 is what fits the compute budget (90 runs ≈ 36 h); it is chosen
honestly rather than to guarantee a result.

Should this study also return null, the correct conclusion is **not** "try
again with more seeds". It is that any RR-QPSO advantage over plain QPSO on this
objective is smaller than d_z ≈ 0.35, which given the 0.40-wide spread between
methods is too small to matter practically.

## 4. Design

| Item | Value |
|---|---|
| Arms | `qpso`, `rr_qpso` (ablate=none) |
| Objective | `hbahbd` (HBA target 4, HBD target 3, λ = 0.40) |
| Seeds | **10–54** (n = 45) — disjoint from Experiment 8's 0–9 |
| Particles M | 64 |
| Budget | 2,000 evaluations (endpoint read at 1,984) |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for both arms |
| Blocking | Paired: seed fixes both the optimizer RNG and the shot RNG |
| Nodes | DGX102, DGX106 excluded |

Every parameter is identical to Experiment 8 except the seeds. CMA-ES is not
re-run: H2 is already established (δ = +0.940, p_holm = 0.0020) and a third arm
would cost 50% more compute to replicate a settled result rather than to power
the open one.

## 5. Analysis plan

1. **Primary:** one-sided paired Wilcoxon on Δ, α = 0.05.
2. **Effect size:** Cliff's δ and a 95% bootstrap CI (20,000 resamples) on the
   median difference, reported regardless of significance.
3. **Constraint check:** the C_prop of each arm's best solution, to confirm the
   property term is active. Experiment 8 run 1 was voided by this check; it is
   mandatory. If the median C_prop is below 0.5, the batch is void and the cause
   is investigated before anything is reported.
4. **Pooling:** Δ will also be reported pooled with Experiment 8 (n = 55) as a
   secondary, clearly-labelled estimate. The primary test uses the held-out
   seeds only.
5. Runs failing to reach 1,984 evaluations are excluded pairwise and counted.

## 6. Decision rules

- `p < 0.05` → RR-QPSO outperforms plain QPSO on the constrained objective.
  This would be the first evidence in this project that the quantum-inspired
  refinements do something, and it would be reported with its scope: constrained
  objectives only, since the unconstrained comparison at n = 20 goes the other
  way (Δ = −0.0090, p = 0.0152).
- `p ≥ 0.05` → null. **This line of inquiry stops.** No further seeds, no
  alternative endpoint, no different budget, no third objective. The conclusion
  is that the advantage, if any, is below the practical-relevance threshold.

## 7. Interim monitoring

Progress will be checked for infrastructure reasons only — stragglers, dead
nodes, and the constraint check in §5.3, which is a data-validity check rather
than a hypothesis test. **No hypothesis test on partial data.**
