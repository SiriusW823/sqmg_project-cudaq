# Pre-registration — Confirmatory ablation of two RR-QPSO components

**Written and committed before any confirmatory data was collected.**
Git commit of this file timestamps the analysis plan.

---

## 1. Background and motivation

A pilot ablation (n = 10 seeds, 7 configurations, 1,984 evaluations, reported in
`results_ablation/`) measured the contribution of five RR-QPSO components by
disabling one at a time. Findings:

| Component | Δ median | Wins | Cohen's dz |
|---|---|---|---|
| OBL | +0.0250 | 7/10 | +0.513 |
| AE-weighted mbest | +0.0130 | 8/10 | +0.538 |
| Sobol initialization | +0.0100 | 7/10 | +0.344 |
| V–U decoupling | +0.0170 | 7/10 | +0.191 |
| Mode-collapse recovery | +0.0060 | 6/10 | +0.176 |

All five point estimates favour keeping the component, but no individual
comparison reached significance after Holm correction, and a seed-level sign-flip
permutation test on the aggregate gave p = 0.0814 (95% bootstrap CI
[−0.0032, +0.0229]).

The pilot is therefore **inconclusive rather than null**: it is underpowered.
A power calculation from the observed effect sizes gives the seeds needed for
80% power at α = 0.05: OBL n ≈ 30, AE mbest n ≈ 28, Sobol n ≈ 67,
V–U n ≈ 215, mode-collapse n ≈ 253.

Only OBL and AE mbest are reachable at feasible cost. This study tests those two.

---

## 2. Hypotheses (pre-specified, confirmatory)

Let `F(c, s)` be the best-so-far V×U reached by configuration `c` under seed `s`
at exactly 1,984 evaluations.

- **H1 (OBL):** `Δ_OBL(s) = F(rr_full, s) − F(rr_noOBL, s) > 0`
- **H2 (AE mbest):** `Δ_AE(s) = F(rr_full, s) − F(rr_noAE, s) > 0`

Both are directional (one-sided), because the pilot established the direction and
the mechanistic claim in the paper is that these components help.

**Not tested in this study:** the remaining three components, and RR-QPSO versus
plain QPSO. The latter was already answered separately: at the paper's full
budget (9,664 evaluations, n = 10) the two are statistically equivalent
(TOST at ±0.02, p = 0.0011; Wilcoxon p = 0.121). This study does **not** revisit
that question and its result must not be presented as bearing on it.

---

## 3. Design

| Item | Value |
|---|---|
| Configurations | `rr_full` (ablate=none), `rr_noOBL` (ablate=obl), `rr_noAE` (ablate=ae) |
| Seeds | **10–39** (n = 30) — disjoint from the pilot's 0–9 |
| Particles M | 64 |
| Evaluation budget | 2,000 (endpoint read at 1,984, matching the pilot) |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for every configuration |
| Objective | V × U |
| Blocking | Paired: seed `s` controls both the optimizer RNG and the shot RNG |
| Nodes | DGX101 and DGX102 excluded (drained / ~14× slower) |

**Fresh seeds are essential.** The two components were selected because they had
the largest effects in the pilot, so the pilot's effect estimates are inflated by
selection. Reusing seeds 0–9 would test the hypothesis on the data that generated
it. Seeds 10–39 are new, so this is a genuine out-of-sample confirmation.

---

## 4. Analysis plan

1. **Primary test:** one-sided paired Wilcoxon signed-rank on `Δ_OBL` and
   `Δ_AE`, α = 0.05.
2. **Multiplicity:** Holm–Bonferroni across the two primary hypotheses.
3. **Effect size:** Cliff's δ and a 95% bootstrap CI (20,000 resamples) on the
   median difference, reported regardless of significance.
4. **Reported estimate:** the confirmatory effect size from seeds 10–39 is the
   one to report. The pilot's estimate is reported separately and explicitly
   labelled as the (upward-biased) discovery sample.
5. Configurations that fail to reach 1,984 evaluations are excluded pairwise,
   and the number excluded is reported.

## 5. Decision rules

- `p_holm < 0.05` for a hypothesis → that component contributes to RR-QPSO at
  this budget.
- Otherwise → reported as a null result. **No further seeds will be added and no
  alternative endpoint will be substituted to try to reach significance.**

## 6. What a positive result would and would not mean

A confirmed H1 or H2 supports: *"this component contributes to RR-QPSO's
performance."*

It does **not** support: *"RR-QPSO outperforms QPSO."* Those are separate claims,
and the second has already been tested and answered in the negative. If some
components help while the full method is nonetheless equivalent to the baseline,
the honest reading is that other parts of the method offset them — which is
itself a reportable finding about which quantum-inspired mechanisms carry their
weight.

## 7. Interim monitoring

Progress will be checked during the run for infrastructure reasons only
(stragglers, node failures). **No hypothesis test will be run on partial data,
and no decision about the experiment will be made from an interim look.**
