# Pre-registration — does RR-QPSO's broader exploration produce more distinct molecules?

**Written and committed before the logging pipeline existed and before any
diversity data was collected.** The git commit of this file timestamps the plan.

---

## 1. Why a different endpoint, and why that needs justifying

Nine experiments (~500 runs, three pre-registered confirmatory studies) have
tested RR-QPSO against plain QPSO on `max over evaluations of V×U`. None found an
advantage; pooled at n = 20 on the unconstrained objective RR-QPSO is
significantly *worse* (Δ = −0.0090, p = 0.0152), and on the constrained objective
a promising n = 10 result reversed under a pre-registered n = 45 replication
(d_z +0.427 → −0.312).

**Changing the endpoint after failing on the original one is exactly the practice
that makes results untrustworthy.** This study is only defensible because the new
endpoint is derived from a measured property of the algorithm, not chosen because
it happens to look better — and because the original endpoint is still reported.

### The measured property

Analysis of the existing 55 paired runs shows what RR-QPSO's mechanisms actually
do. All three fire regularly: the V–U decoupling term sits at its 0.15 cap for
50–77% of evaluations, mode-collapse recovery triggers ~179 times per long run,
and stagnation reinitialization ~3 times. Their consistent, measurable effect is
**broader exploration**:

| | Grid cells covered in (V,U) space (20×20) | σ_V | σ_U |
|---|---|---|---|
| RR-QPSO | **323.5** / 400 | 0.1995 | 0.2480 |
| QPSO | 296.0 / 400 | 0.1893 | 0.2282 |

RR-QPSO also visits more low-quality points (6.09% of evaluations at U < 0.20 vs
4.60%), which is the cost side of the same trade: it explores harder, and the
mode-collapse guard exists to stop those points from corrupting the population.

### The mismatch

`max V×U` rewards a single parameter vector. Molecular generation delivers a
**set of molecules**. An optimizer that maintains population diversity must lose
on the single-point metric — exploration costs evaluations — while potentially
winning on the set. That is the hypothesis this study tests.

`U` (uniqueness) already measures diversity, but only *within one sampling call*.
No experiment so far has measured diversity *across the search*.

## 2. Hypotheses (pre-specified)

Let `D(a, s)` be the number of distinct valid SMILES encountered across **all**
evaluations of a run of algorithm `a` under seed `s`, at exactly 1,984
evaluations.

- **H1 (primary):** `D(rr_qpso, s) > D(qpso, s)`. One-sided paired Wilcoxon,
  α = 0.05.
- **H2 (primary, the one that matters):** RR-QPSO produces more distinct
  molecules **after controlling for how much it explored**. Tested by paired
  Wilcoxon on the residuals of `D` regressed on grid coverage `C` (the number of
  occupied cells in the 20×20 (V,U) grid), pooled across arms.

H2 exists because H1 alone is close to tautological: RR-QPSO is already known to
explore ~10% more, so finding more molecules would be unsurprising and
uninformative. The question with scientific content is whether its exploration is
**better directed**, not merely larger. **If H1 is supported and H2 is not, the
correct conclusion is that RR-QPSO buys diversity purely by spending more search
breadth, which plain QPSO could also do by other means.**

## 3. Secondary endpoints (reported regardless of outcome)

1. **`max V×U`** — the original endpoint, on the same runs. This study does not
   replace it; if RR-QPSO wins on `D` while losing on `V×U`, both are reported
   with equal prominence.
2. `D` at 500 / 1,000 / 1,500 evaluations, to show whether any gap grows or
   closes with budget.
3. Grid coverage `C`, to confirm the exploration difference replicates.
4. The fraction of distinct molecules unique to each arm (set difference), which
   distinguishes "finds more of the same" from "finds different things".

## 4. Design

| Item | Value |
|---|---|
| Arms | `qpso`, `rr_qpso` (ablate=none) |
| Objective | `vu` (unconstrained) |
| Seeds | **100–129** (n = 30) — disjoint from every prior experiment |
| Particles M | 64 |
| Budget | 2,000 evaluations (endpoint read at 1,984) |
| Shots | 1,000 |
| α schedule | [0.3, 1.2] for both arms |
| Blocking | Paired: seed fixes both the optimizer RNG and the shot RNG |
| Nodes | DGX102, DGX106 excluded |

The unconstrained objective is used deliberately: it is where RR-QPSO's
single-point deficit is largest and best established (n = 20, p = 0.0152). If a
diversity advantage exists there, it is not an artefact of a regime chosen to
flatter the method.

n = 30 gives 80% power at d_z ≈ 0.47. Given that every effect measured in this
project has shrunk or reversed on held-out seeds, a smaller true effect will not
be detected, and that limitation is accepted in advance rather than used
afterwards to argue for more seeds.

## 5. Analysis plan

1. **H1:** one-sided paired Wilcoxon on `D`, α = 0.05.
2. **H2:** fit `D ~ C` by ordinary least squares on all 60 runs pooled; take the
   per-run residual; one-sided paired Wilcoxon on the residuals, α = 0.05.
3. **Multiplicity:** Holm–Bonferroni across H1 and H2.
4. **Effect size:** Cliff's δ and 95% bootstrap CI (20,000 resamples) on the
   median difference, for every endpoint, significant or not.
5. **Validity check:** total distinct molecules per run must be > 0 and the
   cumulative curve must be monotonic. A run failing either is excluded and
   counted — the same class of check that caught the voided HBA/HBD batch.
6. Runs not reaching 1,984 evaluations are excluded pairwise and counted.

## 6. Decision rules

- **H1 and H2 both supported** → RR-QPSO generates a more diverse molecule set,
  and does so beyond what its extra exploration alone explains. This would be the
  first demonstrated advantage of the method, and would be reported with its
  scope: a set-level generation benefit, alongside a single-point optimization
  deficit.
- **H1 supported, H2 not** → the diversity gain is explained by exploring more
  broadly, not by exploring better. Reported as such, explicitly not as a
  demonstration that the quantum-inspired mechanisms are effective.
- **H1 not supported** → null. **This closes the search for an advantage.** The
  conclusion is that the mechanisms measurably change search behaviour without
  improving any endpoint tested — optimization quality or generation diversity —
  and that conclusion is itself the finding.

No further endpoints will be tried after this study. Four attempts, each
pre-registered, is the limit set here in advance.

## 7. Implementation note

Distinct SMILES are not currently recorded; the workers discard `smiles_dict`
after computing V and U. The pipeline added for this study writes them to a
per-task append-only file behind a flag that is **off by default**, so no existing
experiment changes behaviour. The flag's effect is verified by a smoke test
before submission, and the union is recomputed from the raw files at analysis
time rather than accumulated in memory during the run.

## 8. Interim monitoring

Infrastructure only — stragglers, dead nodes, and the §5.5 validity check.
**No hypothesis test on partial data.**
