# sqmg_project-cudaq

**Rank-Refined Quantum-Behaved Particle Swarm Optimization (RR-QPSO) for Quantum
Molecular Generation (QMG).**

CUDA-Q 0.7.1 implementation of the optimizer and experiments in the paper
*Rank-Refined Quantum-Behaved Particle Swarm Optimization for Quantum Molecular
Generation* (Sing-Yun Wu, Sheng Yun Wu, I-Min Chiang, Tai-Yue Li). RR-QPSO is a
population-based replacement for Bayesian optimization (BO) in QMG parameter
search, targeting the 9-heavy-atom benchmark on NVIDIA V100 GPUs.

> With **M = 64** particles and **T = 150** iterations, RR-QPSO reaches a
> validity–uniqueness product of **V×U = 0.930**. Increasing the swarm to
> **M = 128** raises it to **V×U = 0.942**, compared with **0.902** for the
> re-run BO baseline. A multi-objective extension targeting **HBA = 4, HBD = 3**
> keeps a higher V×U than BO while steering molecular properties toward the
> target region.

---

## Overview

Quantum Molecular Generation (QMG) formulates molecule generation as a
parameterized quantum sampling problem: a chemistry-inspired *dynamic* quantum
circuit encodes sequential atom-then-bond decisions, and mid-circuit
measurements are decoded into molecular graphs. For molecules with `N` heavy
atoms the circuit is controlled by `θ ∈ R^D` with

```
D = 8 + 3(N-2)(N+3)/2
```

This project uses the **9-heavy-atom** setting, giving **D = 134** trainable
parameters on a **20-qubit** dynamic circuit. Each candidate `θ` is expensive to
score (stochastic circuit sampling → bitstring decoding → RDKit validity/
uniqueness), but candidate evaluations are mutually independent, which makes a
parallel, population-based optimizer a natural fit.

The QMG circuit and molecular decoding pipeline are **unchanged** from Chen et
al. 2025; the contribution here is entirely at the **optimizer** level.

---

## Method — RR-QPSO

The optimizer maintains a swarm of `M` particles, each a 134-D parameter vector,
and improves them over `T` iterations. Three components distinguish RR-QPSO from
standard QPSO.

**1. Sobol-based initialization.** Particles are initialized with Owen-scrambled
Sobol low-discrepancy sequences (`scipy.stats.qmc.Sobol(d=134, scramble=True)`)
for broad, deterministic coverage of the 134-D domain (no lucky-seed variance).
`M = 2^k` (e.g. 64, 128) satisfies Sobol's power-of-two uniformity guarantee.

**2. Rank-refined mean-best update.** Standard QPSO uses the plain mean of
personal-best positions as the swarm attractor. RR-QPSO sorts personal bests by
fitness and adds a rank-based correction that separates high- and low-fitness
regions:

```
m_RR = (1/M) Σ p_i  +  ρ Σ_{k=1..⌊M/2⌋} ( p_(k) − p_(M−k+1) )
```

with a fixed correction strength `ρ = 0.015` (`--rotate_factor 0.015`). This
gives a stronger population-level search signal than simple averaging.

**3. Fitness-guided refinement.** The product objective `F = V × U` can hide
different failure modes, so the attractor is blended with validity- and
uniqueness-oriented elites, each gated by its complementary metric (τ = 0.5):

```
m̃ = (w_RR·m_RR + I_V·w_V·x_V + I_U·w_U·x_U) / (w_RR + I_V·w_V + I_U·w_U)
w_RR = 0.70,  w_V = w_U = 0.15
```

The gates prevent chasing a high-validity solution that has collapsed in
diversity, or a high-uniqueness solution with poor validity.

**Particle update.** Positions follow the QPSO contraction–expansion update with
`m̃` as attractor; the coefficient `α` is annealed within `[α_min, α_max] =
[0.3, 1.2]` (broad early exploration, smaller late steps). All particles are
clipped to the valid parameter range after each update.

**Multi-GPU parallel evaluation.** At each iteration the `M` candidates are
distributed across `K` GPUs; each GPU worker runs circuit sampling, decoding and
scoring independently, then the optimizer collects the scores and performs the
RR-QPSO update.

> **Naming note.** "RR-QPSO" is the paper-level name. The implementation modules
> keep their original filenames and log tags (`qpso_optimizer_ae.py`, log prefix
> `AE-QPSO`); the algorithm is identical. The optimizer flags map to the paper as
> `--sobol_init` → Sobol init, `--rotate_factor`/`--ae_weighting` → rank-refined
> mean-best, `--vu_decouple` (`--w_vu/--w_v/--w_u`) → fitness-guided refinement.

---

## Results (9-heavy-atom benchmark)

All runs use the 134-parameter, 20-qubit dynamic circuit, CUDA-Q 0.7.1
(cuStateVec) on NVIDIA V100 GPUs, 5000 shots per evaluation unless noted, and
report the validity–uniqueness product `V × U`.

### Optimizer comparison — Fig. 2 (M = 64, T = 150)

| Optimizer | V (%) | U (%) | V × U (%) |
|---|---|---|---|
| BO (re-run baseline) | 94.2 | 95.7 | 90.2 |
| QPSO (no Sobol init) | — | — | 90.5 |
| QPSO + Sobol init | — | — | 91.4 |
| **RR-QPSO (this work)** | **95.9** | **97.0** | **93.0** |

![Fig. 2 — optimizer comparison](figures/fig2_VU_bars.png)

Adding Sobol init alone lifts the product modestly (91.4%); the larger gain
comes from combining rank-refined mean-best guidance with fitness-guided
refinement.

### Effect of particle count — Fig. 3 / Table I (T = 150)

| M | V (%) | U (%) | V × U (%) | Time (h) |
|---|---|---|---|---|
| 16 | 95.2 | 94.8 | 90.2 | 7.21 |
| 32 | 94.4 | 95.6 | 90.2 | 15.44 |
| 48 | 96.0 | 96.3 | 92.4 | 21.59 |
| 64 | 95.9 | 97.0 | 93.0 | 47.12 |
| 96 | 96.6 | 97.3 | 94.0 | 43.67 |
| **128** | **97.5** | **96.6** | **94.2** | 58.92 |

![Fig. 3 — particle-count convergence](figures/fig3_convergence.png)

Larger swarms give broader search coverage and converge to higher final scores;
the best result is **V × U = 0.942 at M = 128**.

### Multi-objective (HBA = 4, HBD = 3) — Fig. 4

A scalarized target-property objective folds H-bond acceptor/donor counts into
the fitness:

```
F_MO = (V × U) · [ (1 − λ) + λ · C_prop ],   λ = 0.40
C_prop = exp( −0.5 [ ((H̄_HBA − 4)/σ)^2 + ((H̄_HBD − 3)/σ)^2 ] ),  σ = 1
```

| Optimizer | V × U (%) | HBA (→4) | HBD (→3) |
|---|---|---|---|
| BO | 43.8 | 3.97 | 3.16 |
| **RR-QPSO (M = 32)** | **79.0** | **3.88** | **3.15** |

![Fig. 4 — multi-objective HBA/HBD](figures/fig4_hbahbd_compare.png)

Both optimizers steer mean HBA/HBD near the target region, but RR-QPSO retains a
substantially higher V × U under the added property constraint.

---

## Workflow

![Fig. 1 — RR-QPSO workflow for QMG](figures/fig1_workflow.png)

Sobol sampling initializes a swarm of `M` parameter vectors → each is evaluated
through QMG circuit sampling, bitstring decoding, molecular-graph generation and
fitness computation → the `M` evaluations are distributed across `K` GPUs → the
scores drive the RR-QPSO update. Iterate until convergence.

---

## Environment

### Hardware
```
Cluster : NCHC DGX (DGX111)
GPU     : 8 × V100-SXM2-16GB (Volta, sm_70)
CUDA    : Driver 535.x, Toolkit 12.2
```

### Software (hard constraints — do not change)

| Package | Version | Reason |
|---|---|---|
| `cuda-quantum-cu12` | **== 0.7.1** | Only version shipping sm_70 SASS; newer wheels silently fall back to CPU on V100 |
| `numpy` | `>= 1.24, < 2.0` | CUDA-Q 0.7.x is incompatible with NumPy 2.x |
| `rdkit` | `>= 2023.9.5` | SMILES validity / Lipinski HBA-HBD |
| `scipy` | recent | Sobol (`scipy.stats.qmc`) initialization |
| Python | 3.10 | matches the `cudaq-v071` conda env |

Install & verify:
```bash
conda activate cudaq-v071            # Python 3.10
pip install cuda-quantum-cu12==0.7.1
pip install "numpy>=1.24,<2.0" rdkit pandas matplotlib scikit-learn scipy

python -c "import cudaq; print(cudaq.__version__)"   # 0.7.1.x
python -c "import numpy; print(numpy.__version__)"   # < 2.0
python -c "from scipy.stats import qmc; print('scipy OK')"
```

---

## Repository Layout

```
sqmg_project-cudaq/
├── run_qpso_qmg_cudaq.py                    ← primary entry point (unconditional + opt-in HBA/HBD measure-only)
├── run_qpso_qmg_cudaq_hbahbd_multiobj.py    ← multi-objective runner (HBA/HBD in the objective)
├── qpso_optimizer_ae.py                     ← RR-QPSO optimizer (Sobol / rank-refined mbest / fitness-guided refinement)
├── qpso_optimizer_qmg.py                    ← legacy QPSO (reference only)
├── worker_eval.py                           ← per-GPU subprocess worker (sample → decode → V/U [+ HBA/HBD])
├── run_qpso_qmg_mpi.py                       ← MPI fallback (subprocess pool is the primary path)
├── run_sweep.sh / run_hbahbd_multiobj.sh     ← Fig. 3 sweep / Fig. 4 multi-objective drivers
├── qmg/
│   ├── generator_cudaq.py                    ← MoleculeGeneratorCUDAQ
│   └── utils/                                ← dynamic circuit, chemistry processing, V/U scoring, weights
├── figures/                                  ← paper figures (fig1_workflow, fig2_VU_bars,
│                                                fig3_convergence, fig4_hbahbd_compare) + generators
├── results_*/                                ← run logs / CSVs (large logs git-ignored; live on the cluster)
├── docs/                                      ← reference log-format template
├── requirements.txt
└── .gitignore
```

Large run logs / CSVs / `.npy` are git-ignored (they are big and live on the
cluster). Paper-relevant logs are grouped per figure under `results_*/`.

---

## Usage

### Quick sanity check (~5 min, 8 GPUs)
```bash
python run_qpso_qmg_cudaq.py \
    --backend cudaq_nvidia --num_heavy_atom 9 --num_sample 100 \
    --particles 8 --iterations 1 --n_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7 \
    --subprocess_timeout 120 --sobol_init --obl --vu_decouple \
    --task_name sanity --data_dir results_sanity
```

### Main run — RR-QPSO (M = 64, T = 150)
Run inside `tmux` to survive SSH disconnects.
```bash
python run_qpso_qmg_cudaq.py \
    --backend cudaq_nvidia --num_heavy_atom 9 --num_sample 5000 \
    --particles 64 --iterations 150 --n_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7 \
    --subprocess_timeout 360 \
    --sobol_init --obl --vu_decouple \
    --w_vu 0.70 --w_v 0.15 --w_u 0.15 \
    --alpha_max 1.2 --alpha_min 0.3 \
    --ae_weighting --rotate_factor 0.015 --seed 0 \
    --task_name unconditional_9_rrqpso_M64T150 --data_dir results_rrqpso
```
Set `--particles 128` for the best unconditional result (V×U = 0.942).

### Particle-count sweep (Fig. 3 / Table I)
`run_sweep.sh` runs the full method at `M = 16, 32, 48, 96, 128` at T = 150
(M = 64 reuses the main run), sequentially.

### Multi-objective HBA/HBD (Fig. 4)
```bash
python run_qpso_qmg_cudaq_hbahbd_multiobj.py \
    --backend cudaq_nvidia --num_heavy_atom 9 --num_sample 10000 \
    --particles 32 --iterations 150 --n_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7 \
    --hba_target 4 --hbd_target 3 --chem_weight 0.40 \
    --task_name chemistry_constraint_cudaq_multiobj_4HBA_3HBD_M32 \
    --data_dir results_hbahbd_multiobj
```
`run_hbahbd_multiobj.sh` launches the full `M ∈ {16, 32, 64, 128}` set. Each run
writes `{task}.log`, `{task}.csv` (gbest V×U), `{task}_multiobj.csv`
(per-iteration score / HBA / HBD) and `{task}_multiobj_best.json`.

### Monitoring
```bash
tail -f results_rrqpso/unconditional_9_rrqpso_M64T150.log
# GPU utilization — do NOT use `watch -n 5 nvidia-smi` on DGX111 (segfault)
while true; do clear; nvidia-smi; sleep 10; done
```

---

## Implementation notes

- **Subprocess pool, not MPI.** Each evaluation batch spawns fresh per-GPU
  subprocesses. The parent sets `CUDA_VISIBLE_DEVICES` before `Popen()`, so each
  child sees exactly one GPU; when the child exits after one `cudaq.sample()` the
  CUDA driver reclaims all pinned memory. This sidesteps two CUDA-Q 0.7.1 issues
  on the cluster — a `/dev/nvidia-ctl` serialization lock that made MPI ranks run
  sequentially, and a `cudaMallocHost` pinned-memory leak that OOM-killed
  long-lived ranks. `run_qpso_qmg_mpi.py` remains only as a fallback.
- **Backend.** Use `cudaq_nvidia` (cuStateVec). The `tensornet` backend hangs on
  these dynamic circuits and is disabled in the worker.
- **Do not upgrade CUDA-Q** past 0.7.1 (drops sm_70 SASS) or NumPy past 2.0.
- **`num_sample = 5000`** matches Chen 2025 for a fair comparison; `10000` biases
  uniqueness upward.

The primary runner also supports an opt-in *measure-only* HBA/HBD channel that
records mean HBA/HBD without changing the objective; the multi-objective runner
above instead folds HBA/HBD into the fitness.

---

## References

1. L.-Y. Chen, T.-Y. Li, Y.-P. Li, N.-Y. Chen, F. You. *Exploring Chemical Space
   with Chemistry-Inspired Dynamic Quantum Circuits in the NISQ Era.* J. Chem.
   Theory Comput., 2025.
2. J. Sun, B. Feng, W. Xu. *Particle Swarm Optimization with Particles Having
   Quantum Behavior.* IEEE CEC, 2004.
3. I. M. Sobol'. *On the Distribution of Points in a Cube and the Approximate
   Evaluation of Integrals.* USSR Comput. Math. Math. Phys., 1967.
4. A. B. Owen. *Randomly Permuted (t,m,s)-Nets and (t,s)-Sequences.* Monte Carlo
   and Quasi-Monte Carlo Methods, 1995.

## Citation

> S.-Y. Wu, S. Y. Wu, I-M. Chiang, T.-Y. Li. *Rank-Refined Quantum-Behaved
> Particle Swarm Optimization for Quantum Molecular Generation.*

Computational resources provided by the National Center for High-performance
Computing (NCHC), National Institutes of Applied Research (NIAR), Taiwan.
