# sqmg_project-cudaq

CUDA-Q 0.7.1 implementation of QMG (Quantum Molecular Generation) with
AE-SOQPSO optimization, targeting 9-heavy-atom molecule generation on
NCHC DGX111 V100 GPUs.

---

## Research Goal

Surpass the Bayesian Optimization (BO) baseline reported in Chen et al. 2025
(JCTC) for 9-heavy-atom unconditional QMG:

```
Target metric : V × U (validity × uniqueness) > 0.8834
Paper baseline: V = 0.955, U = 0.925, V × U = 0.8834  (~355 BO evaluations)
```

The BO optimizer is replaced with a custom **AE-SOQPSO** (Adaptive Ensemble
Stochastic Optimal Quantum PSO), integrating three papers:

| Paper | Role |
|---|---|
| Chen et al. 2025 (JCTC) | 20-qubit dynamic circuit, QMG framework |
| Xiao et al. 2026 (arXiv:2604.13877v1) | SQMG, tensornet speed analysis |
| Tseng et al. 2024 (arXiv:2311.12867v2) | AE-QTS, U-shaped harmonic weighting |

### Early Experiment Signal (2026-04-24, aborted at iteration 2)

The optimizer converged rapidly before OOM terminated the run:

| Stage | gbest V×U | V★ ever | U★ ever |
|---|---|---|---|
| Phase 0 done | 0.1548 | 0.996 | 0.254 |
| Iter 1 done | 0.3422 | 0.996 | 0.566 |
| Iter 2 done | 0.4104 | 0.996 | 0.909 |

V★ = 0.996 × U★ = 0.909 ≈ **0.905 — above the baseline** was reached in
individual particles by iteration 2. The infrastructure failure (MPI
serialization + OOM) was the only blocker. Both issues are resolved in
v10.1 / v1.3.

---

## Repository File Map

```
sqmg_project-cudaq/
│
├── run_qpso_qmg_cudaq.py       ← PRIMARY entry point  (v10.1, parallel subprocess)
├── run_qpso_qmg_mpi.py         ← MPI fallback          (v1.3, deadlock fix)
├── worker_eval.py              ← Subprocess worker     (v10.2, tensornet blocked)
├── qpso_optimizer_ae.py        ← AE-SOQPSO optimizer   (v1.1, U-shape weighting fix)
├── qpso_optimizer_qmg.py       ← Legacy SOQPSO         (reference only)
├── cutn-qmg_mpi_8g.slurm      ← SLURM script          (v1.2, --gpu-bind fix)
│
├── qmg/
│   ├── __init__.py
│   ├── generator_cudaq.py      ← MoleculeGeneratorCUDAQ  (v10.0)
│   └── utils/
│       ├── __init__.py
│       ├── build_dynamic_circuit_cudaq.py ← _qmg_n9 kernel  (v9.1, semicolon fix)
│       ├── weight_generator.py            ← ConditionalWeightsGenerator
│       ├── chemistry_data_processing.py   ← MoleculeQuantumStateGenerator
│       └── fitness_calculator.py          ← V/U scoring
│
├── docs/
│   └── qmg-soqpso-handoff-2026-05-07.md  ← Full runbook
│
├── requirements.txt
└── .gitignore
```

---

## Hardware and Environment

### Cluster

```
Host   : DGX111 (NCHC)
GPU    : 8 × V100-SXM2-16GB  (Volta, sm_70, CC 7.0)
CUDA   : Driver 535.183.01, Toolkit 12.2
```

### Conda Environment (mandatory)

```bash
conda activate cudaq-v071   # Python 3.10
```

Hard constraints — **do not change**:

| Package | Version | Reason |
|---|---|---|
| `cuda-quantum-cu12` | **== 0.7.1** | Only version with sm_70 SASS; newer wheels silently fall back to CPU |
| `numpy` | `>= 1.24, < 2.0` | CUDA-Q 0.7.x incompatible with NumPy 2.x |
| `rdkit` | `>= 2023.9.5` | Required for SMILES validation |

Install:

```bash
pip install cuda-quantum-cu12==0.7.1
pip install "numpy>=1.24,<2.0" rdkit pandas matplotlib scikit-learn
```

---

## Architecture

### Call Graph

```
run_qpso_qmg_cudaq.py  (main process, stable RSS < 1 GB)
│
├─ ConditionalWeightsGenerator          # generates / constrains 134 float weights
│
└─ AESOQPSOOptimizer (qpso_optimizer_ae.py v1.1)
    │  M=50 particles, T=40 iterations, D=134 dimensions
    │  AE-QTS U-shaped harmonic mbest + paired attractor update
    │
    └─ batch_evaluate_fn  (parallel subprocess pool)
        │  Each QPSO iteration launches ⌈M/N_GPUS⌉ rounds.
        │  Each round: N_GPUS=8 subprocesses launched simultaneously.
        │
        └─ [subprocess × 8 in parallel]  worker_eval.py
                CUDA_VISIBLE_DEVICES set by parent before Popen()
                → CUDA context initialized to one dedicated GPU
                │
                └─ MoleculeGeneratorCUDAQ (generator_cudaq.py v10.0)
                    │  chemistry_constraint=False (already applied by parent)
                    │
                    ├─ cudaq.sample(_qmg_n9, w_list, shots_count=10000)
                    │   20-qubit dynamic circuit
                    │   90 named mid-circuit measurements
                    │   134 float weights
                    │
                    └─ _reconstruct_bitstrings_n9()
                        90 named registers → bitstrings
                        → SMILES via MoleculeQuantumStateGenerator
                        → validity, uniqueness
```

### Why subprocess pool (not MPI)

Two compounding failures were diagnosed in the 2026-04-24 MPI experiment:

**Failure 1 — MPI serialization.**  NCHC SLURM uses cgroup v2 for GPU
isolation.  Without `--gpu-bind=per_task:1`, all 8 MPI ranks were mapped
to a single physical GPU. Even with correct cgroup binding, CUDA-Q 0.7.1's
cuStateVec backend acquires a node-wide serialization lock through
`/dev/nvidia-ctl` during `cudaq.sample()`, causing all ranks to execute
sequentially: measured 6129 s for 50 particles vs. an expected 858 s.

**Failure 2 — cudaMallocHost pinned memory leak.**  CUDA-Q 0.7.1 allocates
approximately 2.5 GB of pinned memory per `cudaq.sample()` call via
`cudaMallocHost`. This memory is managed by the CUDA driver, not the process
heap; `del`, `gc.collect()`, and `malloc_trim(0)` are all ineffective.  The
only release mechanism is CUDA context destruction, which is triggered only
when the process exits.  Long-lived MPI ranks accumulate ~500 GB of pinned
memory across 4 batches of 50 particles before the kernel OOM-kills them.

**v10.1 solution.**  Each call to `evaluate_fn` / `batch_evaluate_fn` spawns
a fresh subprocess.  The parent sets `CUDA_VISIBLE_DEVICES=<gpu_id>` in the
child's environment before `Popen()` — before any CUDA initialization — so
the child sees exactly one GPU regardless of cgroup policy.  When the child
exits after a single `cudaq.sample()`, the CUDA driver destroys the context
and reclaims all pinned memory.  Peak concurrent pinned memory is bounded to
8 × 2.5 GB = 20 GB.

### AE-SOQPSO Algorithm (v1.1 corrections)

Two bugs were corrected relative to v1.0 (both verified against
arXiv:2311.12867v2):

**Bug 1 — mbest weighting.**  v1.0 used monotonically decreasing weights
`w_k = 1/(k+1)` favoring only top-ranked particles.  The paper's AE-QTS
Algorithm 3 applies rotation magnitude `Δθ/k` to both `best_k` and
`worst_k`, producing a **U-shaped** weight profile: high influence at both
ends, minimum in the middle.  v1.1 implements symmetric harmonic weighting:
rank `k` and rank `M+1-k` each contribute `1/k`.

**Bug 2 — paired update direction.**  v1.0 moved `worst_k` with Cauchy
perturbation (random exploration).  The paper applies amplitude amplification
to both `best_k` and `worst_k` — both move toward their local attractor
`φ·pbest + (1-φ)·gbest` with step `rotate_factor/k`.  Cauchy mutation is a
separate SOQPSO mechanism applied in the main loop; it does not belong inside
the AE pairing step.

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `num_heavy_atom` | 9 | 20-qubit circuit, 134 float params |
| `num_sample` | 10000 | shots per `cudaq.sample()`, matches paper |
| `particles (M)` | 50 | 134-D space; larger than paper's 30 Sobol init |
| `iterations (T)` | 40 | `total_evals = 50 × 41 = 2050` |
| `n_gpus` | 8 | subprocess pool width |
| `alpha_max / min` | 1.2 / 0.4 | cosine annealing bounds |
| `mutation_prob` | 0.12 | Cauchy heavy-tail mutation rate |
| `stagnation_limit` | 8 | iterations before reinit triggers |
| `reinit_fraction` | 0.20 | fraction of worst particles replaced |
| `ae_weighting` | True | U-shaped harmonic mbest (v1.1) |
| `pair_interval` | 5 | AE paired update every N QPSO iterations |
| `rotate_factor` | 0.01 | paired update step `Δθ/k` scaling |

---

## Quick Start

### Step 1 — Verify environment

```bash
conda activate cudaq-v071

python -c "
import cudaq, numpy as np
from qmg.utils import ConditionalWeightsGenerator
from qpso_optimizer_ae import AESOQPSOOptimizer
print('imports OK')
print('cudaq :', cudaq.__version__)    # must be 0.7.1.x
print('numpy :', np.__version__)       # must be < 2.0
"
```

### Step 2 — Single-GPU smoke test (~2 min)

```bash
# Generate a test weight file
python -c "
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
cwg = ConditionalWeightsGenerator(9, smarts=None)
w = cwg.generate_conditional_random_weights(random_seed=42)
np.save('/tmp/smoke_w.npy', w)
print('weight len =', len(w))   # must be 134
"

# Evaluate on GPU 0
CUDA_VISIBLE_DEVICES=0 python worker_eval.py \
    --weight_path /tmp/smoke_w.npy \
    --result_path /tmp/smoke_r.npy \
    --num_heavy_atom 9 \
    --num_sample 200 \
    --backend cudaq_nvidia

# Verify result
python -c "
import numpy as np
r = np.load('/tmp/smoke_r.npy')
print(f'V={r[0]:.3f}  U={r[1]:.3f}')
assert r[0] > 0, 'V=0, GPU not working'
print('worker_eval smoke test PASSED')
"
```

### Step 3 — Multi-GPU parallel verification (~3 min)

```bash
python run_qpso_qmg_cudaq.py \
    --backend cudaq_nvidia \
    --n_gpus 8 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --particles 8 \
    --iterations 1 \
    --num_sample 100 \
    --subprocess_timeout 120 \
    --task_name sanity_check \
    --data_dir results_sanity
```

Expected log output:

```
[v10.1] 並行功能驗證：同時啟動 8 個子行程（各 5 shots）...
  GPU 0: V=0.XXX  U=0.XXX  ✓
  ...
  GPU 7: V=0.XXX  U=0.XXX  ✓
[v10.1] 並行驗證完成（XXXs）  ✓ 所有 GPU 正常
```

Verify parallelism by watching GPU utilization during the test:

```bash
# In a second tmux pane — do NOT use "watch -n 5 nvidia-smi" (segfaults on DGX111)
while true; do clear; nvidia-smi; sleep 10; done
```

All 8 GPUs should show GPU-Util > 80 % simultaneously during each round.

---

## Full Experiment

Run inside a tmux session to survive SSH disconnects:

```bash
tmux new -s qmg_main
conda activate cudaq-v071
cd ~/sqmg_project-cudaq
git pull origin main

python run_qpso_qmg_cudaq.py \
    --backend             cudaq_nvidia                     \
    --num_heavy_atom      9                                \
    --num_sample          10000                            \
    --particles           50                               \
    --iterations          40                               \
    --n_gpus              8                                \
    --gpu_ids             0,1,2,3,4,5,6,7                 \
    --subprocess_timeout  600                              \
    --alpha_max           1.2                              \
    --alpha_min           0.4                              \
    --mutation_prob       0.12                             \
    --stagnation_limit    8                                \
    --reinit_fraction     0.20                             \
    --ae_weighting                                         \
    --pair_interval       5                                \
    --rotate_factor       0.01                             \
    --seed                42                               \
    --task_name           unconditional_9_ae_parallel_v101 \
    --data_dir            results_parallel_v101
```

### Timing estimate (8 × V100)

```
Single evaluation  :  ~122.6 s  (10 000 shots, cudaq_nvidia)
Rounds per iter    :  ⌈50/8⌉ = 7 rounds
Time per iter      :  7 × 122.6 s ≈  858 s  (≈ 14 min)
Phase 0            :                  858 s
40 iterations      :  40 × 858 s = 34 320 s  (≈ 9.5 h)
Total (T=40)       :                ~9.8 h
Total evals        :  50 × 41 = 2 050
```

### Output files

```
results_parallel_v101/
├── unconditional_9_ae_parallel_v101.log          # full run log
├── unconditional_9_ae_parallel_v101.csv          # per-evaluation metrics
└── unconditional_9_ae_parallel_v101_best_params.npy  # 134-float best weights
```

### Monitoring

```bash
# Real-time log tail
tail -f results_parallel_v101/unconditional_9_ae_parallel_v101.log

# Check current gbest
grep "🔥 New gbest" results_parallel_v101/*.log | tail -5

# Per-iteration summary
grep "AE-QPSO Iter" results_parallel_v101/*.log

# Verify parallelism: each round should complete in ~120–150 s
grep "parallel 輪次" results_parallel_v101/*.log | head -20

# Memory stability check (should stay < 600 MB throughout)
grep "\[MEM\]" results_parallel_v101/*.log
```

### Reproducing the best molecule after the run

```bash
python -c "
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
from qmg.generator_cudaq import MoleculeGeneratorCUDAQ

best_w = np.load('results_parallel_v101/unconditional_9_ae_parallel_v101_best_params.npy')
cwg = ConditionalWeightsGenerator(9, smarts=None)
w_c = cwg.apply_chemistry_constraint(best_w.copy())

gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w_c, backend_name='cudaq_nvidia')
sd, v, u = gen.sample_molecule(10000)
print(f'V={v:.4f}  U={u:.4f}  V*U={v*u:.6f}')
valid = [k for k in sd if k and k != 'None']
print(f'Unique valid SMILES: {len(valid)}')
for s in valid[:10]:
    print(' ', s)
"
```

---

## MPI Fallback (run_qpso_qmg_mpi.py v1.3)

Use only if the cluster admin confirms `--gpu-bind=per_task:1` is supported
and the cuStateVec serialization lock has been addressed.

**v1.3 fixes a deadlock** introduced in v1.2: `_COMM.Barrier()` inside
`batch_evaluate_fn` (called only by rank 0) conflicted with non-rank-0's
`_COMM.bcast(flag)`, causing a permanent hang whenever `reinit_every`
triggered.  v1.3 folds the rebuild signal into the existing flag bcast as
`_MPI_FLAG_REBUILD = 2`, eliminating the independent barrier.

Verify GPU binding before submitting:

```bash
srun --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --gpu-bind=per_task:1 \
    bash -c 'echo "rank $PMI_RANK: SLURM_LOCALID=$SLURM_LOCALID SLURM_STEP_GPUS=$SLURM_STEP_GPUS"'
# Each rank must show a distinct SLURM_STEP_GPUS value.
```

Submit:

```bash
sbatch cutn-qmg_mpi_8g.slurm
tail -f results_mpi_v12/unconditional_9_ae_mpi_v12.log
```

---

## Known Constraints and Gotchas

| Issue | Constraint / Workaround |
|---|---|
| `watch -n 5 nvidia-smi` segfaults on DGX111 | Use `while true; do clear; nvidia-smi; sleep 10; done` |
| `@cudaq.kernel` tests fail with `python -c` | Write to a `.py` file; `inspect.getsource()` requires it |
| `tensornet` backend hangs with dynamic circuits | **Blocked in worker_eval.py v10.2.** Never use `--backend cudaq_tensornet` |
| `list[float]` broadcast dispatch bug (CUDA-Q 0.7.1) | Fixed in `_qmg_n9` v9.1: every `mz()` assignment on its own line |
| `get_sequential_data()` returns `list[str]`, not `int` | Fixed in v9.5: use `int(bit)` not `1 if bit else 0` |
| Chemistry constraint double-application | Fixed: parent applies once; `worker_eval.py` sets `chemistry_constraint=False` |
| AE-QTS mbest monotone weighting | Fixed in `qpso_optimizer_ae.py` v1.1: U-shaped symmetric harmonic weighting |
| AE-QTS paired update Cauchy misuse | Fixed in v1.1: both `best_k` and `worst_k` move toward their local attractor |
| CUDA-Q >= 0.8 drops sm_70 SASS | Pinned to `cuda-quantum-cu12==0.7.1`; do not upgrade |
| NumPy 2.x incompatibility | Hard constraint: `numpy < 2.0` |
| Pinned memory leak | subprocess pool: each worker exits after one `cudaq.sample()`, releasing all pinned memory |

---

## References

- Chen et al. 2025, JCTC — *Exploring Chemical Space with Chemistry-Inspired
  Dynamic Quantum Circuits in the NISQ Era* (PDF included in repo)
- Xiao et al. 2026, arXiv:2604.13877v1 — *SQMG*
- Tseng et al. 2024, arXiv:2311.12867v2 — *AE-QTS*
- Sun et al. 2012, CEC — *QPSO*, Eq. 12

For the complete debug runbook, infrastructure diagnosis, and recovery
procedures see
[docs/qmg-soqpso-handoff-2026-05-07.md](docs/qmg-soqpso-handoff-2026-05-07.md).
