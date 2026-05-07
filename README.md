# sqmg_project-cudaq

CUDA-Q implementation of QMG molecular generation with AE-SOQPSO optimization.

This repository is currently tuned for the DGX111 V100 environment and the
`run_qpso_qmg_cudaq.py` v10.1 parallel subprocess workflow. For the complete
handoff, debug flow, monitoring commands, and troubleshooting notes, read:

[docs/qmg-soqpso-handoff-2026-05-07.md](docs/qmg-soqpso-handoff-2026-05-07.md)

## Goal

The experiment aims to exceed the Chen et al. 2025 JCTC Bayesian Optimization
baseline for 9-heavy-atom QMG generation:

```text
Target metric: validity * uniqueness > 0.8834
Paper baseline: V = 0.955, U = 0.925, V*U = 0.8834
```

The current approach replaces BO with AE-SOQPSO and evaluates CUDA-Q dynamic
circuits through isolated subprocess workers.

## Required Environment

Use the fixed Conda environment on DGX111:

```bash
conda activate cudaq-v071
python --version
python -c "import cudaq; print(cudaq.__version__)"
```

Required constraints:

- Python 3.10
- `cuda-quantum-cu12==0.7.1`
- `numpy>=1.24,<2.0`
- `rdkit>=2023.9.5`

CUDA-Q must stay at 0.7.1 on V100 because newer prebuilt CUDA-Q wheels do not
support Volta `sm_70` correctly for this workflow.

## Main Entry Point

Use `run_qpso_qmg_cudaq.py` v10.1. It launches one `worker_eval.py`
subprocess per GPU evaluation so CUDA contexts are destroyed after each sample,
which avoids the pinned-memory leak observed with long-lived CUDA-Q 0.7.1
processes.

The MPI path, `run_qpso_qmg_mpi.py`, is retained as a fallback only.

## Quick Debug Flow

```bash
cd ~/sqmg_project-cudaq
git pull origin main

python -c "
import cudaq
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
from qpso_optimizer_ae import AESOQPSOOptimizer
print('imports ok')
print('cudaq version:', cudaq.__version__)
print('numpy version:', np.__version__)
"
```

Create a smoke-test weight file and evaluate it on one GPU:

```bash
python -c "
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
cwg = ConditionalWeightsGenerator(9, smarts=None)
w = cwg.generate_conditional_random_weights(random_seed=42)
np.save('/tmp/smoke_w.npy', w)
print('weight len =', len(w))
"

CUDA_VISIBLE_DEVICES=0 python worker_eval.py \
    --weight_path /tmp/smoke_w.npy \
    --result_path /tmp/smoke_r.npy \
    --num_heavy_atom 9 \
    --num_sample 100 \
    --backend cudaq_nvidia
```

Run the built-in multi-GPU sanity check:

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

## Full Experiment

```bash
python run_qpso_qmg_cudaq.py \
    --backend             cudaq_nvidia \
    --num_heavy_atom      9 \
    --num_sample          10000 \
    --particles           50 \
    --iterations          40 \
    --n_gpus              8 \
    --gpu_ids             0,1,2,3,4,5,6,7 \
    --subprocess_timeout  600 \
    --alpha_max           1.2 \
    --alpha_min           0.4 \
    --mutation_prob       0.12 \
    --stagnation_limit    8 \
    --reinit_fraction     0.20 \
    --ae_weighting \
    --pair_interval       5 \
    --rotate_factor       0.01 \
    --seed                42 \
    --task_name           unconditional_9_ae_parallel_v101 \
    --data_dir            results_parallel_v101
```

Expected outputs:

- `results_parallel_v101/unconditional_9_ae_parallel_v101.log`
- `results_parallel_v101/unconditional_9_ae_parallel_v101.csv`
- `results_parallel_v101/unconditional_9_ae_parallel_v101_best_params.npy`

## Monitoring Notes

On DGX111, avoid `watch -n 5 nvidia-smi` because it can segfault. Use:

```bash
while true; do clear; nvidia-smi; sleep 10; done
```

For details on expected timing, log patterns, memory behavior, and recovery
limitations, see the full runbook linked above.
