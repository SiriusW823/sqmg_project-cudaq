#!/bin/bash
# Figure 3 particle-count sweep: AE-QPSO (V8 method) at varying M, full length T=150.
# Runs sequentially AFTER the pure-QPSO run (tmux session 'qpso_pure') finishes.
set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate cudaq-v071
cd ~/sqmg_project-cudaq

echo "[sweep] waiting for qpso_pure tmux session to end... $(date)"
while tmux has-session -t qpso_pure 2>/dev/null; do sleep 300; done
echo "[sweep] GPUs free; starting AE-QPSO particle sweep $(date)"

# ascending M -> cheap points first (M=64 reuses results_v8)
for M in 16 32 48 96 128; do
  D=results_sweep_M${M}
  mkdir -p "$D"
  echo "[sweep] ===== M=${M} START $(date) ====="
  python run_qpso_qmg_cudaq.py \
    --backend cudaq_nvidia --num_heavy_atom 9 --num_sample 5000 \
    --particles ${M} --iterations 150 \
    --n_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7 --subprocess_timeout 360 \
    --seed 0 \
    --task_name unconditional_9_ae_M${M}T150 --data_dir "$D" \
    > "$D/console.log" 2>&1
  echo "[sweep] ===== M=${M} DONE $(date) rc=$? ====="
done
echo "[sweep] ALL SWEEP RUNS COMPLETE $(date)"
