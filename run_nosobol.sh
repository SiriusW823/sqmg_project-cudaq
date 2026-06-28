#!/bin/bash
# Fig 2 fourth curve: pure QPSO with NO Sobol (random init) and NO OBL.
set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate cudaq-v071
cd ~/sqmg_project-cudaq
echo "[nosobol] waiting for ae_sweep to end... $(date)"
while tmux has-session -t ae_sweep 2>/dev/null; do sleep 120; done
echo "[nosobol] GPUs free; starting pure-QPSO (no Sobol, no OBL) $(date)"
D=results_qpso_nosobol
mkdir -p "$D"
python run_qpso_qmg_cudaq.py \
  --backend cudaq_nvidia --num_heavy_atom 9 --num_sample 5000 \
  --particles 64 --iterations 150 \
  --n_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7 --subprocess_timeout 360 \
  --no_sobol_init --no_obl --no_vu_decouple --no_ae_weighting --pair_interval 0 \
  --seed 0 \
  --task_name unconditional_9_qpso_nosobol_M64T150 --data_dir "$D" \
  > "$D/console.log" 2>&1
echo "[nosobol] DONE $(date)"
