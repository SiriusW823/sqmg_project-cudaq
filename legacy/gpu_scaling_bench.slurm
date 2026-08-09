#!/bin/bash
#SBATCH -p nchc
#SBATCH -J gpuscale
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
# submit with:  sbatch -N <nodes> -o results_gpuscale/G<G>.out --export=ALL,G=<G> gpu_scaling_bench.slurm
set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate cudaq-v071
cd ~/sqmg_project-cudaq
mkdir -p results_gpuscale
N=$SLURM_NNODES
NLOCAL=$(( G < 8 ? G : 8 ))
OUT=results_gpuscale/G${G}.txt
echo "=== G=$G nodes=$N nlocal_per_node=$NLOCAL nodelist=$SLURM_NODELIST $(date) ===" > "$OUT"
# build a shared weight vector once
python -c "import numpy as np,sys; sys.path.insert(0,'.'); from qmg.utils import ConditionalWeightsGenerator; np.save('bench_w.npy', ConditionalWeightsGenerator(9,smarts=None).generate_conditional_random_weights(random_seed=42)); print('weight ok')" >> "$OUT" 2>&1
for rep in 1 2 3 4; do
  t0=$(date +%s.%N)
  srun -N "$N" --ntasks-per-node=1 --gres=gpu:$NLOCAL python bench_node.py $NLOCAL >> "$OUT" 2>&1
  t1=$(date +%s.%N)
  awk "BEGIN{printf \"G=$G rep=$rep round_wall=%.2f\n\", $t1-$t0}" >> "$OUT"
done
echo "G=$G DONE $(date)" >> "$OUT"
