#!/usr/bin/env bash
# =============================================================================
# run_hbahbd_multiobj.sh
#
# Usage on DGX:
#   bash run_hbahbd_multiobj.sh 128
#   bash run_hbahbd_multiobj.sh 16
#
# Runs the CUDA-Q AE-SOQPSO HBA/HBD multi-objective experiment:
#   maximize V*U while minimizing distance to HBA=4 and HBD=3.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

M="${1:-128}"
TASK="chemistry_constraint_cudaq_multiobj_4HBA_3HBD_M${M}"
DATA_DIR="results_hbahbd_multiobj"
mkdir -p "$DATA_DIR"

echo "[run_hbahbd_multiobj] start: $(date)"
echo "[run_hbahbd_multiobj] M=${M}"
echo "[run_hbahbd_multiobj] main log: ${DATA_DIR}/${TASK}.log"
echo "[run_hbahbd_multiobj] multi-objective csv: ${DATA_DIR}/${TASK}_multiobj.csv"

PYTHONPATH=. python run_qpso_qmg_cudaq_hbahbd_multiobj.py \
  --particles          "${M}" \
  --iterations         150 \
  --num_heavy_atom     9 \
  --num_sample         10000 \
  --n_gpus             8 \
  --gpu_ids            0,1,2,3,4,5,6,7 \
  --backend            cudaq_nvidia \
  --hba_target         4 \
  --hbd_target         3 \
  --hba_sigma          1.0 \
  --hbd_sigma          1.0 \
  --chem_weight        0.40 \
  --subprocess_timeout 900 \
  --task_name          "${TASK}" \
  --data_dir           "${DATA_DIR}"

echo "[run_hbahbd_multiobj] end: $(date)"
