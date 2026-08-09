#!/usr/bin/env bash
# =============================================================================
# run_m128_hbahbd.sh  — 在 DGX111 上以 tmux 跑 M=128 chemistry_constraint 實驗
#   目標：最大化 V×U（架構不變），同時量測 HBA(→4)/HBD(→3) 並寫入 log。
#   對比參考檔：chemistry_constraint_qiskit_4HBA_3HBD_0.log
#
# 用法（在 DGX111）：
#   cd /beegfs/home/sirius/sqmg_project-cudaq   # 依實際路徑
#   git pull origin main                        # 先確定已同步
#   tmux new -s m128                            # 開一個可 detach 的 session
#   bash run_m128_hbahbd.sh                     # 執行；之後可 Ctrl-b d 離開
# =============================================================================
set -euo pipefail

# 專案根目錄（腳本所在位置）
cd "$(dirname "$0")"

TASK="chemistry_constraint_cudaq_4HBA_3HBD_M128"
DATA_DIR="results_hbahbd"
mkdir -p "$DATA_DIR"

echo "[run_m128_hbahbd] 開始：$(date)"
echo "[run_m128_hbahbd] 主 log 將位於：$DATA_DIR/${TASK}.log"
echo "[run_m128_hbahbd] 逐代 HBA/HBD CSV：$DATA_DIR/${TASK}_hbahbd.csv"

PYTHONPATH=. python run_qpso_qmg_cudaq.py \
  --particles          128 \
  --iterations         150 \
  --num_heavy_atom     9 \
  --num_sample         10000 \
  --n_gpus             8 \
  --gpu_ids            0,1,2,3,4,5,6,7 \
  --backend            cudaq_nvidia \
  --hba_target         4 \
  --hbd_target         3 \
  --subprocess_timeout 900 \
  --task_name          "$TASK" \
  --data_dir           "$DATA_DIR"

echo "[run_m128_hbahbd] 結束：$(date)"
echo "[run_m128_hbahbd] 完成。請檢視 $DATA_DIR/${TASK}.log 與 ${TASK}_hbahbd.csv"
