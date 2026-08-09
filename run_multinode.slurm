#!/bin/bash
# ==============================================================================
# run_multinode.slurm — RR-QPSO 多節點執行（v11.0）
#
# 目的：把「一次模擬可用的最大 GPU 數」從單節點 8 張，提升為 <NODES> × 8 張。
#
# 架構（與 gpu_scaling_bench.slurm 已驗證的模式相同）：
#     sbatch 配額（N 節點 × 8 GPU）
#       └── 批次腳本在第 0 個節點執行 run_qpso_qmg_cudaq.py --dispatch slurm
#             └── 每輪一次 srun -N n --ntasks-per-node=1 --gres=gpu:8
#                   └── 每節點一個 node_agent.py
#                         └── 每 GPU 一個 worker_eval.py
#
# 為何不用 SSH：叢集 node-to-node SSH 被封鎖（只能經跳板機），
#               但 srun 在配額內可直接跨節點啟動 step。
#
# 提交範例（4 節點 = 32 GPU，M=128 → 每批次 4 輪，相對單節點 4× 加速）：
#
#   sbatch -N 4 -J rrqpso.mn4 \
#          -o results_multinode/console_M128_N4.log \
#          --export=ALL,NODES=4,M=128,T=150,SHOTS=5000,TASK=rrqpso_mn_M128 \
#          run_multinode.slurm
#
# 監控：
#   squeue -u $USER
#   tail -f results_multinode/rrqpso_mn_M128.log
#
# 注意：
#   1. --time 必須涵蓋整個長跑。M=128 / T=150 / 32 GPU 約需 ⌈128/32⌉=4 輪
#      × 151 批次 × ~142s ≈ 24 小時，下方預設留 48 小時。
#   2. job_dir 必須在 beegfs（共享）上，不可用 /tmp。程式會主動擋下 /tmp。
#   3. 節點數不要超過實際空閒節點，否則整個 job 會排隊等待。
# ==============================================================================

#SBATCH -p nchc
#SBATCH -J rrqpso.mn
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

set -u

source /opt/conda/etc/profile.d/conda.sh
conda activate cudaq-v071

REPO=~/sqmg_project-cudaq
cd "$REPO"

# ── 參數（可由 --export 覆寫）─────────────────────────────────────────────
NODES=${NODES:-$SLURM_NNODES}
M=${M:-128}
T=${T:-150}
SHOTS=${SHOTS:-5000}
NHEAVY=${NHEAVY:-9}
TASK=${TASK:-rrqpso_mn_M${M}_N${NODES}}
DATA_DIR=${DATA_DIR:-results_multinode}

mkdir -p "$DATA_DIR"

echo "=== multinode run start $(date) ==="
echo "  SLURM_JOB_ID   : $SLURM_JOB_ID"
echo "  SLURM_NNODES   : $SLURM_NNODES"
echo "  SLURM_NODELIST : $SLURM_NODELIST"
echo "  NODES(dispatch): $NODES   總並行 GPU: $((NODES * 8))"
echo "  M=$M  T=$T  shots=$SHOTS  heavy_atom=$NHEAVY  task=$TASK"

# 安全檢查：要求的節點數不能超過 SLURM 實際配給的節點數，
# 否則 srun 會在每一輪都嘗試取用不存在的節點而失敗（且要跑到中途才會發現）。
if [ "$NODES" -gt "$SLURM_NNODES" ]; then
  echo "ERROR: NODES=$NODES 超過配額 SLURM_NNODES=$SLURM_NNODES，請以 sbatch -N $NODES 提交" >&2
  exit 1
fi

python run_qpso_qmg_cudaq.py \
    --dispatch      slurm \
    --nodes         "$NODES" \
    --gpus_per_node 8 \
    --job_dir       "$REPO/.mn_jobs/$TASK" \
    --particles     "$M" \
    --iterations    "$T" \
    --num_sample    "$SHOTS" \
    --num_heavy_atom "$NHEAVY" \
    --backend       cudaq_nvidia \
    --task_name     "$TASK" \
    --data_dir      "$DATA_DIR"

echo "=== multinode run done $(date) ==="
