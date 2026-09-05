#!/bin/bash
# ==============================================================================
# benchmark/benchmark.slurm — 單一 (演算法, seed) 的比較實驗作業（v12.0）
#
# 由 launch_benchmark.py 提交，所有參數透過 --export 傳入：
#   ALGO SEED M T SHOTS NODES OBJECTIVE DATA_DIR DISPATCH
#
# 手動提交範例：
#   sbatch -N 4 -p nchc --time=12:00:00 \
#     --export=ALL,ALGO=cmaes,SEED=0,M=32,T=16,SHOTS=5000,NODES=4 \
#     benchmark/benchmark.slurm
# ==============================================================================

#SBATCH -p nchc
#SBATCH -J sqmg.bench
#SBATCH --ntasks-per-node=1
# 注意：--gres 刻意不寫死在這裡，由 launch_benchmark.py 於提交時指定
# （批次型 2 顆、序列型 1 顆），SLURM 才能把多個 run 打包到同一節點。

set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate cudaq-v071

REPO=~/sqmg_project-cudaq
cd "$REPO"

ALGO=${ALGO:?must set ALGO}
SEED=${SEED:-0}
M=${M:-32}
T=${T:-16}
SHOTS=${SHOTS:-5000}
OBJECTIVE=${OBJECTIVE:-vu}
DATA_DIR=${DATA_DIR:-results_benchmark}
DISPATCH=${DISPATCH:-slurm}
NODES=${NODES:-$SLURM_NNODES}

TASK="${TASK_NAME:-${ALGO}_${OBJECTIVE}_M${M}T${T}_s${SEED}}"
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi

EXTRA=""
[ -n "${MAX_EVALS:-}" ]  && EXTRA="$EXTRA --max_evals $MAX_EVALS"
[ -n "${ALPHA_MAX:-}" ]  && EXTRA="$EXTRA --alpha_max $ALPHA_MAX"
[ -n "${ALPHA_MIN:-}" ]  && EXTRA="$EXTRA --alpha_min $ALPHA_MIN"
[ -n "${SHOT_SEED:-}" ]  && EXTRA="$EXTRA --shot_seed $SHOT_SEED"
[ -n "${ABLATE:-}" ]     && EXTRA="$EXTRA --ablate $ABLATE"
[ -n "${MAX_GP_POINTS:-}" ] && EXTRA="$EXTRA --max_gp_points $MAX_GP_POINTS"
[ -n "${TUNE_EVERY:-}" ]    && EXTRA="$EXTRA --tune_every $TUNE_EVERY"
[ "${SMILES_LOG:-0}" = "1" ] && EXTRA="$EXTRA --smiles_log"

# ── ax_bo 專用：跨環境執行 ──────────────────────────────────────────────
# ax-platform 0.4.3 會拉 CUDA 13 的 torch，與 cuda-quantum 0.7.1 的 CUDA 12
# 衝突，兩者無法共存於同一個 conda 環境。因此：
#   父行程（Ax 的 GP 配適與選點）跑在 ax-bo，
#   子行程（分子模擬）由 QMG_WORKER_PYTHON 指向 cudaq-v071。
# 這是唯一會用到跨環境的演算法；其餘一律留在 cudaq-v071。
PYBIN=python
if [ "$ALGO" = "ax_bo" ]; then
  AXPY=/beegfs/home/sirius/.conda/envs/ax-bo/bin/python
  CQPY=$(command -v python)                 # 此刻已 activate cudaq-v071
  if [ ! -x "$AXPY" ]; then
    echo "[FATAL] 找不到 ax-bo 直譯器：$AXPY"; exit 3
  fi
  export QMG_WORKER_PYTHON="$CQPY"
  PYBIN="$AXPY"
  echo "  跨環境：父=$AXPY  worker=$QMG_WORKER_PYTHON"
fi

echo "=== benchmark start $(date) ==="
echo "  job=$SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_NODELIST"
echo "  algo=$ALGO seed=$SEED M=$M T=$T shots=$SHOTS dispatch=$DISPATCH"

mkdir -p "$DATA_DIR"

if [ "$DISPATCH" = "local" ]; then
  "$PYBIN" run_experiment.py \
      --optimizer "$ALGO" --objective "$OBJECTIVE" \
      --M "$M" --T "$T" --seed "$SEED" \
      --num_sample "$SHOTS" \
      --dispatch local --n_gpus "${NGPUS:-8}" \
      --data_dir "$DATA_DIR" --task_name "$TASK" $RESUME_FLAG $EXTRA
else
  "$PYBIN" run_experiment.py \
      --optimizer "$ALGO" --objective "$OBJECTIVE" \
      --M "$M" --T "$T" --seed "$SEED" \
      --num_sample "$SHOTS" \
      --dispatch slurm --nodes "$NODES" --gpus_per_node 8 \
      --job_dir "$REPO/.mn_jobs/$TASK" \
      --data_dir "$DATA_DIR" --task_name "$TASK" $RESUME_FLAG $EXTRA
fi

echo "=== benchmark done $(date) ==="
