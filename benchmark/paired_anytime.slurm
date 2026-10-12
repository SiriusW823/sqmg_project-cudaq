#!/bin/bash
# ==============================================================================
# benchmark/paired_anytime.slurm — 同節點、同時段的配對執行（兩個 arm、同一 seed）
#
# 預登記：docs/PREREGISTRATION_anytime_vs_bo.md
#
# 一個作業獨佔一個節點（8×V100），同時啟動兩個 arm：
#   arm A：GPU 0-3（4 個評估 worker）
#   arm B：GPU 4-7（ax_bo：worker 在 GPU 4、GP 在 GPU 5；其餘兩張閒置）
# 兩者共用同一節點、同一時段，節點速度與負載差異因此被配對抵消。
#
# 必填 export：ARM_A ARM_B SEED DATA_DIR TIME_BUDGET_H CHECKPOINT_HOURS
# 選填 export：SHOTS(1000) M(64) MAX_EVALS(999936) ALPHA_MAX ALPHA_MIN
#   population arm 一律 ALPHA_BY_TIME=1：α 在 24 h 預算內線性退火（§10）
#
#   sbatch -N1 -p nchc --exclusive --gres=gpu:8 --time=26:00:00 --no-requeue \
#     --export=ALL,ARM_A=rr_qpso2,ARM_B=ax_bo,SEED=500,DATA_DIR=results_anytime,\
#              TIME_BUDGET_H=24,CHECKPOINT_HOURS=2:4:8:12:16:20:24 \
#   ★ CHECKPOINT_HOURS 必須用冒號：--export 以逗號切分變數，"2,4,8" 會被截成 "2"。
#     benchmark/paired_anytime.slurm
# ==============================================================================
#SBATCH -p nchc
#SBATCH -J sqmg.pair
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

set -u
REPO=~/sqmg_project-cudaq
cd "$REPO"

ARM_A=${ARM_A:?}; ARM_B=${ARM_B:?}; SEED=${SEED:?}
DATA_DIR=${DATA_DIR:?}; TIME_BUDGET_H=${TIME_BUDGET_H:?}; CHECKPOINT_HOURS=${CHECKPOINT_HOURS:?}
SHOTS=${SHOTS:-1000}; M=${M:-64}; MAX_EVALS=${MAX_EVALS:-999936}   # 時間是唯一預算（64 的倍數）
mkdir -p "$DATA_DIR"

case "$CHECKPOINT_HOURS" in
  *,*) echo "[FATAL] CHECKPOINT_HOURS 含逗號（$CHECKPOINT_HOURS）；請用冒號分隔"; exit 3 ;;
esac
echo "  budget=${TIME_BUDGET_H}h  checkpoints=${CHECKPOINT_HOURS}  shots=${SHOTS}  max_evals=${MAX_EVALS}"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "=== paired start $(date)  host=$(hostname)  job=${SLURM_JOB_ID:-}  GPUs visible=$NGPU ==="
if [ "$NGPU" -lt 8 ]; then
  echo "[FATAL] 需要整個節點 8 張 GPU，只看到 $NGPU"; exit 3
fi

launch () {   # $1=arm  $2=GPU_IDS  $3=NGPUS  $4=額外 export
  local ARM=$1
  local TASK="any_${ARM}_s${SEED}"
  env ALGO="$ARM" SEED="$SEED" M="$M" T=$(( MAX_EVALS / M )) MAX_EVALS="$MAX_EVALS" \
      SHOTS="$SHOTS" OBJECTIVE=vu DATA_DIR="$DATA_DIR" DISPATCH=local RESUME=0 \
      NGPUS="$3" GPU_IDS="$2" SHOT_SEED="$SEED" FRESH_SHOT_NOISE=1 \
      TIME_BUDGET_H="$TIME_BUDGET_H" CHECKPOINT_HOURS="$CHECKPOINT_HOURS" \
      TASK_NAME="$TASK" QMG_RECYCLE_MINUTES=18 QMG_RECYCLE_EVERY=0 $4 \
      bash benchmark/benchmark.slurm > "$DATA_DIR/console_${TASK}.log" 2>&1 &
  echo "  launched $ARM  GPUs=$2  pid=$!  log=$DATA_DIR/console_${TASK}.log"
}

arm_env () {  # 依 arm 給出專屬設定（population arm 帶 α 覆寫；ax_bo 指定 GP 卡）
  case "$1" in
    qpso|rr_qpso2) echo "ALPHA_MAX=${ALPHA_MAX:-1.2} ALPHA_MIN=${ALPHA_MIN:-0.3} ALPHA_BY_TIME=1" ;;
    ax_bo)         echo "AX_TORCH_DEVICE=cuda:5" ;;
    *)             echo "" ;;
  esac
}

gpus_for () {  # $1=arm $2=A|B
  if [ "$1" = "ax_bo" ]; then
    [ "$2" = "A" ] && echo "0 1" || echo "4 1"
  else
    [ "$2" = "A" ] && echo "0,1,2,3 4" || echo "4,5,6,7 4"
  fi
}

read GA NA <<< "$(gpus_for "$ARM_A" A)"
read GB NB <<< "$(gpus_for "$ARM_B" B)"
EA=$(arm_env "$ARM_A"); EB=$(arm_env "$ARM_B")
if [ "$ARM_A" = "ax_bo" ]; then EA="AX_TORCH_DEVICE=cuda:1"; fi

launch "$ARM_A" "$GA" "$NA" "$EA"
launch "$ARM_B" "$GB" "$NB" "$EB"
wait
echo "=== paired done $(date) ==="
