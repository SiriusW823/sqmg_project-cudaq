#!/bin/bash
# ==============================================================================
# cutn-qmg_mpi_8g.slurm  (v1.2 GPU 綁定修正版)
# QMG AE-SOQPSO + MPI 並行實驗 — DGX V100-SXM2-16GB × 8
#
# v1.1 → v1.2 根本修正：
#
#   ★ 問題確認（from full_run.txt 2026-04-24 實驗）：
#       50 粒子 Phase 0 耗時 6129s（序列 50 × 122.6s）
#       8 GPU 真正並行應為 858s（⌈50/8⌉=7輪 × 122.6s）
#       根本原因：SLURM 沒有為每個 task 分配獨立 GPU cgroup
#
#   ★ 修正一：加入 --gpu-bind=per_task:1
#       讓 SLURM 為每個 MPI task 分配獨立的 GPU，並設定對應的 cgroup。
#       這樣 SLURM_STEP_GPUS（或 SLURM_LOCALID）才能正確反映每個 task 的 GPU。
#
#   ★ 修正二：SLURM_LOCALID 環境變數傳遞
#       run_qpso_qmg_mpi.py v1.2 在模組頂層（import cudaq 之前）讀取
#       SLURM_LOCALID 並設定 CUDA_VISIBLE_DEVICES。
#       SLURM 會自動在每個 task 的環境中設定 SLURM_LOCALID。
#
#   ★ 修正三：移除 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#       v1.1 在 slurm 腳本設定 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7，
#       這讓所有 task 看到全部 8 張 GPU，
#       程式碼才需要根據 rank 選擇，但這在 cgroup 保護下會失效。
#       v1.2 不設定此環境變數，讓 SLURM 的 --gpu-bind 自動處理。
#
# 提交：sbatch cutn-qmg_mpi_8g.slurm
# 監控：squeue -u $USER
#       tail -f results_mpi/unconditional_9_ae_mpi.log
#
# 預估時間（8 卡 V100，cudaq_nvidia 後端）：
#   每輪：⌈50/8⌉=7 批 × 122.6s ≈ 858s ≈ 14 分鐘
#   Phase 0：858s
#   40 迭代：40 × 858s = 34,320s ≈ 9.5 小時
#   總計（含 Phase 0）：41 × 858s ≈ 9.7 小時
#
#   與 v1.1 失效版比較：
#     v1.1（序列化）：41 × 50 × 122.6s = 251,330s ≈ 69.8 小時（中途 OOM）
#     v1.2（修正版）：41 × 7 × 122.6s  ≈  35,210s ≈  9.8 小時（無 OOM）
# ==============================================================================

#SBATCH -J qmg.ae.mpi.v12          # 作業名稱（v1.2 修正版）
#SBATCH -p nchc                     # 佇列
#SBATCH --nodes=1                   # 單節點 DGX
#SBATCH --ntasks-per-node=8         # 8 個 MPI rank
#SBATCH --cpus-per-task=4           # 每 rank 4 CPU（RDKit 用）
#SBATCH --gres=gpu:8                # 申請 8 張 GPU
#SBATCH --gpu-bind=per_task:1       # ★ v1.2 關鍵：每個 task 綁定 1 張獨立 GPU
#SBATCH --mem-bind=no

# ── 環境初始化（與原始設定一致）────────────────────────────────────────────
ml purge
ml cuq/12

srun hostname

source /beegfs/_venv/cuq23cu12/bin/activate

# ── ★ v1.2 修正：不設定 CUDA_VISIBLE_DEVICES ───────────────────────────────
# 讓 SLURM 的 --gpu-bind=per_task:1 處理 GPU 分配，
# run_qpso_qmg_mpi.py v1.2 透過 SLURM_LOCALID + SLURM_STEP_GPUS 取得 GPU index。
# 注意：若你的 SLURM 版本不支援 --gpu-bind=per_task:1，
#       可改用以下方式（需確認叢集管理員是否支援）：
#         --gpu-bind=closest 或 --gpu-bind=map_gpu:0,1,2,3,4,5,6,7
# unset CUDA_VISIBLE_DEVICES  # 確保不覆蓋 SLURM 的設定

# ── CUDA-Q 設定 ─────────────────────────────────────────────────────────────
export CUDAQ_REQUIRE_CUDA=1          # 強制 GPU，不 fallback 到 CPU

# ── Python 路徑 ─────────────────────────────────────────────────────────────
export PYTHONPATH=/beegfs/home/${USER}/sqmg_project-cudaq:${PYTHONPATH}

# ── mpi4py 設定 ─────────────────────────────────────────────────────────────
export MPICC=$(which mpicc)

# ── GPU 綁定確認（debug 用，提交前可移除）───────────────────────────────────
echo "=== GPU 綁定確認 ==="
srun -n 8 bash -c 'echo "Rank $PMI_RANK (SLURM_LOCALID=$SLURM_LOCALID): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, SLURM_STEP_GPUS=$SLURM_STEP_GPUS"'
echo "==================="

# ==============================================================================
# ★ 方案 A：cudaq_nvidia 後端 + MPI 8 GPU（推薦，已知穩定）
# 50 粒子 × 41 輪 = 2050 次評估
# 預估時間：~9.7 小時（v1.2 修正後真正並行）
# ==============================================================================
mpirun -n 8 python run_qpso_qmg_mpi.py \
    --backend           cudaq_nvidia     \
    --num_heavy_atom    9                \
    --num_sample        10000            \
    --particles         50               \
    --iterations        40               \
    --alpha_max         1.2              \
    --alpha_min         0.4              \
    --mutation_prob     0.12             \
    --stagnation_limit  8                \
    --reinit_fraction   0.20             \
    --ae_weighting                       \
    --pair_interval     5                \
    --rotate_factor     0.01             \
    --reinit_every      10               \
    --seed              42               \
    --task_name         unconditional_9_ae_mpi_v12 \
    --data_dir          results_mpi_v12

echo "---- Walltime (方案 A): ${SECONDS} sec."

# ==============================================================================
# ★ 若 --gpu-bind=per_task:1 不被支援，改用此方案（手動指定 GPU map）：
# #SBATCH --gpu-bind=map_gpu:0,1,2,3,4,5,6,7
# 或使用 run_qpso_qmg_cudaq.py v10.1（parallel subprocess，不依賴 MPI）：
#
# python run_qpso_qmg_cudaq.py \
#     --backend      cudaq_nvidia                     \
#     --n_gpus       8                                \
#     --gpu_ids      0,1,2,3,4,5,6,7                 \
#     --particles    50                               \
#     --iterations   40                               \
#     --num_sample   10000                            \
#     --task_name    unconditional_9_parallel_v101    \
#     --data_dir     results_parallel_v101
# ==============================================================================

# ==============================================================================
# ★ 方案 B：tensornet 後端（若支援 + mid-circuit 無問題）
# ==============================================================================
# mpirun -n 8 python run_qpso_qmg_mpi.py \
#     --backend           cudaq_tensornet  \
#     --num_heavy_atom    9                \
#     --num_sample        10000            \
#     --particles         50               \
#     --iterations        40               \
#     --alpha_max         1.2              \
#     --alpha_min         0.4              \
#     --mutation_prob     0.15             \
#     --stagnation_limit  8                \
#     --ae_weighting                       \
#     --pair_interval     5                \
#     --reinit_every      10               \
#     --seed              42               \
#     --task_name         unconditional_9_ae_mpi_tn_v12 \
#     --data_dir          results_mpi_tn_v12
# echo "---- Walltime (方案 B tensornet): ${SECONDS} sec."

# ==============================================================================
# 快速功能驗證（提交正式實驗前先確認 GPU 綁定正確）：
#   srun -n 8 python run_qpso_qmg_mpi.py \
#       --backend cudaq_nvidia \
#       --particles 8 --iterations 1 --num_sample 5 \
#       --task_name sanity_v12 --data_dir results_sanity_v12
# ==============================================================================
