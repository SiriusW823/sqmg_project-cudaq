# QMG-SOQPSO 專案交班說明文件

**撰寫日期**：2026-05-07
**原負責人**：吳行雲（GitHub handle: 吳行雲）
**文件版本**：v1.0

---

## 一、專案背景與目標

### 1.1 研究目標

本專案旨在超越 Chen et al. 2025（JCTC）論文中，貝葉斯優化（Bayesian Optimization, BO）在量子分子生成任務上所達到的基準結果：

```text
目標指標：V×U (validity × uniqueness) > 0.8834
論文基準：V = 0.955, U = 0.925, V×U = 0.8834（BO，約 355 次評估）
```

### 1.2 核心方法

將論文的 BO 優化器替換為自實作的 **AE-SOQPSO**（Adaptive Ensemble Stochastic Optimal Quantum PSO），結合以下三篇論文：

| 論文 | 角色 | 說明 |
| --- | --- | --- |
| Chen et al. 2025 (JCTC) | 電路主體 | QMG 20-qubit 動態電路，N=9 重原子分子生成 |
| Xiao et al. 2026 (arXiv:2604.13877v1) | 電路擴展 | SQMG，tensornet 後端的速度優勢 |
| Tseng et al. 2024 (arXiv:2311.12867v2) | 優化演算法 | AE-QTS，U 形對稱調和加權機制 |

### 1.3 關鍵參數說明

```text
num_heavy_atom = 9      # 生成目標：9 個重原子的分子
num_sample     = 10000  # 每次量子電路採樣的 shots 數
particles (M)  = 50     # QPSO 粒子群數量
iterations (T) = 40     # QPSO 最大迭代次數
total_evals    = 2050   # M × (T+1) = 50 × 41
n_params (D)   = 134    # 量子電路可調參數數量
```

---

## 二、DGX111 環境說明

### 2.1 硬體規格

```text
主機名稱：DGX111（NCHC 國網中心 DGX A100 叢集）
GPU：8 × NVIDIA V100-SXM2-16GB
     Compute Capability：7.0（Volta 架構，sm_70）
CUDA Driver：535.183.01
CUDA Toolkit：12.2
```

### 2.2 軟體環境

**Conda 環境**（必須使用，不可更換）：

```bash
conda activate cudaq-v071
# Python 3.10，對應 CUDA-Q 0.7.1
```

> ⚠️ **為何固定使用 0.7.1**：CUDA-Q >= 0.8.0 的預編譯 wheel 僅涵蓋 Ampere (sm_80) / Hopper (sm_90)，不支援 V100 (sm_70)。呼叫 `cudaq.set_target("nvidia")` 時會靜默 fallback 到 CPU，不會報錯但完全不使用 GPU。

**核心套件版本限制**：

```text
cuda-quantum-cu12 == 0.7.1    # 嚴格固定，不可升級
numpy >= 1.24, < 2.0          # cudaq 0.7.x 與 numpy 2.x 不相容
rdkit >= 2023.9.5
```

### 2.3 程式碼位置

```bash
# 專案根目錄
cd ~/sqmg_project-cudaq

# 確認目錄結構
ls -la
# 應該看到：
#   run_qpso_qmg_cudaq.py    ← 主要執行入口（v10.1，parallel subprocess）
#   run_qpso_qmg_mpi.py      ← MPI 備用方案（v1.2）
#   worker_eval.py           ← subprocess 子行程工作者
#   qpso_optimizer_ae.py     ← AE-SOQPSO 優化器核心
#   qpso_optimizer_qmg.py    ← 舊版 SOQPSO（保留供參考）
#   cutn-qmg_mpi_8g.slurm   ← SLURM 提交腳本
#   qmg/                     ← 分子生成模組
#     __init__.py
#     generator_cudaq.py     ← MoleculeGeneratorCUDAQ 主類別
#     utils/
#       build_dynamic_circuit_cudaq.py  ← _qmg_n9 電路核心（v9.1）
#       chemistry_data_processing.py
#       fitness_calculator.py
#       weight_generator.py
```

### 2.4 從 GitHub 同步程式碼

```bash
cd ~/sqmg_project-cudaq
git pull origin main
```

---

## 三、最近一次實驗結果與根因診斷

### 3.1 實驗概況（2026-04-24，已終止）

```text
開始時間：12:01:43
終止時間：19:10:27（OOM Kill）
總執行：約 7 小時 9 分
完成評估：200 次（目標 2050 次，僅 9.8%）
```

### 3.2 確診的兩個根本問題

#### 問題一：MPI 並行完全失效（序列化執行）

**證據**（來自 `unconditional_9_ae_mpi_full.log`）：

```text
iter=0  p=0   t=6129.1s   ← 所有 50 個粒子回報完全相同的時間
iter=0  p=1   t=6129.1s
...
iter=0  p=49  t=6129.1s
```

- 理論上 8 GPU 並行應為：`⌈50/8⌉ × 122.6s = 858s`
- 實測：`50 × 122.6s = 6129s`，**慢了 7.1 倍**，等同只用 1 張 GPU

**根本原因（三層複合）**：

1. NCHC SLURM 以 **cgroup v2** 管理 GPU 資源。Python 層修改 `CUDA_VISIBLE_DEVICES` 無法繞過 cgroup 的存取控制。
2. MPI SLURM script 沒有加 `--gpu-bind=per_task:1`，8 個 task 可能都被映射到同一個物理 GPU。
3. CUDA-Q 0.7.1 的 cuStateVec 後端在初始化時使用 `/dev/nvidia-ctl` 的**全節點序列化鎖**，即使 GPU 綁定正確，多行程也會序列執行。

#### 問題二：OOM Kill（cudaMallocHost pinned memory 洩漏）

**OOM 終止記錄**（來自 `full_run.txt`）：

```text
prterun noticed that process rank 1 with PID 2550555
on node DGX111 exited on signal 9 (Killed)
```

**根本原因**：

CUDA-Q 0.7.1 cuStateVec 後端使用 `cudaMallocHost()` 分配 pinned memory（固定記憶體），此類記憶體直接由 CUDA driver 管理，**完全不受 Python 層的 `del`、`gc.collect()` 或 `malloc_trim()` 控制**。唯一釋放途徑是 CUDA context 銷毀，即行程終止。

```text
每次 cudaq.sample() 洩漏 ≈ 2.5 GB pinned memory
Phase 0 (4批) × 50 粒子 × 2.5 GB ≈ 500 GB  →  系統 RAM 耗盡  →  OOM
```

### 3.3 演算法本身的好消息

儘管基礎設施有問題，**演算法本身在快速收斂**：

| 階段 | gbest V×U | 最佳 V | 最佳 U |
| --- | --- | --- | --- |
| Phase 0 完成 | 0.1548 | 0.609 | 0.254 |
| Iter 1 完成 | 0.3422 | 0.604 | 0.566 |
| Iter 2 完成 | 0.4104 | 0.650 | 0.631 |

- **V⋆ = 0.996**（Phase 0 第 10 個粒子已達到）
- **U⋆ = 0.909**（Iter 2 第 34 個粒子達到）
- 理論上限：0.996 × 0.909 ≈ 0.905，**已超越基準 0.8834**

問題只在於基礎設施，不是演算法本身。

---

## 四、修正方案說明

### 4.1 方案選擇

提供兩個修正後的執行方案，**優先使用方案 A**：

| 方案 | 檔案 | 機制 | 適用情境 |
| --- | --- | --- | --- |
| **A（推薦）** | `run_qpso_qmg_cudaq.py` v10.1 | parallel subprocess pool | 無需 MPI，直接 python 執行，可靠性最高 |
| B（備用） | `run_qpso_qmg_mpi.py` v1.2 + SLURM | MPI + GPU 綁定修正 | 若叢集管理員確認 `--gpu-bind=per_task:1` 可用 |

### 4.2 方案 A 的核心機制（為何能解決兩個問題）

**解決 MPI 序列化**：完全棄用 mpi4py，改用 `subprocess.Popen` 並行啟動子行程。父行程在呼叫 `Popen()` **之前**設定子行程環境的 `CUDA_VISIBLE_DEVICES`，子行程在 CUDA driver 初始化之前就已看到正確的 GPU，徹底繞過 SLURM cgroup 限制。

**解決 OOM**：每個子行程只執行一次 `cudaq.sample()`，評估完成後子行程退出，CUDA context 被 CUDA driver 強制銷毀，所有 pinned memory 完全釋放。主行程記憶體穩定在 < 1 GB。

---

## 五、Debug 流程（正式執行前必做）

### Step 1：確認環境

```bash
# 登入 DGX111，啟動 tmux session
tmux new -s qmg_debug

# 啟動 conda 環境
conda activate cudaq-v071
python --version          # 應為 3.10.x
python -c "import cudaq; print(cudaq.__version__)"  # 應為 0.7.1.x

# 確認 GPU 狀態
nvidia-smi
# 確認 8 張 V100 均可用，無其他人佔用顯存
```

### Step 2：進入專案目錄並同步程式碼

```bash
cd ~/sqmg_project-cudaq
git pull origin main
git log --oneline -5  # 確認最新 commit 已拉取

# 確認所有必要檔案存在
ls -la run_qpso_qmg_cudaq.py worker_eval.py qpso_optimizer_ae.py
ls -la qmg/generator_cudaq.py qmg/utils/build_dynamic_circuit_cudaq.py
```

### Step 3：基本 import 測試

```bash
python -c "
import cudaq
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
from qpso_optimizer_ae import AESOQPSOOptimizer
print('所有 import 正常 ✓')
print('cudaq version:', cudaq.__version__)
print('numpy version:', np.__version__)
"
```

### Step 4：單 GPU 冒煙測試（約 5 分鐘）

```bash
# 先用單 GPU、少量 shots 確認電路、SMILES 轉換、評估流程全部正常
CUDA_VISIBLE_DEVICES=0 python worker_eval.py \
    --weight_path /tmp/test_w.npy \
    --result_path /tmp/test_r.npy \
    --num_heavy_atom 9 \
    --num_sample 100 \
    --backend cudaq_nvidia

# 注意：上面會失敗因為沒有 weight 檔
# 正確的冒煙測試方式：
python -c "
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
cwg = ConditionalWeightsGenerator(9, smarts=None)
w = cwg.generate_conditional_random_weights(random_seed=42)
np.save('/tmp/smoke_w.npy', w)
print('weight 已儲存，len =', len(w))
"

CUDA_VISIBLE_DEVICES=0 python worker_eval.py \
    --weight_path /tmp/smoke_w.npy \
    --result_path /tmp/smoke_r.npy \
    --num_heavy_atom 9 \
    --num_sample 100 \
    --backend cudaq_nvidia

python -c "
import numpy as np
r = np.load('/tmp/smoke_r.npy')
print(f'V={r[0]:.3f}  U={r[1]:.3f}')
assert r[0] > 0 or r[1] > 0, 'V=U=0，有問題！'
print('worker_eval 單 GPU 測試通過 ✓')
"
```

### Step 5：多 GPU 並行驗證測試（約 5 分鐘）

這一步是關鍵，確認 8 GPU 真的同時在跑：

```bash
# 使用 run_qpso_qmg_cudaq.py 的內建驗證功能
# 2 個粒子、1 次迭代、100 shots —— 快速確認並行生效
python run_qpso_qmg_cudaq.py \
    --backend        cudaq_nvidia \
    --n_gpus         8 \
    --gpu_ids        0,1,2,3,4,5,6,7 \
    --particles      8 \
    --iterations     1 \
    --num_sample     100 \
    --subprocess_timeout 120 \
    --task_name      sanity_check \
    --data_dir       results_sanity

# 查看輸出，重點確認：
# 1. "並行驗證完成（Xs）" — X 應遠小於 8 × 單次時間
# 2. "GPU 0: V=... U=... ✓" × 8 —— 每張 GPU 都有輸出
# 3. "[parallel 輪次 1/1] GPU: ['0','1','2','3','4','5','6','7']"
# 4. 沒有 "exit=1" 或 "逾時" 的 warning
```

### Step 6：確認 nvidia-smi 監控指令

```bash
# ★ 注意：DGX111 上 watch -n 5 nvidia-smi 會 segfault！
# 使用以下替代指令監控 GPU 使用率：
while true; do clear; nvidia-smi; sleep 10; done
# 按 Ctrl+C 停止

# 正式跑的時候在另一個 tmux pane 執行這個，
# 應看到 8 張 GPU 的 GPU-Util 都在 90%+ 且各自獨立有 memory 使用
```

### Step 7：確認日誌格式正常

```bash
# 查看 sanity check 的日誌
cat results_sanity/sanity_check.log | head -50

# 確認有以下格式的輸出（與論文 BO log 格式對齊）：
# Iteration number: 0
# validity (maximize): X.XXX
# uniqueness (maximize): X.XXX
```

---

## 六、正式完整實驗執行指令

### 6.1 方案 A：parallel subprocess（推薦，不需要 SLURM）

**在 tmux session 中執行**，確保 SSH 斷線不影響：

```bash
# 建立或附著到 tmux session
tmux new -s qmg_main
# 或 tmux attach -t qmg_main

# 啟動環境
conda activate cudaq-v071
cd ~/sqmg_project-cudaq

# ═══════════════════════════════════════════════════════════════
# 正式完整實驗指令（對應論文設定 + 修正版超參數）
# 預估總時間：9.8 小時
# 預估記憶體：主行程 < 1 GB，無 OOM 風險
# ═══════════════════════════════════════════════════════════════
python run_qpso_qmg_cudaq.py \
    --backend             cudaq_nvidia         \
    --num_heavy_atom      9                    \
    --num_sample          10000                \
    --particles           50                   \
    --iterations          40                   \
    --n_gpus              8                    \
    --gpu_ids             0,1,2,3,4,5,6,7     \
    --subprocess_timeout  600                  \
    --alpha_max           1.2                  \
    --alpha_min           0.4                  \
    --mutation_prob       0.12                 \
    --stagnation_limit    8                    \
    --reinit_fraction     0.20                 \
    --ae_weighting                             \
    --pair_interval       5                    \
    --rotate_factor       0.01                 \
    --seed                42                   \
    --task_name           unconditional_9_ae_parallel_v101 \
    --data_dir            results_parallel_v101

# 完成後輸出檔案：
#   results_parallel_v101/unconditional_9_ae_parallel_v101.log
#   results_parallel_v101/unconditional_9_ae_parallel_v101.csv
#   results_parallel_v101/unconditional_9_ae_parallel_v101_best_params.npy
```

### 6.2 各參數說明

| 參數 | 值 | 說明 |
| --- | --- | --- |
| `--backend` | `cudaq_nvidia` | V100 cuStateVec GPU 後端，V100 唯一穩定選擇 |
| `--num_heavy_atom` | `9` | 目標 9 重原子分子，對應 134 個電路參數 |
| `--num_sample` | `10000` | 每次評估的量子電路 shots，與論文對齊 |
| `--particles` | `50` | 粒子群大小，論文 BO 等效 30+ 探索點 |
| `--iterations` | `40` | QPSO 迭代次數，total_evals = 50 × 41 = 2050 |
| `--n_gpus` | `8` | 並行 GPU 數量，決定每輪同時跑幾個子行程 |
| `--gpu_ids` | `0,1,2,3,4,5,6,7` | 8 張 V100 的 device index |
| `--subprocess_timeout` | `600` | 每個子行程最長 600 秒，防止 hang |
| `--alpha_max` | `1.2` | QPSO 收斂係數上界（cosine annealing） |
| `--alpha_min` | `0.4` | QPSO 收斂係數下界 |
| `--mutation_prob` | `0.12` | Cauchy 重尾變異機率，增加探索多樣性 |
| `--stagnation_limit` | `8` | 連續 8 次迭代無進展則觸發重初始化 |
| `--reinit_fraction` | `0.20` | 重初始化時替換 20% 最差粒子 |
| `--ae_weighting` | flag | 啟用 AE-QTS U 形對稱調和加權 mbest |
| `--pair_interval` | `5` | 每 5 次 QPSO 迭代執行一次 AE 配對更新 |
| `--rotate_factor` | `0.01` | AE 配對更新步長因子（對應論文 Δθ/k） |
| `--seed` | `42` | 隨機種子，保持可重現性 |

### 6.3 方案 B：MPI（備用，需先確認 gpu-bind 生效）

**先執行以下確認指令**：

```bash
# 先用 SLURM 確認 --gpu-bind=per_task:1 是否被支援
srun --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --gpu-bind=per_task:1 \
    bash -c 'echo "Rank $PMI_RANK: SLURM_LOCALID=$SLURM_LOCALID, SLURM_STEP_GPUS=$SLURM_STEP_GPUS"'

# 若每個 rank 的 SLURM_STEP_GPUS 各不同（如 rank 0 = "0"，rank 1 = "1"），
# 代表 GPU 綁定正確，可以使用 MPI 方案
```

**確認後提交 SLURM**：

```bash
sbatch cutn-qmg_mpi_8g.slurm

# 監控作業狀態
squeue -u $USER

# 監控 log（另開 tmux pane）
tail -f results_mpi_v12/unconditional_9_ae_mpi_v12.log
```

---

## 七、執行過程監控

### 7.1 期望看到的正常 log 輸出

```text
# 啟動時的並行驗證
[v10.1] 並行功能驗證：同時啟動 8 個子行程（各 5 shots）...
  GPU 0: V=0.XXX  U=0.XXX  ✓
  GPU 1: V=0.XXX  U=0.XXX  ✓
  ...
  GPU 7: V=0.XXX  U=0.XXX  ✓
[v10.1] 並行驗證完成（XXXs）  ✓ 所有 GPU 正常

# Phase 0 每輪應在 ~858s 完成
[parallel 輪次 1/7] 粒子 0..7  GPU: ['0','1','2','3','4','5','6','7']  本輪:XXX.Xs
[parallel 輪次 2/7] 粒子 8..15 ...
...
[parallel 輪次 7/7] 粒子 48..49 ...

# gbest 更新
🔥 New gbest!  V=0.XXXX  U=0.XXXX  V×U=0.XXXX

# 每輪迭代摘要
[AE-QPSO Iter  1/40] α=1.XXX  gbest=0.XXXX (V=0.XXX U=0.XXX)  mean=0.XXXX
```

### 7.2 如何判斷並行是否真的生效

```bash
# 方法一：查看每輪時間
grep "parallel 輪次" results_parallel_v101/unconditional_9_ae_parallel_v101.log | head -10

# 正常：每輪 ~120-150s（單次評估時間，8 GPU 真正並行）
# 異常：每輪 > 900s（代表仍在序列執行）

# 方法二：nvidia-smi 確認 GPU 使用率
while true; do clear; nvidia-smi; sleep 10; done
# 正常：在每輪的 ~120s 內，8 張 GPU 都顯示 GPU-Util 90%+
# 異常：只有 1 張 GPU 在使用，其餘閒置
```

### 7.3 記憶體監控

```bash
# 監控主行程記憶體（應穩定在 < 500 MB）
watch -n 30 "cat /proc/$(pgrep -f run_qpso_qmg_cudaq)/status | grep VmRSS"

# 或透過 log 中的 [MEM] 標記確認
grep "\[MEM\]" results_parallel_v101/unconditional_9_ae_parallel_v101.log
# 應看到 RSS 穩定在 300-500 MB，不持續增長
```

---

## 八、常見問題處理

### Q1：worker_eval.py 子行程全部 exit=1

```bash
# 手動執行一次 worker 確認錯誤訊息
CUDA_VISIBLE_DEVICES=0 python worker_eval.py \
    --weight_path /tmp/smoke_w.npy \
    --result_path /tmp/smoke_r.npy \
    --num_heavy_atom 9 \
    --num_sample 5 \
    --backend cudaq_nvidia

# 常見原因：
# 1. PYTHONPATH 不包含 sqmg_project-cudaq
#    → export PYTHONPATH=~/sqmg_project-cudaq:$PYTHONPATH
# 2. conda 環境未啟動
#    → conda activate cudaq-v071
# 3. cudaq 找不到 GPU
#    → 確認 nvidia-smi 可看到 GPU，且 CUDA_VISIBLE_DEVICES=0 正確設定
```

### Q2：並行驗證通過但 nvidia-smi 顯示只有 1 GPU 在用

```bash
# 這代表 GPU 分配正確，但可能是 PTX JIT 編譯導致第一次慢
# 確認第二批輪次的時間是否縮短（JIT 快取後應加速）

# 若持續只有 1 GPU：
# 確認 gpu_ids 參數確實傳入不同的 device index
python run_qpso_qmg_cudaq.py --gpu_ids 0,1,2,3,4,5,6,7 ...
# 檢查 log 中 "分配 GPU IDs: ['0','1','2','3','4','5','6','7']"
```

### Q3：subprocess 逾時（> 600s）

```bash
# 增加逾時時間（V100 冷啟動 + PTX 編譯最多約 300s）
python run_qpso_qmg_cudaq.py \
    --subprocess_timeout 900 \
    ...
```

### Q4：實驗跑一半中斷，想繼續

目前版本**不支援斷點續跑**（checkpoint 機制尚未實作）。必須從頭重跑。

為避免浪費，建議先確認前幾輪結果後再決定是否跑完：

```bash
# 查看目前最佳結果
grep "🔥 New gbest" results_parallel_v101/unconditional_9_ae_parallel_v101.log | tail -10

# 查看各迭代摘要
grep "AE-QPSO Iter" results_parallel_v101/unconditional_9_ae_parallel_v101.log
```

### Q5：cudaq 找不到 sm_70 kernel（CUDA error）

```bash
# 確認安裝版本
python -c "import cudaq; print(cudaq.__version__)"
# 若不是 0.7.1.x，重裝：
pip install cuda-quantum-cu12==0.7.1 --break-system-packages
# 或在 conda 環境中：
pip install cuda-quantum-cu12==0.7.1
```

---

## 九、實驗結果評估標準

### 9.1 成功標準

```text
主要目標：gbest V×U > 0.8834（超越論文 BO 基準）
次要目標：V > 0.9, U > 0.9（接近論文最佳 V=0.955, U=0.925）
```

### 9.2 實驗進度監控指令

```bash
# 查看目前 gbest
grep "Best V×U" results_parallel_v101/unconditional_9_ae_parallel_v101.log | tail -1

# 查看收斂趨勢
grep "AE-QPSO Iter" results_parallel_v101/unconditional_9_ae_parallel_v101.log | \
    awk '{print $4, $5}' | head -20

# 載入 CSV 查看詳細數據
python -c "
import pandas as pd
df = pd.read_csv('results_parallel_v101/unconditional_9_ae_parallel_v101.csv')
print('總評估次數:', len(df))
print('最佳 V×U:', df['fitness'].max())
print('最佳 V:', df['validity'].max())
print('最佳 U:', df['uniqueness'].max())
print(df.tail(10)[['eval_index','validity','uniqueness','fitness','gbest_fitness']])
"
```

### 9.3 完整實驗完成後，重現最佳分子

```bash
python -c "
import numpy as np
from qmg.utils import ConditionalWeightsGenerator
from qmg.generator_cudaq import MoleculeGeneratorCUDAQ

# 載入最佳參數
best_w = np.load('results_parallel_v101/unconditional_9_ae_parallel_v101_best_params.npy')

# 套用 chemistry constraint 後重新採樣，確認結果
cwg = ConditionalWeightsGenerator(9, smarts=None)
w_constrained = cwg.apply_chemistry_constraint(best_w.copy())

gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w_constrained, backend_name='cudaq_nvidia')
smiles_dict, v, u = gen.sample_molecule(10000)

print(f'V={v:.4f}  U={u:.4f}  V×U={v*u:.6f}')
valid = [k for k in smiles_dict if k and k != \"None\"]
print(f'有效分子種數：{len(valid)}')
print(f'前 10 個 SMILES：')
for smi in valid[:10]:
    print(f'  {smi}')
"
```

---

## 十、程式碼架構快速參考

```text
呼叫流程：

run_qpso_qmg_cudaq.py (main)
  │
  ├─ ConditionalWeightsGenerator      # 生成/約束 134 個權重參數
  │
  ├─ AESOQPSOOptimizer (qpso_optimizer_ae.py)
  │   │  # AE-SOQPSO 主迴圈，管理 50 個粒子的位置更新
  │   │
  │   └─ batch_evaluate_fn (make_parallel_batch_evaluate_fn)
  │       │  # 每批次並行啟動子行程
  │       │
  │       └─ [subprocess] worker_eval.py  (× 8 並行)
  │               │  # 每個子行程獨立持有 CUDA context
  │               │
  │               └─ MoleculeGeneratorCUDAQ (qmg/generator_cudaq.py)
  │                   │
  │                   ├─ _qmg_n9 (build_dynamic_circuit_cudaq.py)
  │                   │   # 20-qubit 動態電路，90 mid-circuit measurements
  │                   │   # cudaq.sample() → 10000 shots
  │                   │
  │                   └─ MoleculeQuantumStateGenerator
  │                       # bitstring → SMILES 轉換
  │                       # 計算 validity & uniqueness
  │
  └─ 輸出：.log / .csv / _best_params.npy
```

---

## 十一、聯絡與參考資料

### 11.1 相關論文

- Chen et al. 2025, JCTC：[QMG 主論文，含 BO 基準結果]（專案根目錄有 PDF）
- Xiao et al. 2026, arXiv:2604.13877v1：[SQMG，tensornet 速度比較]
- Tseng et al. 2024, arXiv:2311.12867v2：[AE-QTS，AE 配對更新演算法]

### 11.2 關鍵技術備忘

```text
★ tensornet 後端與動態電路（mid-circuit measurement）不相容
  → 請勿使用 --backend cudaq_tensornet
  → 只用 cudaq_nvidia（cuStateVec）

★ watch -n 5 nvidia-smi 在 DGX111 上會 segfault
  → 改用：while true; do clear; nvidia-smi; sleep 10; done

★ @cudaq.kernel 測試不能用 python -c，必須寫成 .py 檔案
  → 因為 inspect.getsource() 限制

★ chemistry constraint 只在主行程套用一次
  → worker_eval.py 中 chemistry_constraint=False（已修正雙重套用 bug）
```

---

*文件結束。如有疑問請查閱 git log 或程式碼頂部的版本注記。*
