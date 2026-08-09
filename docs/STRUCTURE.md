# 專案結構速查（v12.0）

重構的目標是：**日後只要記得一個指令**。

```
run_experiment.py  --optimizer <演算法>  --objective <目標>  [--dispatch local|slurm]
```

---

## 一、核心檔案（會用到的）

| 檔案 | 角色 |
|---|---|
| `run_experiment.py` | **統一入口**。所有演算法、所有目標函數、單/多節點都走這支。 |
| `optimizers/` | 八個最適化器，共用介面與預算控制。 |
| `evaluator.py` | 目標函數評估器（參數向量 → V, U），含單節點與多節點兩種派工。 |
| `worker_eval.py` | 最底層：一個子行程、一顆 GPU、一次 `cudaq.sample()`。 |
| `node_agent.py` | 多節點模式下每個節點的代理，認領自己那段粒子。 |
| `benchmark/` | 比較實驗的提交與分析工具。 |
| `run_iqm_qpu.py` | IQM Resonance 真機／可行性分析。 |

## 二、八個最適化器

| `--optimizer` | 演算法 | 類型 | 平行度 |
|---|---|---|---|
| `sobol` | Sobol random search | 低差異隨機搜尋 | M |
| `spsa` | SPSA | 隨機逼近 | M |
| `de` | Differential Evolution | 演化 | M |
| `cmaes` | CMA-ES | 演化策略 | M |
| `bo` | Bayesian Optimization | GP + EI，**序列** | **1** |
| `batch_bo` | Batch BO | GP + q-EI | M |
| `qpso` | QPSO | 量子行為粒子群 | M |
| `rr_qpso` | **RR-QPSO（本方法）** | QPSO + Sobol/rank-refined/OBL | M |

`bo` 平行度是 1 —— 這不是缺陷，是它的本質，也是這次比較要量的東西之一。

## 三、兩種目標函數

| `--objective` | fitness |
|---|---|
| `vu`（預設） | `V × U` |
| `hbahbd` | `(V×U) × ((1-w) + w·exp(-½((|HBA-4|/σ)² + (|HBD-3|/σ)²)))` |

同一組演算法、同一份程式碼，只換這個旗標。

## 四、常用指令

```bash
# 單節點 8 GPU
python run_experiment.py --optimizer rr_qpso --M 32 --T 16

# 多節點（必須在 sbatch 配額內）
python run_experiment.py --optimizer cmaes --M 32 --T 16 --dispatch slurm --nodes 4

# HBA/HBD 目標
python run_experiment.py --optimizer rr_qpso --objective hbahbd

# 完整比較實驗：8 演算法 × 5 seeds
python benchmark/launch_benchmark.py --dry_run      # 先看排程計畫
python benchmark/launch_benchmark.py --M 32 --T 16 --seeds 5

# 出圖與統計表
python benchmark/analyze_benchmark.py --data_dir results_benchmark
```

## 五、公平比較是怎麼保證的

1. **預算**：`--M × --T` 就是總評估次數，由 `optimizers/base.py` 的
   `_evaluate_metrics()` 強制執行——超出就截斷並中止。因此序列的 BO 與
   族群式的 CMA-ES 吃到的目標函數次數完全相同。
2. **紀錄**：所有演算法寫出欄位一致的 CSV，分析程式只有一支。
3. **變異**：每個 (演算法, seed) 各一個 run，預設 5 個 seed，
   圖上以中位數 + min–max 帶呈現，避免用單次結果下結論。
4. **刻意不統一**：初始化策略保持各演算法原生做法（Sobol 初始化本身
   就是 RR-QPSO 的貢獻之一，抽掉就不是在比同一件事）。

## 六、`legacy/`

以下已被取代，移入 `legacy/` 僅供追溯，**不要再用**：

`run_qpso_qmg_cudaq v100.py`、`run_qpso_qmg_cudaq_v94_backup.py`、
`run_qpso_qmg_mpi.py`、`qpso_optimizer_qmg.py`、`cudaq_*_diagnostic.py`、
`cutn-qmg_mpi_8g.slurm`、`gpu_scaling_bench.slurm`、`bench_node.py`、
`run_m128_hbahbd.sh`、`run_nosobol.sh`、`run_sweep.sh`、`run_hbahbd_multiobj.sh`

保留在根目錄的 `run_qpso_qmg_cudaq.py` 與 `run_qpso_qmg_cudaq_hbahbd_multiobj.py`
是已驗證的生產 runner（論文結果由它們產出），暫時保留以確保可重現；
新的實驗一律用 `run_experiment.py`。
