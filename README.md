# sqmg_project-qiskit

**QMG Dynamic Circuit + SOQPSO Optimizer**
超越 Chen et al. 2025 Bayesian Optimization 基線的量子分子生成實驗。

---

## 專案定位

| 面向 | 原始 QMG (Chen et al. 2025) | 本專案 |
|------|---------------------------|--------|
| 量子電路 | Qiskit 動態電路 (DynamicCircuitBuilder) | **相同**（直接使用 qmg/ 套件） |
| 優化器 | Bayesian Optimization (GPEI/SAASBO) | **SOQPSO** (Delta 勢阱 + Cauchy 變異) |
| 框架 | Qiskit + Ax/BoTorch | Qiskit + 自訂 QPSO |
| 目標 | validity × uniqueness (maximize) | **相同** |
| 後端 | Qiskit Aer (CPU local) | Qiskit Aer (CPU/GPU local) |

**基線**（`unconditional_9.log`）：
- Best V×U = **0.8834**（V=0.955, U=0.925），第 355 次評估達到
- 需要 ~300 次評估才收斂至穩定 V×U > 0.80 域

**目標**：在相同計算預算（~600 次電路評估）內，
更快收斂至 V×U ≥ 0.88，並維持穩定的高 V×U。

---

## 資料夾結構

```
sqmg_project-qiskit/
├── qmg/                          # 來自 PEESEgroup/QMG，分子生成模型（原版不修改）
│   ├── __init__.py
│   ├── generator.py              # MoleculeGenerator 主類別
│   └── utils/
│       ├── __init__.py
│       ├── build_dynamic_circuit.py  # N=9 動態電路（20 qubits）
│       ├── build_circuit_functions.py
│       ├── chemistry_data_processing.py
│       ├── fitness_calculator.py
│       └── weight_generator.py   # ConditionalWeightsGenerator（134 維映射）
│
├── qpso_optimizer_qmg.py         # ★ SOQPSO 優化器（由 CUDA-Q 版移植）
├── run_qpso_qmg.py               # ★ 主入口：QMG 電路 + QPSO
├── requirements.txt
└── README.md
```

---

## 環境安裝（DGX111）

```bash
# 建立獨立 conda 環境（與 Project 1 的 CUDA-Q 環境完全分離）
conda create -n qmg_qpso python=3.12 -y
conda activate qmg_qpso
pip install -r requirements.txt
```

驗證安裝：
```bash
python -c "
from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator
import numpy as np
cwg = ConditionalWeightsGenerator(9, smarts=None)
n = int((cwg.parameters_indicator == 0.).sum())
print(f'QMG OK  flexible params for N=9: {n}')   # 應為 134
"
```

---

## 執行實驗

```bash
# 標準實驗（對應 unconditional_9.log 參數，M=5 × T=120 = 605 evals）
python run_qpso_qmg.py \
  --num_heavy_atom  9 \
  --num_sample      10000 \
  --particles       5 \
  --iterations      120 \
  --seed            42 \
  --task_name       unconditional_9_qpso \
  --data_dir        results_qpso

# 快速驗證（M=3 × T=10 = 33 evals，~30 分鐘）
python run_qpso_qmg.py \
  --num_heavy_atom 9 --num_sample 10000 \
  --particles 3 --iterations 10 --seed 42
```

---

## 輸出檔案

| 檔案 | 內容 |
|------|------|
| `results_qpso/unconditional_9_qpso.log` | 完整優化 log（格式與 unconditional_9.log 一致）|
| `results_qpso/unconditional_9_qpso.csv` | 每次評估的詳細指標 |
| `results_qpso/unconditional_9_qpso_best_params.npy` | 最佳參數向量（134 維）|

---

## 關鍵指標比較目標

| 指標 | BO 基線 | QPSO 目標 |
|------|---------|-----------|
| Best V×U | 0.8834 | **> 0.88** |
| 達到 V×U > 0.80 所需評估數 | ~200 次 | **< 100 次** |
| 後期平均 V×U（最後 100 次）| 0.595 | **> 0.75** |

---

## 引用

```bibtex
@article{chen2025qmg,
  title   = {Exploring Chemical Space with Chemistry-Inspired Dynamic Quantum Circuits in the NISQ Era},
  author  = {Chen, ...},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2025},
}
```
