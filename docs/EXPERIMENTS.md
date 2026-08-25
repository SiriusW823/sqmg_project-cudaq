# 實驗總表

> 相關文件：[`RESULTS.md`](RESULTS.md)（完整敘事與解讀）·
> [`CLUSTER.md`](CLUSTER.md)（叢集操作）·
> [`STRUCTURE.md`](STRUCTURE.md)（程式架構）

**更新日期**：2026-08-24　**資料位置**：`SQMG/experiments/`（1,075 檔，60 MB，**不進 git**——體積過大，
留在本機與叢集上；本文件是索引，說明每份資料是什麼、怎麼重現）
**完整性**：全部檔案已以 SHA-256 對照 `manifest.csv` 驗證通過

問題設定一律為：9 heavy-atom QMG，D=134，20 qubits，CUDA-Q 0.7.1，V100，
目標函數 V×U，M=64，1,000 shots，配對設計（seed 同時控制最適化器與 shot 亂數）。

---

## 一、八種優化方法橫向對照

**完整預算（9,664 次評估），M=64，n=5 配對 seed**　資料：`07_eight_algorithms_M64/`

| 排名 | 方法 | 中位數 V×U | 標準差 | 平均秩 |
|---|---|---|---|---|
| 1 | **QPSO** | **0.9750** | 0.0096 | 1.40 |
| 2 | **CMA-ES** | **0.9710** | 0.0055 | 2.00 |
| 3 | RR-QPSO（本方法） | 0.9640 | 0.0169 | 2.80 |
| 4 | Differential Evolution | 0.9160 | 0.0187 | 3.80 |
| 5 | SPSA | 0.7190 | 0.0143 | 5.00 |
| 6 | Batch BO | 0.6950 | 0.0168 | 6.00 |
| 7 | Sobol random search | 0.5700 | 0.0318 | 7.00 |
| — | Bayesian Optimization | 0.6925† | 0.0139 | — |

**Friedman χ²=28.114，p=0.00009** → 演算法之間確實有差異。

† BO 在這批的預算上限設為 2,000（序列式跑滿 9,664 不可行），故完整預算的數字
取自 `05_bo_comparison/b_capped_9664`（n=10）。同批 2,000 次預算下 BO 為 0.6700。

**與 RR-QPSO 的成對對照**（完整預算）：

| 對手 | Δ (RR − 對手) | RR 勝場 | Cliff's δ | |
|---|---|---|---|---|
| QPSO | −0.0160 | 1/5 | −0.68 | RR 較差 |
| CMA-ES | −0.0010 | 1/5 | −0.36 | RR 較差 |
| Differential Evolution | +0.0620 | 4/5 | +0.92 | RR 較佳 |
| SPSA | +0.2430 | 5/5 | +1.00 | RR 較佳 |
| Batch BO | +0.2640 | 5/5 | +1.00 | RR 較佳 |
| Sobol | +0.4110 | 5/5 | +1.00 | RR 較佳 |

**兩個值得注意的結果**：

1. **CMA-ES 與 RR-QPSO 實質同等**（Δ=−0.0010），而 CMA-ES 是完全標準、
   不含任何量子啟發成分的演化策略，且標準差只有 RR-QPSO 的三分之一（0.0055 vs 0.0169）。
2. **Sobol random search 墊底**（0.5700）——這是必要的健全性檢查：
   若隨機搜尋不是最差，代表比較框架有問題。

**檢定力限制**：n=5 的配對 Wilcoxon 最小可能 p 值為 0.0625，經 7 個比較的 Holm
校正後為 0.4375 ——**成對比較結構上不可能顯著**。整體的 Friedman 檢定不受此限，
故「演算法間有差異」成立，但「哪兩個之間有差異」在此批資料無法斷言。
QPSO / RR-QPSO / BO 三者已另以 n=10 重測（實驗二、六）。

**各預算下的中位數**：

| 方法 | 1,000 | 2,000 | 4,000 | 6,000 | 8,000 | 9,664 |
|---|---|---|---|---|---|---|
| QPSO | 0.7300 | 0.7950 | 0.8850 | 0.9420 | 0.9680 | 0.9750 |
| CMA-ES | 0.6670 | 0.7710 | 0.8790 | 0.9250 | 0.9580 | 0.9710 |
| RR-QPSO | 0.6610 | 0.7550 | 0.8440 | 0.9160 | 0.9510 | 0.9640 |
| DE | 0.5860 | 0.6910 | 0.8100 | 0.8630 | 0.9030 | 0.9160 |
| SPSA | 0.4480 | 0.5260 | 0.6110 | 0.6530 | 0.7040 | 0.7190 |
| Batch BO | 0.6080 | 0.6520 | 0.6520 | 0.6570 | 0.6870 | 0.6950 |
| Sobol | 0.4730 | 0.5030 | 0.5030 | 0.5050 | 0.5570 | 0.5700 |

Batch BO 與 Sobol 在 2,000 次後幾乎停滯；族群式方法持續改善。

---

### 1b. 小規模對照（M=32、512 次評估）

`06_algorithm_benchmark/`　同樣八個方法，5 個 seed，屬前導批次。

| 方法 | 中位數 | | 方法 | 中位數 |
|---|---|---|---|---|
| QPSO | 0.7520 | | Differential Evolution | 0.5950 |
| RR-QPSO | 0.7360 | | Batch BO | 0.5850 |
| CMA-ES | 0.6500 | | SPSA | 0.4910 |
| Bayesian Optimization | 0.6130 | | Sobol | 0.4730 |

Friedman p=0.00006。排序與完整預算大致一致，唯 CMA-ES 在小預算下較弱
（0.6500，排第三）而在完整預算下升到第二——顯示 CMA-ES 需要較多評估才進入狀況。

---

### 1c. BO 專門比較（n=10）

`05_bo_comparison/`　同預算 1,984 次

| 方法 | 中位數 | 標準差 |
|---|---|---|
| QPSO | 0.8895 | 0.0355 |
| RR-QPSO | 0.8770 | 0.0150 |
| BO（無 GP 上限） | 0.6765 | 0.0206 |
| BO（GP 上限 400） | 0.6650 | 0.0240 |
| Batch BO | 0.6495 | 0.0313 |

> ⚠ **這一節的 BO 數字目前不可信。** 見第八節「已知的效度威脅」。
> 論文原始的 BO 是 Ax/BoTorch GPEI，與此處手寫的實作有系統性差異。

---

## 二、實驗一覽

| # | 實驗 | 資料夾 | 組態 × seeds | 總 runs | 結論 |
|---|---|---|---|---|---|
| 0 | **八演算法比較（正式）** | `07_eight_algorithms_M64` | 8 × 5 | 40 | QPSO > CMA-ES > RR-QPSO > DE ≫ BO 類 > Sobol |
| 1 | α 排程對齊 | `01_alpha_alignment` | 4 × 10 | 40 | α 是混淆變因 |
| 2 | 長時程（T=150） | `02_longhorizon` | 2 × 10 | 20 | RR ≡ QPSO（TOST p=0.0011） |
| 3 | 組件消融（前導） | `03_ablation_pilot` | 7 × 10 | 70 | Friedman p=0.297，無差異 |
| 4 | 消融確認（預登記） | `04_ablation_confirmatory` | 3 × 30 | 90 | 兩個假說皆 null |
| 5 | shot 雜訊偏誤 | （見 `00_summary`） | 5 組 × 24 重複 | — | 膨脹 +0.011~+0.027 |
| 6 | BO 比較 | `05_bo_comparison` | 4×10 + 10 + 10 | 60 | 待重測 |
| — | 八演算法前導批 | `06_algorithm_benchmark` | 8 × 5 | 40 | M=32、512 次，排序與正式批一致 |

**正式 run 共 320 個**（實驗零~六），另有 40 個前導 run（`06_algorithm_benchmark`）。

---

## 三、實驗一：α 排程對齊

`01_alpha_alignment/`　4 組態 × 10 seeds，2,000 次評估

| 組態 | 說明 | 檔名前綴 |
|---|---|---|
| `qpso_a_def` | QPSO，α=[0.5, 1.0]（文獻預設） | `qpso_a_def_s*` |
| `qpso_a_rr` | QPSO，α=[0.3, 1.2] | `qpso_a_rr_s*` |
| `rr_a_def` | RR-QPSO，α=[0.5, 1.0] | `rr_a_def_s*` |
| `rr_a_rr` | RR-QPSO，α=[0.3, 1.2]（原設定） | `rr_a_rr_s*` |

**發現**：RR-QPSO 原本用 α=[0.3,1.2]、QPSO 基準用 [0.5,1.0]，兩者的差異因此混合了
「RR 機制的效果」與「α 排程的效果」。α 對齊後，**QPSO 在每個預算都領先**
（6–9/10 勝場，1,500 次評估處 p=0.0273）。

---

## 四、實驗二：長時程（論文的操作點）

`02_longhorizon/`　2 組態 × 10 seeds，**9,664 次評估**（T=150）

| 組態 | 檔名前綴 |
|---|---|
| QPSO 基準 | `lh_qpso_s*` |
| RR-QPSO 完整 | `lh_full_s*` |

| | 中位數 | 平均 | 標準差 |
|---|---|---|---|
| QPSO | 0.9720 | 0.9689 | 0.0187 |
| RR-QPSO | 0.9655 | 0.9630 | 0.0193 |

- Δ = **−0.0080**，95% CI [−0.0160, +0.0040]，勝場 3/10，p=0.1211
- **TOST（±0.02）p=0.0011 → 統計上等價**（主動證實，非「沒測出差異」）
- 穩定性：σ 比值 1.03，Fligner p=0.589 → 無差異
- **4,000 與 5,000 次評估處 RR 顯著較差**（Holm p=0.018，Cliff's δ≈−0.77，0/10）

圖：`00_summary/fig_m1_longhorizon.png`

---

## 五、實驗三：組件消融（前導）

`03_ablation_pilot/`　7 組態 × 10 seeds，1,984 次評估

| 組態 | 關閉的組件 | 檔名前綴 |
|---|---|---|
| `ab_qpso` | （原版 QPSO 基準） | `ab_qpso_s*` |
| `ab_full` | 無（完整 RR-QPSO） | `ab_full_s*` |
| `ab_nosobol` | Sobol 初始化 | `ab_nosobol_s*` |
| `ab_noobl` | OBL 對立式學習 | `ab_noobl_s*` |
| `ab_noae` | AE-QTS 有符號 mbest | `ab_noae_s*` |
| `ab_novu` | V-U 解耦 | `ab_novu_s*` |
| `ab_nomc` | mode-collapse 回收 | `ab_nomc_s*` |

| 組件 | Δ 中位數 | 勝場 | 95% CI | dz |
|---|---|---|---|---|
| OBL | +0.0250 | 7/10 | [−0.0250, +0.0485] | 0.513 |
| V-U 解耦 | +0.0170 | 7/10 | [−0.0070, +0.0335] | 0.191 |
| AE mbest | +0.0130 | 8/10 | [−0.0055, +0.0270] | 0.538 |
| Sobol | +0.0100 | 7/10 | [−0.0110, +0.0250] | 0.344 |
| mode-collapse | +0.0060 | 6/10 | [−0.0140, +0.0270] | 0.176 |

**Friedman p=0.2969**；五個 Δ 全為正但每條 CI 都跨零；聚合排列檢定 p=0.0814。
→ 檢定力不足，非效果為零。

> **本實驗中發現的實作缺陷**：Sobol 初始化不在 `AESOQPSOOptimizer` 內，而是舊 runner
> 以 `make_sobol_positions()` 覆寫 `positions`/`pbest` 實現的。v12 wrapper 漏了這步，
> 因此**在此之前的所有 RR-QPSO run 都缺少 Sobol 初始化**。本實驗已修正。

圖：`00_summary/fig_m2_ablation_forest.png`

---

## 六、實驗四：消融確認（預登記）

`04_ablation_confirmatory/`　3 組態 × **30 個全新 seed（10–39）**，1,984 次評估

預登記於 git commit `0e36407`，早於任何確認性資料。
計畫全文：[`PREREGISTRATION_ablation_confirmatory.md`](PREREGISTRATION_ablation_confirmatory.md)

| 假說 | Δ 中位數 | 勝場 | p | **p_holm** | 判定 |
|---|---|---|---|---|---|
| H1 OBL 有貢獻 | +0.0025 | **15/30** | 0.131 | **0.131** | ✗ |
| H2 AE mbest 有貢獻 | +0.0110 | 18/30 | 0.038 | **0.075** | ✗ |

**效果量收縮**：

| 組件 | 前導 dz | 確認 dz | 收縮 | 80% 檢定力需 n |
|---|---|---|---|---|
| OBL | 0.513 | 0.342 | ×0.67 | 68 |
| AE mbest | 0.538 | 0.297 | ×0.55 | 89 |

換全新 seed 是關鍵：這兩個組件是從前導資料挑效果最大的兩個，沿用舊 seed 等於
用產生假說的資料驗證假說。若當初直接補到 n=30，AE mbest 幾乎必然「顯著」——
而那會是假陽性。

圖：`00_summary/fig_m3_shrinkage.png`

> **資料註記**：`cf_*` 有 22 個 run 長度為 2,000 而非 1,984。已驗證非 resume 汙染
> （best-so-far 單調、iteration 無回跳、無重複列），成因是觸發 mode-collapse
> 回收的 run 會多消耗評估直到 2,000 的硬上限。分析一律取 index 1,984。

---

## 七、實驗六：BO 比較

`05_bo_comparison/`　三個子資料夾

| 子資料夾 | 內容 | 檔名前綴 | runs |
|---|---|---|---|
| `a_capped_2000` | BO / Batch BO / QPSO / RR-QPSO，2,000 次 | `bo_{algo}_s*` | 40 |
| `b_capped_9664` | BO，9,664 次 | `bof_bo_s*` | 10 |
| `c_uncapped_2000` | BO，`max_gp_points=2000`（不設限） | `bou_bo_s*` | 10 |

**2,000 次預算**：QPSO 0.8895、RR-QPSO 0.8770、BO 0.6650、Batch BO 0.6495
Friedman p=0.00002；QPSO/RR 對 BO 皆 10/10、p_holm=0.0059、Cliff's δ=+1.000

**9,664 次預算**：QPSO 0.9720、RR-QPSO 0.9655、BO 0.6925
差距隨預算擴大：500 次時 BO 領先（Δ=−0.018），1,000 次後反轉，5,000 次起 δ=+1.00

**GP 上限對照**（唯一變因）：capped 0.6650 vs uncapped 0.6765，Δ=+0.0125，
5/10，p=0.156。健全性檢查通過（400 次時兩者完全相同）。
→ 上限至多解釋差距的 22%，不是主因。

圖：`00_summary/fig_m5_bo_comparison.png`

---

## 八、已知的效度威脅

**1. BO baseline 可能過弱（未解決，最嚴重）**

論文原始的 BO 是 [PEESEgroup/QMG](https://github.com/PEESEgroup/QMG) 的
Ax/BoTorch `Models.GPEI`，與本專案手寫的 `optimizers/bayesopt.py` 有系統性差異：

| | 論文（Ax/BoTorch） | 本專案 |
|---|---|---|
| kernel | BoTorch 預設 **ARD** Matérn 5/2 | **各向同性** Matérn |
| GP 訓練點 | 全部 | 上限 400 |
| Sobol 初始 | 5 | 128 |
| 加速 | GPU | CPU |

在 D=134 下，各向同性 kernel 等於假設 134 個參數敏感度相同——這幾乎必然是
本專案 BO 只到 0.68、而論文報告 0.902 的主因。
**因此第一節與第七節的 BO 數字必須視為暫定，待以 Ax/BoTorch 重測。**

**2. shots 不可跨組比較**

上游 `constrained_bo.py` 的 `num_sample` 預設為 **10,000**，本研究用 1,000。
uniqueness 的定義是 `相異分子數 / 有效分子數`——shots 增加時分母近似線性成長、
分子飽和，故 **shots 越多 uniqueness 越低**。本研究的 V×U≈0.97 高於論文的 0.930，
主要來自 shots 較少，**不代表方法較好**。

**3. 天花板效應**

1,000 shots 下雙方都逼近可達上限（0.97），可能壓縮了 RR 與 QPSO 的差異。
5,000 shots 重跑約需 7 天，尚未進行。

**4. 單一問題實例**

全部結論限於 9 heavy-atom 的無條件 V×U。HBA/HBD 多目標的 landscape 更崎嶇，
探索機制可能在該處才有價值——尚未檢驗。

---

## 九、資料夾結構

```
SQMG/experiments/
├── EXPERIMENTS.md              ← 本檔
├── manifest.csv                ← 953 檔的 SHA-256 與行數
├── run_counts.json             ← 各組態的 run 數與評估數
├── 00_summary/                 ← 統計 JSON + 五張圖（PNG/PDF）
├── 01_alpha_alignment/         ← 實驗一
├── 02_longhorizon/             ← 實驗二
├── 03_ablation_pilot/          ← 實驗三
├── 04_ablation_confirmatory/   ← 實驗四
├── 05_bo_comparison/           ← 實驗六
│   ├── a_capped_2000/
│   ├── b_capped_9664/
│   └── c_uncapped_2000/
├── 06_algorithm_benchmark/     ← 八演算法前導批（M=32、512 次）
└── 07_eight_algorithms_M64/    ← 八演算法正式批（M=64、9,664 次）
```

每個 run 有三個檔案：`{task}.csv`（逐次評估紀錄）、
`{task}_best_params.npy`（最佳參數向量）、`{task}_summary.json`（摘要）。

**log 檔未下載**（535 MB），留在 DGX 的 `~/sqmg_project-cudaq/results_*/`。
需要時用 `python ../SQMG/tools/dgx_fetch.py "sqmg_project-cudaq/results_XXX/*.log" <本機目錄>` 取回。

---

## 十、重現與驗證

```bash
# 驗證本機資料完整性（比對 manifest 的 SHA-256）
python ../SQMG/tools/verify_experiments.py

# 八演算法比較（本機即可跑）
python ../SQMG/tools/analyze_8algo.py       # 前導批 M=32
# 正式批 M=64 需在 DGX 上跑：
bash ../SQMG/scripts/analyze_prod.sh        # 八演算法 @ 2,000（BO 受限）
bash ../SQMG/scripts/analyze_prod_full.sh   # 七演算法 @ 9,664

# 檢查確認性實驗的資料異常
python ../SQMG/tools/check_cf_anomaly.py

# 在 DGX 上重跑各實驗的分析
bash ../SQMG/scripts/lh_final.sh            # 實驗二
bash ../SQMG/scripts/ablation_analyze.sh    # 實驗三
bash ../SQMG/scripts/ablation_aggregate.sh  # 實驗三（聚合）
bash ../SQMG/scripts/confirm_analyze.sh     # 實驗四（預登記分析）
bash ../SQMG/scripts/bo_analyze.sh          # 實驗六
bash ../SQMG/scripts/bofull_analyze.sh      # 實驗六（完整預算）
bash ../SQMG/scripts/bouncap_analyze.sh     # 實驗六（GP 上限對照）
bash ../SQMG/scripts/make_figures.sh        # 全部圖表 + stats.json
```

完整的敘事與解讀見 `docs/RESULTS.md`。

**叢集操作註記**：DGX101 曾 drained；DGX102 實測慢約 14 倍（約 100 s/eval
vs 健康節點 7 s/eval），提交時應 `--exclude=DGX101,DGX102`；
每節點 2 個 worker 為最佳，8 個會靜默丟失評估。
