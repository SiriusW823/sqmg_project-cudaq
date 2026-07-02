# HBA/HBD 量測記錄（v10.4）— chemistry_constraint 4HBA/3HBD 對比實驗 handoff

日期：2026-07-02
目的：在**完全不改動既有 QPSO 架構與 V×U 最適化**的前提下，讓 CUDA-Q + AE-SOQPSO
架構在最大化 `product_validity_uniqueness (V×U)` 的同時，額外**量測並記錄**生成分子群
的平均 HBA / HBD，用以對比參考 log `chemistry_constraint_qiskit_4HBA_3HBD_0.log`
（qiskit + 不同優化器）的 HBA≈4、HBD≈3 結果。

> 一句話結論：本次改動屬「量測儀器」，不是「最適化目標」。QPSO 的目標、更新公式、
> 收斂行為與 v10.3 **逐位元相同**；HBA/HBD 只是掛在分子生成之後被讀出並記錄。

---

## 1. 改了什麼（只有兩個檔案，opt-in，向後相容）

### `worker_eval.py`
- 新增 `compute_mean_hba_hbd(smiles_dict)`：以 RDKit `Lipinski.NumHAcceptors` /
  `Lipinski.NumHDonors` 對「有效分子」依出現次數加權求平均。此定義與多數 QMG 論文
  （含 Chen 等）一致；若參考實作用的是 `rdMolDescriptors.CalcNumHBA/CalcNumHBD`，
  只要改函式內 `_HBA_FN` / `_HBD_FN` 兩行即可。
- 新增 `--report_hbahbd` 旗標（預設關）。**未指定時執行流程與舊版完全相同、不做任何
  RDKit 計算、不增加執行時間。**
- 結果檔由 `[V, U]` 兩欄擴為 `[V, U, HBA, HBD]` 四欄。父行程只讀 `arr[0], arr[1]`
  作為 (V, U)，因此對 optimizer 是透明的。

### `run_qpso_qmg_cudaq.py`
- 新增 CLI：`--hba_target`（參考值 4）、`--hbd_target`（參考值 3）。**任一設定即開啟
  量測；兩者皆 None（預設）時完全等同 v10.3。**
- 新增 `HBAHBDRecorder` 類別：掛在平行評估函式之後，逐「批次」還原 qiskit 的
  `Iteration number: N` 逐代編號，並記錄該批次 **V×U 最佳粒子** 的 V×U / HBA / HBD，
  另存逐代 CSV（`{data_dir}/{task_name}_hbahbd.csv`）。
- 開啟量測時，log 標頭改為與參考 log 對齊的
  `Task: ['product_validity_uniqueness', 'HBA', 'HBD']` /
  `Condition: ['None', '4', '3']` / `objective: ['maximize', 'measure', 'measure']`，
  並明確標註「HBA/HBD 為量測欄位，非最適化目標」。

### 沒有動的部分（重點）
- **`qpso_optimizer_ae.py` 一行未改**：mbest、AE-QTS、OBL、VU-decouple、mode-collapse
  全部原封不動，fitness 仍只有 V×U。
- 單 GPU 序列模式、MPI 版、其他所有檔案未改。

---

## 2. 三個環境同一化（DGX111 / 這台電腦 / GitHub）

改動只在 `worker_eval.py` 與 `run_qpso_qmg_cudaq.py`。請在**這台電腦（Windows，本
OneDrive 資料夾）** 執行：

```bash
git add worker_eval.py run_qpso_qmg_cudaq.py docs/hbahbd_measurement_v10.4_handoff.md
git commit -m "v10.4: opt-in HBA/HBD measurement (measure-only; V×U optimization unchanged)"
git push origin main
```

然後在 **DGX111** 拉取：

```bash
cd /beegfs/home/sirius/sqmg_project-cudaq   # 依實際路徑
git fetch origin && git checkout main && git pull origin main
```

> 注意：目前 repo 內尚有其它未提交的既有修改（例如 `qmg/generator_cudaq.py`、
> `qmg/utils/*` 等）。那些是你先前的工作，本次未觸碰；若也要讓三環境一致，請自行決定
> 是否一併 commit/push。三方 `git rev-parse HEAD` 一致即代表版本相同。

---

## 3. 在 DGX111 上執行 M=128（log 準拠參數）

```bash
cd /beegfs/home/sirius/sqmg_project-cudaq
nohup env PYTHONPATH=. python run_qpso_qmg_cudaq.py \
  --particles        128 \
  --iterations       150 \
  --num_heavy_atom   9 \
  --num_sample       10000 \
  --n_gpus           8 \
  --gpu_ids          0,1,2,3,4,5,6,7 \
  --backend          cudaq_nvidia \
  --hba_target       4 \
  --hbd_target       3 \
  --subprocess_timeout 900 \
  --task_name        chemistry_constraint_cudaq_4HBA_3HBD_M128 \
  --data_dir         results_hbahbd \
  > results_hbahbd_M128.console.log 2>&1 &
```

參數對應參考 log：`num_heavy_atom=9`、`num_sample=10000`、HBA 目標 4、HBD 目標 3；
`--particles 128` 為本次指定的 M；`--iterations 150` 為你現行預設 T。

**產出檔案**（皆在 `results_hbahbd/`）：
- `chemistry_constraint_cudaq_4HBA_3HBD_M128.log` — 主 log（含逐代 V×U 與 HBA/HBD 量測，
  即你要的「一份 log 檔案做紀錄」）。
- `chemistry_constraint_cudaq_4HBA_3HBD_M128_hbahbd.csv` — 逐代 HBA/HBD 量測明細。
- `chemistry_constraint_cudaq_4HBA_3HBD_M128.csv` — optimizer 既有的逐粒子 V/U/fitness。
- `chemistry_constraint_cudaq_4HBA_3HBD_M128_best_params.npy` — 最佳參數。

---

## 4. 重要提醒（執行前務必看）

- **執行時間**：M128 + 5000 shots 的既有紀錄約 95.9h。本次 **10000 shots，每次評估時間約
  翻倍**，預估 **~190h（約 8 天）**，與參考 qiskit log 的天數量級相當。log 內 `預估：Xh`
  那行是用 142s/eval 的硬編碼估算，10000 shots 下會低估約一半，屬正常。
- **`--subprocess_timeout 900`**：現行預設 360s 是為 5000 shots（~142s/eval）設計。
  10000 shots 每次約 ~284s，360s 餘量太小可能誤殺子行程，故上調至 900s。
- **後端**：務必 `cudaq_nvidia`（tensornet 與動態電路不相容，會 silent hang）。
- **HBA/HBD 是「量測」不是「目標」**：本架構最適化的是 V×U，權重套用的是 SQMG 結構約束
  （`apply_chemistry_constraint`：sum θ=π 等），並非把 HBA=4/HBD=3 當條件去 conditioning。
  因此本實驗回答的是：「當 V×U 被推高時，生成分子的 HBA/HBD 實際落在哪」。這正是你要驗證
  的命題；HBA/HBD 是否自然接近 4/3 由實測決定，不由程式碼保證。

---

## 5. 驗證狀態（在本機 sandbox 完成的檢查）

- `worker_eval.py`：Python AST 完整解析通過。
- `run_qpso_qmg_cudaq.py`：`HBAHBDRecorder` 解析 + 執行 smoke test 通過（逐代編號
  phase0→0、OBL→略過、iter→1…；best 粒子挑選與 CSV 輸出正確）；
  `make_parallel_batch_evaluate_fn` 與 main 編輯片段解析通過。
- 後向相容：未給 `--hba_target/--hbd_target` 時，傳給 optimizer 的仍是 `(V, U)`，
  行為與 v10.3 相同。
- 未在 sandbox 執行的部分：RDKit 未安裝於 sandbox，`compute_mean_hba_hbd` 的 RDKit 數值
  未實跑（邏輯為標準加權平均，DGX 上 RDKit 為既有依賴，會正常運作）；GPU/CUDA-Q 實跑需在
  DGX111 進行。
