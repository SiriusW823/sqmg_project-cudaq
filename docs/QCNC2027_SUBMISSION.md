# QCNC 2027 投稿準備

**建立**：2026-10-12　**截稿**：**2026-10-20**（剩 8 天）　**狀態**：方向已定——A 方法學再評估稿，並同時重跑公平比較（§4、§5）

> 相關文件：[`PAPER_ERRATA.md`](PAPER_ERRATA.md)（預印本勘誤）·
> [`RESULTS.md`](RESULTS.md)（十一個實驗的完整結果）·
> [`FAIRNESS_AUDIT.md`](FAIRNESS_AUDIT.md)（RR-QPSO 自身的公平性審查）·
> [`EXPERIMENTS.md`](EXPERIMENTS.md)（資料索引）·
> 預印本：[`../paper/`](../paper/)

---

## 1. 會議規範（2026-10-12 讀自官網）

| 項目 | 內容 |
|---|---|
| 會議 | IEEE QCNC 2027 — International Conference on Quantum Communications, Networking, and Computing |
| 日期／地點 | 2027-03-22 ～ 03-26，日本北海道 |
| **論文截稿** | **2026-10-20**（官網未註明時區，建議台灣時間 10/19 前上傳） |
| 錄取通知 | 2027-01-20 |
| Camera-ready | 2027-02-20 |
| 頁數 | 審稿版 **≤ 8 頁**；camera-ready 可到 10 頁，超頁另收費 |
| 格式 | IEEE Computer Society Proceedings Format：雙欄、10pt、Letter，PDF |
| 審稿 | **Double-blind**——不得出現作者姓名、單位，避免明顯的自我引用 |
| 原創性 | 不得已發表，也不得同時在其他會議／期刊審稿中 |
| Track | 每篇只能投一個 track；不合規定者不審即退 |
| 投稿系統 | EasyChair：<https://easychair.org/conferences/?conf=qcnc2027> |
| 出版 | 錄取並發表者收錄於 IEEE Xplore |

來源：<https://www.qcnc.org/2027/cf-papers.php>、<https://www.qcnc.org/2027/submission.php>

官網**沒有**提到：arXiv 預印本政策、註冊／no-show 規定、PDF eXpress、摘要預先登錄。
IEEE 一般允許作者在 arXiv 張貼預印本，但 double-blind 下送審稿件本身必須匿名。

### Track 選擇

| Track | 相關主題（官網） | 適合的稿件 |
|---|---|---|
| **2 — Quantum Computing and Sensing** | algorithms and protocols for computing／optimization、NISQ devices、hybrid quantum-classical systems | 以最適化器為主體的稿件（不論正面或方法學結果）**← 建議** |
| 4 — Quantum Simulations, Prototypes, Testbeds and Applications | benchmarking, validation, reproducibility, and numerical techniques | 純再現性／基準研究；但此 track 其餘主題幾乎都是量子網路 |

---

## 2. 預印本現況

- **arXiv:2607.10284v1**（quant-ph，2026-07-11），6 頁，IEEE 會議格式。
  *Rank-Refined Quantum-Behaved Particle Swarm Optimization for Quantum Molecular Generation.*
  PDF 存於 [`../paper/RR-QPSO_arXiv-2607.10284v1.pdf`](../paper/RR-QPSO_arXiv-2607.10284v1.pdf)。
- 只有 v1；arXiv 頁面沒有 comments／journal-ref。
- LaTeX 原稿在 Overleaf（不在 repo）。
- 預印本頁數 6 < 8，格式上有空間補統計結果。

---

## 3. 預印本的每一個宣稱 vs 目前的證據

**結論先講：v1 的主要宣稱，以專案自己後續預登記的資料來看都站不住。原樣投稿等於
送出作者已知與資料矛盾的數字。** 下表每列都可追到 repo 內文件或叢集上的檔案。

| # | 預印本宣稱 | 出處 | 目前證據 | 判定 |
|---|---|---|---|---|
| C1 | BO 基準 V×U = **0.902**，「在相同實作與協定下重跑」 | 摘要、§IV-A、Fig. 2、§IV-B、結論 | 找不到任何能產生 0.902（V 94.2／U 95.7）的 run。codebase 中唯一 0.9 級的 BO 數字是寫死的文獻值 0.8834（Chen 2025，V 95.5／U 92.5）。自家 BO 全預算最佳 0.715。 | ✗ 無法溯源（ERRATA #3） |
| C2 | RR-QPSO 0.930 > QPSO+Sobol 0.914 > QPSO 0.905（M=64） | Fig. 2、§IV-A | n = 1。Fig. 2 兩根 QPSO 柱標籤**對調**，Sobol 的結論方向寫反（ERRATA #2）。10 配對 seed：p = 0.71。後續比較又帶四個偏向 QPSO 的混淆（FAIRNESS_AUDIT），**兩者至今未被公平比較**；公平重跑第一批因參數被靜默丟棄而作廢（§5）。 | ✗ 不可主張 |
| C3 | M 掃描：M=128 達 0.942（Table I、Fig. 3） | §IV-B | Table I 六列 V×U **皆可由 CSV 重現**。但全是 seed=0 單次執行；winner's curse 使單次最大值高估 +0.011～+0.027（RESULTS §7）。專案筆記中 M=128 的 best_params 以 24 個取樣種子重評，期望值約 0.931 ± 0.004。 | △ 數字可溯源，需改報期望值並註明 n = 1 |
| C4 | 比 BO 高 0.040（4 個百分點） | §IV-B、結論 | 分母即 C1。參考 BO（Ax/BoTorch GPEI，ARD）預登記比較：960 次評估中位 **0.929** vs RR-QPSO 0.779，H1/H2 **反向**（δ = −0.84／−0.86）；205 次（原始 QMG 程式自己的預算）也是 BO 完勝。9,664 次未能測（約 760 h/seed）。 | ✗ 反向 |
| C5 | 多目標：RR-QPSO（M=32）V×U 79.0% vs BO 43.8% | §IV-C、Fig. 4 | n = 1；BO 43.8% 的出處尚未像 C1 一樣核對。預登記實驗八（n = 10）：RR-QPSO **勝 CMA-ES 10/10**（δ = +0.94，p_holm = 0.002，成立）；勝 QPSO 不成立。實驗九（n = 45）RR vs QPSO **反轉**（Δ = −0.027）。 | ✗／△ BO 部分未驗證 |
| C6 | 「8 顆 V100 平行評估」＋Table I 執行時間 | 摘要、Table I | 實測每節點 8 個 CUDA-Q worker 會因逾時作廢約一半評估（32 個只回 17 個有效）；2 個／節點吞吐量最佳。原始長跑可能受影響，執行時間非乾淨的效能數據。 | △ 需註明 |

### 仍然站得住、而且有證據的結果

| # | 結果 | 證據強度 | 來源 |
|---|---|---|---|
| P1 | 單次執行比較最適化器會系統性誤導：取樣雜訊造成的選擇偏誤 +0.011～+0.027，大於 v1 報告的兩個差距 | 5 組參數 × 24 取樣種子，z = +2.9～+3.8 | RESULTS §7 |
| P2 | **基準實作品質主導結論方向**：手寫各向同性 GP-EI vs 參考 Ax GPEI，差 +0.296（10/10，p_holm = 0.0029）；換成參考 BO 後「族群式勝 BO」反轉 | 預登記，n = 10 | RESULTS §7h、PREREGISTRATION_axbo |
| P3 | 約束目標（HBA/HBD）上 QPSO 家族勝 CMA-ES（δ = +0.94，10/10） | 預登記，n = 10 | RESULTS §7e |
| P4 | 八演算法 9,664 次評估比較：QPSO 0.975、CMA-ES 0.971、RR-QPSO 0.964、DE 0.916、SPSA 0.719、Sobol 0.570（BO 列為缺陷實作，已劃掉） | n = 5，Friedman p = 0.00009；成對比較結構上不可能顯著 | EXPERIMENTS §1 |
| P5 | 確認性消融：OBL、AE mbest 皆未通過（p_holm = 0.131／0.075）；效果量收縮 ×0.55～0.67 | 預登記，n = 30 | RESULTS §6 |
| P6 | 工程：每節點 2 個 CUDA-Q worker 最佳；預算由 `BaseOptimizer` 強制；完整 N=9 動態電路無法經 CUDA-Q 送上 QPU（Base Profile QIR 禁止分支；85 個 `if`、90 次 MCM） | 實測 | README「Implementation notes」 |

---

## 4. 投稿前的三個決定

1. **重複發表（2026-10-12 確認）。** 稿件未錄取於 IEEE QCE 2026 workshop，故無重複發表問題
   （送 QCNC 時不可同時在其他會議／期刊審稿）。QCE 的審稿意見將由作者提供，改寫時逐條回應。
2. **稿件方向：A，並同時重跑公平比較（2026-10-12 決定）。** 重跑結果若能在
   定稿前依預登記 §8.5 的規則完成就納入，否則稿件只描述為「已預登記、進行中」。
   - **A. 方法學／再評估稿（建議）**：以 P1–P6 為主體，誠實報告 v1 宣稱在嚴格設定下不成立。
     資料已齊，工作量主要在寫作。草稿基礎：[`PAPER_DRAFT_results_discussion.md`](PAPER_DRAFT_results_discussion.md)
     ——**但其 §2「族群式決定性勝過 BO」與建議標題已被實驗十一推翻，不可沿用。**
   - **B. 演算法稿（先補公平比較）**：依 [`PREREGISTRATION_rrqpso_fair.md`](PREREGISTRATION_rrqpso_fair.md)
     以修正後的程式重跑。上一批 150 runs 約花 3 天（9/17 22:15 → 9/21 00:07），
     今天送出最快 10/15–16 完成，只剩 3–4 天寫作。即使 H1 成立，C1／C4 的 BO 問題仍在，
     標題與摘要仍須改寫。
   - ~~C. 原樣投 v1~~：不建議——C1、C2、C4 與作者自己的預登記資料矛盾。
3. **共同作者同意**（待辦）：改變稿件方向與結論前，需取得其餘三位作者同意（v1 有四位作者）。

---

## 5. 公平重跑（rrfair）第一批：作廢

| 項目 | 內容 |
|---|---|
| 資料 | 叢集 `~/sqmg_project-cudaq/results_rrfair/`；本機 `SQMG/experiments/99_voided_superseded/results_rrfair_void_0917/` |
| 規模 | 5 arms × 30 seeds（300–329）× 2,000 evals，150 runs，2026-09-17 ～ 09-21 |
| 作廢原因 | `RRQPSO2.__init__(*args, **kwargs)` 讓 `inspect.signature` 看不到具名參數，`run_experiment.py` 靜默丟掉所有覆寫（commit `7c09dcb` 已修）。log 可見 `⚠ rr_qpso2 不接受 alpha 參數，忽略覆寫` |
| 後果 | (1) 三個消融 arm 與 `rr2` 跑的是**同一組態**——500 次評估時四個 arm 的 30 個數值完全相同；(2) `qpso` 用 α∈[0.3,1.2]，`rr_qpso2` 退回預設 [0.5,1.0]，**重新引入了本研究要消除的 α 混淆** |
| 修正後重跑 | 2026-10-12 送出，`results_rrfair2/`；預登記修訂 §8（含新增的組態檢查與「截稿前不偷看」規則）在送出前 commit |

作廢資料上的分析數字（`results_rrfair/stats/analysis.txt`）**不可引用**：
@2,000 Δ = −0.0045（p = 0.55）、@500 Δ = +0.033（p = 0.013）。
後者極可能就是 α 差異造成的早期收斂差，而非 RR 機制。

---

## 6. 匿名化檢查清單（double-blind）

- [ ] 作者、單位、email 全部移除（IEEEtran 無匿名模式，手動改 `\author{}`）
- [ ] 刪除 Acknowledgment（NCHC 計算資源會洩漏單位）；camera-ready 再加回
- [ ] 不放 GitHub 連結（repo 名與 README 含作者姓名）；改用 <https://anonymous.4open.science> 產生匿名鏡像
- [ ] 引用自己的 arXiv:2607.10284 時用第三人稱（"Wu et al. [x] reported …"），不寫 "our previous work"
- [ ] 若沿用 v1 標題，審稿人可直接搜到 arXiv 版本——方向 A 本來就需要新標題
- [ ] PDF metadata（Title／Author 欄位）清空：`\hypersetup{pdfauthor={}}` 或 `exiftool -all=`
- [ ] 圖檔內嵌文字、檔名不含姓名或單位（`fig1_workflow.pdf` 需檢查）
- [ ] 範本改為 `\documentclass[conference,compsoc]{IEEEtran}`（Computer Society 格式）
- [ ] AI 使用聲明：IEEE 要求揭露；不含身分資訊，可保留在正文末

---

## 7. 時程（建議）

| 日期 | 工作 |
|---|---|
| 10/12 | ✓ 決定方向；✓ 送出 rrfair 重跑（H1 兩臂先排，預估 10/13 晚間完成，全部 150 runs 約 10/15） |
| 10/13–17 | 寫作；所有數字從 `figures_method/stats*.json` 取，不手抄 |
| 10/17 | 共同作者審閱 |
| 10/18 | 匿名化、compsoc 範本編譯、頁數 ≤ 8 檢查 |
| **10/19** | 上傳 EasyChair（留一天緩衝，因時區未註明） |

---

## 8. 資料地圖：預印本的每張圖表 → 原始資料

本機根目錄 `SQMG/experiments/`（不進 git）；完整打包見 GitHub Release
`data-2026-10-12`（附 `manifest.csv` SHA-256）。

| 預印本 | 內容 | 資料 |
|---|---|---|
| Fig. 2 | BO／QPSO／QPSO+Sobol／RR-QPSO 長條圖 | `90_paper_original_runs/results_qpso_nosobol`（**無** Sobol，0.9144）、`results_qpso_pure`（**有** Sobol，0.9046）、`results_v8`（RR-QPSO M=64，0.9310）；BO 0.902 **無資料** |
| Fig. 3、Table I | M 掃描 16–128 | `90_paper_original_runs/results_sweep_M{16,32,48,96,128}` + `results_v8`（M=64） |
| Fig. 4 | HBA/HBD 多目標 | `90_paper_original_runs/results_hbahbd_multiobj`；BO 43.8% 出處待查 |
| — | 方法學研究（P1–P5） | `01`–`13` 各資料夾；統計值在 repo `figures_method/stats*.json` |
| — | 作廢批次 | `99_voided_superseded/`（含 rrfair 第一批） |

⚠ 目錄名 `results_qpso_pure` 與 `results_qpso_nosobol` 的語意與內容**相反**，一律以 log 內的初始化設定為準。
