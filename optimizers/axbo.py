"""
==============================================================================
optimizers/axbo.py — Ax / BoTorch GPEI（論文原始 BO baseline 的忠實移植）
==============================================================================

為何需要這支
------------
本專案原有的 `optimizers/bayesopt.py` 是手寫的 GP-EI，在三個地方比論文採用的
實作弱，而且每一項都對論文有利：

  |            | 論文（PEESEgroup/QMG） | optimizers/bayesopt.py |
  |------------|------------------------|------------------------|
  | kernel     | BoTorch 預設 ARD Matérn 5/2 | 各向同性 Matérn      |
  | GP 訓練點  | 全部                   | 上限 400               |
  | Sobol 初始 | 5 trials               | 128                    |
  | 加速       | GPU                    | CPU                    |

在 D=134 下，各向同性 kernel 等於宣稱 134 個參數的敏感度相同——這幾乎必然是
手寫版停在 0.69、而論文報告 0.90 的主因。實測顯示訓練點上限只能解釋差距的
22%（Δ=+0.0125，5/10，p=0.156），kernel 才是主嫌。

在這個 baseline 被修正之前，「族群式勝過 BO」只能當方向性主張。本檔案的目的
就是把它變成量化主張。

組態來源
--------
逐項對照 upstream `constrained_bo.py`：

    GenerationStep(model=Models.SOBOL, num_trials=5,
                   max_parallelism=1, model_kwargs={"seed": 42})
    GenerationStep(model=Models.GPEI, num_trials=-1, max_parallelism=1,
                   model_kwargs={"torch_dtype": torch.float64,
                                 "torch_device": cuda if available else cpu})
    AxClient(random_seed=42, generation_strategy=gs)
    parameters: x1..xD, range [0,1], float
    迴圈：get_next_trial() → evaluate → complete_trial()

唯一的刻意偏離
--------------
1. `torch_device` 固定為 CPU。ax-platform 0.4.3 會拉 torch 2.14 + CUDA 13，
   與叢集的 CUDA 12.2 驅動及 cuda-quantum 0.7.1 衝突，混裝會破壞整個環境。
   論文原始碼本來就有 CPU fallback（`"cuda" if torch.cuda.is_available()`），
   因此這是它支援的組態，不是我們發明的。GP 在 CPU 上較慢但結果相同。
2. `random_seed` 由實驗的 seed 提供而非固定 42——本研究是配對設計，
   每個 seed 必須給出獨立的一次執行。固定 42 會讓 10 個 seed 全部相同。

放置位置：optimizers/axbo.py
==============================================================================
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class AxBO(BaseOptimizer):
    """Ax/BoTorch GPEI。序列式：每次提出一個點。"""

    name = "ax_bo"

    def __init__(self, *args, n_sobol: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        # 論文用 5 個 Sobol trials 起手（相對 D=134 極少，但這是它的設定）
        self.n_sobol = n_sobol
        self._audited = False
        self._step_counts: dict = {}    # {模型名稱: 提出的 trial 數}

    # ── 預登記 §4.3 的強制組態檢查 ────────────────────────────────────────
    # 「確認 Ax 用的是 GPEI 且 kernel 為 ARD、沒有套用訓練點上限、且 GP 步之前
    #   恰好有 5 個 Sobol trial。若組態不符，整批作廢，不做任何假設檢定。」
    # 這三件事必須從 log 讀得出來，所以在這裡主動探測並印出，而不是靠事後推論。
    def _audit_kernel(self, ax_client) -> None:
        """第一次 GP 配適之後探測 kernel 與訓練點數，印成可稽核的一行。"""
        if self._audited:
            return
        try:
            mb = ax_client.generation_strategy.model      # ModelBridge
            if mb is None:
                return
            inner = getattr(mb, "model", None)            # BotorchModel
            gp = None
            for path in ("surrogate.model", "model"):
                obj = inner
                for attr in path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None:
                    gp = obj
                    break
            if gp is None:
                self.logger.warning(
                    "  [ax_bo][組態稽核] 找不到 GP 物件，無法驗證 kernel"
                    "——依預登記 §4.3 需人工確認後才可解讀結果")
                self._audited = True        # 只警告一次，不刷版
                return
            # ModelListGP 之類的容器：取第一個子模型
            if hasattr(gp, "models") and len(getattr(gp, "models", [])) > 0:
                gp = gp.models[0]

            ls = None
            for path in ("covar_module.base_kernel.lengthscale",
                         "covar_module.lengthscale"):
                obj = gp
                for attr in path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None:
                    ls = obj
                    break

            n_train = None
            for attr in ("train_inputs",):
                ti = getattr(gp, attr, None)
                if ti:
                    n_train = int(ti[0].shape[-2])
                    break

            if ls is None:
                self.logger.warning("  [ax_bo][組態稽核] 取不到 lengthscale")
            else:
                n_ls = int(ls.shape[-1])
                ard = (n_ls == self.D)
                self.logger.info(
                    f"  [ax_bo][組態稽核] kernel={type(gp.covar_module).__name__}"
                    f"/{type(getattr(gp.covar_module, 'base_kernel', gp.covar_module)).__name__}"
                    f"  lengthscale 維度={n_ls}  D={self.D}  "
                    f"ARD={'是' if ard else '否'}"
                )
                if not ard:
                    self.logger.error(
                        "  [ax_bo][組態稽核] ★ kernel 不是 ARD——依預登記 §4.3，"
                        "此批次作廢。")
            self.logger.info(
                f"  [ax_bo][組態稽核] GP 訓練點數={n_train}  已完成評估="
                f"{self.n_evals}  訓練上限=無（Ax 不設 max_gp_points）")
            self._audited = True
        except Exception as e:      # 稽核本身絕不能弄垮實驗
            self.logger.warning(f"  [ax_bo][組態稽核] 探測失敗（不影響執行）：{e}")
            self._audited = True

    def _optimize(self) -> None:
        # ★ ax_bo 不能續跑 ─────────────────────────────────────────────────
        # BaseOptimizer 的續跑機制只還原 CSV 與 n_evals；Ax 的內部狀態
        # （已完成的 trial、GP 的訓練資料、generation step 的位置）不在其中。
        # 若讓它續跑，Ax 會重新從 5 個 Sobol 開始，但預算已被扣掉一部分——
        # 得到的既不是 2,000 次的 GPEI，也不是任何可描述的演算法。
        # 這種「看起來有結果、其實不是預登記那個東西」正是作廢 HBA/HBD 批次的
        # 同一類錯誤，所以這裡選擇大聲失敗，讓該次 run 依 §4.4 被排除。
        if self.n_evals > 0:
            raise RuntimeError(
                f"ax_bo 偵測到既有進度（n_evals={self.n_evals}），但 Ax 的內部"
                "狀態無法還原，續跑會產生與預登記不同的演算法。\n"
                "  → 請刪除該 task 的 CSV 後從頭重跑，或依預登記 §4.4 將此 seed "
                "配對排除。切勿使用 --resume 跑 ax_bo。"
            )

        import torch
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        from ax.modelbridge.generation_strategy import (
            GenerationStrategy, GenerationStep)
        from ax.modelbridge.registry import Models

        torch.set_default_dtype(torch.float64)
        dev = torch.device("cpu")   # 見檔頭「刻意偏離」第 1 點

        gs = GenerationStrategy(steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=self.n_sobol,
                max_parallelism=1,
                model_kwargs={"seed": self.seed},
            ),
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,          # 不限；其餘預算全部給 GPEI
                max_parallelism=1,
                model_kwargs={"torch_dtype": torch.float64,
                              "torch_device": dev},
            ),
        ])

        ax_client = AxClient(random_seed=self.seed, generation_strategy=gs,
                             verbose_logging=False)
        ax_client.create_experiment(
            name=f"sqmg_{self.seed}",
            parameters=[{"name": f"x{i+1}", "type": "range",
                         "bounds": [0.0, 1.0], "value_type": "float"}
                        for i in range(self.D)],
            objectives={"vu": ObjectiveProperties(minimize=False)},
            overwrite_existing_experiment=True,
            is_test=True,
        )

        self.logger.info(
            f"  [ax_bo] Ax {getattr(__import__('ax'), '__version__', '?')}  "
            f"GPEI on {dev}  Sobol trials={self.n_sobol}  D={self.D}  "
            f"預算={self.max_evals}")

        # ── 主迴圈：BaseOptimizer 的預算控制會在上限時拋 BudgetExhausted ──
        prev_step = None
        try:
            while True:
                params, trial_index = ax_client.get_next_trial()

                # 記錄這一個 trial 是哪個 generation step 提出的（Sobol vs GPEI）。
                step = "?"
                try:
                    step = str(ax_client.generation_strategy._curr.model_name)
                except Exception:
                    try:
                        step = str(ax_client.generation_strategy.current_step.model)
                    except Exception:
                        pass
                self._step_counts[step] = self._step_counts.get(step, 0) + 1
                if step != prev_step:
                    self.logger.info(
                        f"  [ax_bo][組態稽核] 第 {self.n_evals} 次評估起改用 "
                        f"generation step：{step}"
                        + (f"（先前 {prev_step} 共提出 "
                           f"{self._step_counts.get(prev_step)} 個 trial）"
                           if prev_step else ""))
                    prev_step = step

                x = np.array([params[f"x{i+1}"] for i in range(self.D)],
                             dtype=np.float64)
                metrics = self._evaluate_metrics(x.reshape(1, -1))[0]
                fit = self._to_fitness(metrics)
                ax_client.complete_trial(trial_index=trial_index,
                                         raw_data={"vu": (float(fit), None)})
                self.iteration += 1

                # 第一次進到 GP 步之後才有 kernel 可探測
                if "SOBOL" not in step.upper():
                    self._audit_kernel(ax_client)
        finally:
            self.logger.info(
                f"  [ax_bo][組態稽核] 各 generation step 的 trial 數："
                f"{self._step_counts}（預登記要求 Sobol 恰為 {self.n_sobol}）")
