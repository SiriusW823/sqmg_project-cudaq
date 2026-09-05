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

    def _optimize(self) -> None:
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
        while True:
            params, trial_index = ax_client.get_next_trial()
            x = np.array([params[f"x{i+1}"] for i in range(self.D)],
                         dtype=np.float64)
            metrics = self._evaluate_metrics(x.reshape(1, -1))[0]
            fit = self._to_fitness(metrics)
            ax_client.complete_trial(trial_index=trial_index,
                                     raw_data={"vu": (float(fit), None)})
            self.iteration += 1
