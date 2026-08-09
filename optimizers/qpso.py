"""
==============================================================================
optimizers/qpso.py — QPSO（原版）與 RR-QPSO（本專案方法）（v12.0）
==============================================================================

兩個 QPSO 家族的成員：

`QPSO`      —— 標準量子行為粒子群（Sun et al. 2004/2012）的乾淨實作，
               只有 delta 勢阱位置更新 + mbest，不含本專案的任何增強。
               它是 RR-QPSO 的「消融基準」：兩者差多少，就是 RR-QPSO
               那些機制（Sobol 初始化、rank-refined mbest、OBL、V-U 解耦）
               真正貢獻的量。

`RRQPSO`    —— 直接包裝既有的 `qpso_optimizer_ae.AESOQPSOOptimizer`，
               而不是重寫一份。這點很重要：論文宣稱的就是那份程式碼的行為，
               重寫一份「應該等價」的版本只會製造出無法辯護的差異。
               包裝層只做兩件事：把評估導向 BaseOptimizer 的預算/紀錄管線，
               以及把 max_evals 換算成它需要的 (M, T)。

放置位置：optimizers/qpso.py
==============================================================================
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, BudgetExhausted


# ===========================================================================
# 標準 QPSO（消融基準）
# ===========================================================================

class QPSO(BaseOptimizer):
    """
    標準 QPSO（Sun et al. 2012, Eq. 12）。

        mbest = (1/M) Σ pbest_i
        p_i   = φ·pbest_i + (1-φ)·gbest,        φ ~ U(0,1)
        x_i   = p_i ± α·|mbest - x_i|·ln(1/u),  u ~ U(0,1)

    收縮擴張係數 α 由 alpha_max 線性遞減到 alpha_min（文獻最常見的排程）。
    初始化用均勻亂數——**刻意不用 Sobol**，因為 Sobol 初始化是 RR-QPSO
    的貢獻之一，放進基準線就測不出它的效果了。
    """
    name = "qpso"

    def __init__(self, *args, alpha_max: float = 1.0, alpha_min: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min

    def _optimize(self) -> None:
        M = self.batch_size
        # 預估總迭代數，供 α 線性排程使用（預算用盡時會提前中斷）
        T_est = max(1, self.max_evals // M - 1)

        X = self._uniform(M)
        f = self._evaluate(X)
        n = len(f)
        X = X[:n]

        pbest   = X.copy()
        pbest_f = f.copy()
        g       = int(np.argmax(pbest_f))
        gbest   = pbest[g].copy()
        self.iteration += 1

        t = 0
        while True:
            alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * (t / T_est)
            alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))

            mbest = pbest.mean(axis=0)

            phi = self.rng.random((len(X), self.D))
            p   = phi * pbest + (1 - phi) * gbest

            u    = np.maximum(self.rng.random((len(X), self.D)), 1e-12)
            sign = np.where(self.rng.random((len(X), self.D)) < 0.5, -1.0, 1.0)
            X    = self._clip(p + sign * alpha * np.abs(mbest - X) * np.log(1.0 / u))

            f = self._evaluate(X)
            n = len(f)

            better = f > pbest_f[:n]
            pbest[:n][better]   = X[:n][better]
            pbest_f[:n][better] = f[better]

            g = int(np.argmax(pbest_f))
            gbest = pbest[g].copy()

            t += 1
            self.iteration += 1


# ===========================================================================
# RR-QPSO（包裝既有的 AESOQPSOOptimizer）
# ===========================================================================

class RRQPSO(BaseOptimizer):
    """
    本專案的方法。直接驅動 `qpso_optimizer_ae.AESOQPSOOptimizer`。

    預算換算：AESOQPSO 的總評估數為 `M*(T+1) + M`（最後一項是 OBL 的
    對立粒子批次）。給定 max_evals 與 M，取
        T = max_evals // M - 2
    使其略低於預算，剩餘的零頭由 BaseOptimizer 的預算控制吸收——
    無論如何都不會超過 max_evals，因此與其他演算法仍是同一條起跑線。
    """
    name = "rr_qpso"

    # 可個別關閉的組件（供消融實驗使用）
    ABLATIONS = ("none", "sobol", "obl", "ae", "vu", "mc")

    def __init__(self, *args, obl: bool = True,
                 alpha_max: float = None, alpha_min: float = None,
                 ablate: str = "none", **kwargs):
        super().__init__(*args, **kwargs)
        # None = 沿用 AESOQPSOOptimizer 自己的預設（1.2 / 0.30）
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min

        if ablate not in self.ABLATIONS:
            raise ValueError(f"ablate 須為 {self.ABLATIONS} 之一，收到 {ablate!r}")
        self.ablate = ablate

        # ── 各組件的開關（依 ablate 決定）────────────────────────────────
        #   ★ sobol_init 特別說明：
        #     Sobol 初始化「不在」AESOQPSOOptimizer 裡，而是舊 runner
        #     run_qpso_qmg_cudaq.py 在建構後以 make_sobol_positions() 覆寫
        #     optimizer.positions / optimizer.pbest 實現的。
        #     v12.x 的 wrapper 先前漏了這一步 → 所有 RR-QPSO run 其實都
        #     缺少 Sobol 初始化。此處補回，並保留關閉選項供消融。
        self.use_sobol = (ablate != "sobol")
        self.obl       = obl and (ablate != "obl")
        self.use_ae    = (ablate != "ae")
        self.use_vu    = (ablate != "vu")
        self.use_mc    = (ablate != "mc")

    def _sobol_positions(self, M: int):
        """複製舊 runner 的 make_sobol_positions 行為（scramble=True, seed=0）。"""
        import math
        from scipy.stats import qmc
        k = math.ceil(math.log2(max(M, 2)))
        sampler = qmc.Sobol(d=self.D, scramble=True, seed=0)   # 固定 seed=0
        return sampler.random(n=2 ** k)[:M].copy()

    def _optimize(self) -> None:
        from qpso_optimizer_ae import AESOQPSOOptimizer

        M = self.batch_size
        # 扣掉 Phase 0 與 OBL 各一個批次
        T = max(1, self.max_evals // M - (2 if self.obl else 1))

        def batch_fn(positions: np.ndarray):
            # 導回基底：預算控制、CSV、best-so-far 都在這裡統一處理
            return self._evaluate_metrics(positions)

        alpha_kw = {}
        if self.alpha_max is not None:
            alpha_kw["alpha_max"] = self.alpha_max
        if self.alpha_min is not None:
            alpha_kw["alpha_min"] = self.alpha_min

        opt = AESOQPSOOptimizer(
            n_params            = self.D,
            n_particles         = M,
            max_iterations      = T,
            logger              = self.logger,
            batch_evaluate_fn   = batch_fn,
            fitness_fn          = self.fitness_fn,
            seed                = self.seed,
            data_dir            = "/tmp",          # 它自己的 CSV 不用，統一走基底的
            task_name           = f"_rrqpso_inner_{self.seed}",
            obl                 = self.obl,
            ae_weighting        = self.use_ae,
            vu_decouple         = self.use_vu,
            mode_collapse_u_thresh = 0.20 if self.use_mc else 0.0,
            compare_bo_baseline = False,
            **alpha_kw,
        )

        # ── Sobol 初始化（複製舊 runner 的覆寫行為）──────────────────────
        if self.use_sobol:
            try:
                pos = self._sobol_positions(M)
                opt.positions = pos.copy()
                opt.pbest     = pos.copy()
                # pbest_fit 維持 -inf → Phase 0 評估後正常建立
                self.logger.info(
                    f"  [rr_qpso] Sobol 初始化已套用 shape={pos.shape} "
                    f"range=[{pos.min():.4f}, {pos.max():.4f}]")
            except Exception as e:                       # noqa: BLE001
                self.logger.warning(f"  [rr_qpso] Sobol 初始化失敗，改用隨機：{e}")

        self.logger.info(
            f"  [rr_qpso] 組件狀態  ablate={self.ablate}  "
            f"sobol={self.use_sobol} obl={self.obl} ae={self.use_ae} "
            f"vu={self.use_vu} mode_collapse={self.use_mc}")

        opt.optimize()     # BudgetExhausted 會往上傳到 BaseOptimizer.run()
