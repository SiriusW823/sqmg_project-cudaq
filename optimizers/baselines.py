"""
==============================================================================
optimizers/baselines.py — Sobol / DE / CMA-ES / SPSA（v12.0）
==============================================================================

四個非 BO、非 QPSO 的基準演算法。全部用 numpy 自行實作，不引入額外相依
（cma、nevergrad 等），理由是叢集上的 cudaq-v071 環境要盡量保持乾淨，
而這四個演算法的核心都夠短、也夠標準，自己寫反而更容易確認行為正確。

共同約定：
  - 搜尋空間 [0,1]^134，越界一律 clip（而非 reject），與既有 QPSO 一致。
  - 目標為「最大化」fitness；內部若習慣最小化者自行取負號。
  - 預算控制交給 BaseOptimizer._evaluate()，這裡不做次數判斷。

放置位置：optimizers/baselines.py
==============================================================================
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


# ===========================================================================
# 1. Sobol random search（最弱基準線）
# ===========================================================================

class SobolRandomSearch(BaseOptimizer):
    """
    Owen-scrambled Sobol 低差異序列的純隨機搜尋。

    這是整個比較的「地板」：任何有學習能力的演算法都應該勝過它。
    用 Sobol 而非均勻亂數，是為了讓地板本身盡量強——低差異序列在
    134 維下的覆蓋均勻性優於 pseudo-random，是個誠實的對照組。
    """
    name = "sobol"

    def _optimize(self) -> None:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.D, scramble=True, seed=self.seed)
        while True:
            X = sampler.random(self.batch_size)
            self._evaluate(X)
            self.iteration += 1


# ===========================================================================
# 2. Differential Evolution（DE/rand/1/bin）
# ===========================================================================

class DifferentialEvolution(BaseOptimizer):
    """
    標準 DE/rand/1/bin（Storn & Price 1997）。

    參數採文獻常用預設：F=0.5（差分權重）、CR=0.9（交叉率）。
    族群大小 = batch_size，使每一代剛好是一個可完全平行的批次，
    與其他族群式演算法在 GPU 使用率上站在同一個起跑點。
    """
    name = "de"

    def __init__(self, *args, F: float = 0.5, CR: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.F  = F
        self.CR = CR

    def _optimize(self) -> None:
        NP = self.batch_size
        P  = self._uniform(NP)
        f  = self._evaluate(P)
        self.iteration += 1

        while True:
            # 產生一整代的試驗向量（trial vectors），整代一次平行評估
            trials = np.empty_like(P)
            for i in range(NP):
                # rand/1：從族群中隨機挑三個互異且不等於 i 的個體
                idxs = [j for j in range(NP) if j != i]
                a, b, c = self.rng.choice(idxs, size=3, replace=False)
                mutant = P[a] + self.F * (P[b] - P[c])

                # binomial 交叉，至少保證一個維度來自 mutant
                cross = self.rng.random(self.D) < self.CR
                if not cross.any():
                    cross[self.rng.integers(self.D)] = True
                trial = np.where(cross, mutant, P[i])
                trials[i] = self._clip(trial)

            ft = self._evaluate(trials)

            # 貪婪選擇（逐個體比較）。ft 可能因預算截斷而較短。
            n = len(ft)
            better = ft > f[:n]
            P[:n][better] = trials[:n][better]
            f[:n][better] = ft[better]
            self.iteration += 1


# ===========================================================================
# 3. CMA-ES（Covariance Matrix Adaptation Evolution Strategy）
# ===========================================================================

class CMAES(BaseOptimizer):
    """
    CMA-ES（Hansen & Ostermeier 2001）的標準實作，含 rank-μ 更新與
    step-size 的 CSA（cumulative step-size adaptation）。

    在 134 維下 CMA-ES 是相當強的對手，也是這類連續黑箱問題的公認基準。
    注意 λ（族群大小）固定為 batch_size 而非文獻預設的
    `4 + floor(3 ln D)`（D=134 時約 18），理由是要與其他演算法共用同一個
    批次大小，確保每一代的平行度相同；μ 仍取 λ/2 並用對數權重。
    """
    name = "cmaes"

    def __init__(self, *args, sigma0: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma0 = sigma0

    def _optimize(self) -> None:
        D   = self.D
        lam = self.batch_size
        mu  = lam // 2

        # 對數遞減權重（rank-μ）
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # 策略參數（Hansen 的標準設定）
        cc    = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs    = (mueff + 2) / (D + mueff + 5)
        c1    = 2 / ((D + 1.3) ** 2 + mueff)
        cmu   = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        chiN  = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

        xmean = self.rng.random(D)
        sigma = self.sigma0
        pc    = np.zeros(D)
        ps    = np.zeros(D)
        B     = np.eye(D)
        Dg    = np.ones(D)
        C     = np.eye(D)
        eigen_last = 0

        while True:
            # ── 取樣 λ 個子代 ──────────────────────────────────────────
            Z = self.rng.standard_normal((lam, D))
            Y = Z @ (B * Dg).T
            X = xmean + sigma * Y
            Xc = self._clip(X)

            f = self._evaluate(Xc)       # 越界解以 clip 後的值評估
            n = len(f)
            # n < lam 代表預算在這一代中途用盡；仍用拿到的部分做一次更新，
            # 下一次 _evaluate() 會丟出 BudgetExhausted 正常收尾。

            # ── 選擇（最大化 → 由大到小排序）──────────────────────────
            order = np.argsort(-f)[:min(mu, n)]
            k = len(order)
            wk = w[:k] / w[:k].sum()

            xold  = xmean.copy()
            xmean = wk @ Xc[order]
            yw    = (xmean - xold) / sigma

            # ── step-size 控制（CSA）──────────────────────────────────
            C_invsqrt = B @ np.diag(1 / Dg) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_invsqrt @ yw)
            self.iteration += 1
            hsig = (np.linalg.norm(ps) /
                    np.sqrt(1 - (1 - cs) ** (2 * self.iteration)) / chiN) < (1.4 + 2 / (D + 1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw

            # ── 共變異數矩陣更新（rank-1 + rank-μ）────────────────────
            artmp = (Xc[order] - xold) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * (artmp.T * wk) @ artmp)

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-8, 1.0))

            # 特徵分解成本 O(D^3)，依 Hansen 建議攤提，不必每代做
            if self.n_evals - eigen_last > lam * D / (10 * (c1 + cmu) * D):
                eigen_last = self.n_evals
                C = np.triu(C) + np.triu(C, 1).T      # 強制對稱
                vals, B = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-20)
                Dg = np.sqrt(vals)


# ===========================================================================
# 4. SPSA（Simultaneous Perturbation Stochastic Approximation）
# ===========================================================================

class SPSA(BaseOptimizer):
    """
    SPSA（Spall 1992）。

    每次迭代只用「兩次」目標函數評估就估出整個 134 維的梯度方向——
    這是它在高維度、昂貴目標函數下的最大賣點，也是量子變分演算法
    （VQE/QAOA）最常用的最適化器，因此很適合放進這個比較。

    為了填滿 GPU 並與其他演算法對齊批次大小，這裡跑 `batch_size / 2` 條
    獨立的 SPSA 鏈（每條各自 ±擾動 = 2 次評估），最後回報最佳的一條。
    這是 SPSA 在平行硬體上的標準用法，沒有改變演算法本身。

    增益序列採 Spall 建議的形式：
        a_k = a / (k + 1 + A)^alpha,   c_k = c / (k + 1)^gamma
        alpha = 0.602, gamma = 0.101
    """
    name = "spsa"

    def __init__(self, *args, a: float = 0.15, c: float = 0.10,
                 alpha: float = 0.602, gamma: float = 0.101, **kwargs):
        super().__init__(*args, **kwargs)
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma

    def _optimize(self) -> None:
        n_chains = max(1, self.batch_size // 2)
        A = 0.10 * (self.max_evals / (2 * n_chains))   # Spall: A ≈ 10% 的總迭代數

        theta = self.rng.random((n_chains, self.D))

        # 先評估起始點，讓 best-so-far 曲線從第一次評估就有值
        self._evaluate(theta)
        self.iteration += 1

        k = 0
        while True:
            ak = self.a / ((k + 1 + A) ** self.alpha)
            ck = self.c / ((k + 1) ** self.gamma)

            # Rademacher ±1 擾動（SPSA 的標準選擇）
            delta = self.rng.choice([-1.0, 1.0], size=(n_chains, self.D))

            theta_p = self._clip(theta + ck * delta)
            theta_m = self._clip(theta - ck * delta)

            # 兩批各 n_chains 次評估：+ 擾動與 - 擾動
            f_p = self._evaluate(theta_p)
            f_m = self._evaluate(theta_m)

            n = min(len(f_p), len(f_m))
            if n == 0:
                return

            # 最大化 → 沿梯度「上升」
            ghat = ((f_p[:n] - f_m[:n])[:, None]) / (2.0 * ck * delta[:n])
            theta[:n] = self._clip(theta[:n] + ak * ghat)

            k += 1
            self.iteration += 1
