"""
==============================================================================
optimizers/bayesopt.py — Bayesian Optimization / Batch BO（v12.0）
==============================================================================

BO 是這篇工作的主要對照組（Chen 2025 的 QMG 參數搜尋即用 BO），因此這裡
把「序列 BO」與「批次 BO」分開實作，兩者共用同一個 GP 代理模型與 EI 採集函數，
差別只在一次提出幾個點。

為什麼要分兩個
--------------
序列 BO 每次只提一個點，資訊利用最充分，但在 48 GPU 的叢集上只會用到 1 顆 GPU
——它在「相同評估次數」下可能表現最好，卻在「相同 wall-clock」下最慢。
批次 BO（q-EI）才是與族群式演算法對等的比較對象。把兩者都放進來，
才能區分「BO 的樣本效率好」與「BO 不適合平行硬體」這兩件不同的事。

實作選擇
--------
- 代理模型：sklearn 的 GaussianProcessRegressor，Matérn(ν=2.5) + 白雜訊。
  134 維下 GP 本來就吃力，Matérn 2.5 比 RBF 對高維的過度平滑假設稍微寬容。
- 採集函數：Expected Improvement（EI），封閉解，無須取樣。
- 候選點：不做連續空間的梯度最佳化（134 維下不划算且容易卡住），
  改用 Thompson 式的大量隨機候選 + EI 排序，這是高維 BO 的常見務實做法。
- 批次策略：Kriging Believer（又稱 constant liar 的 GP 版本）——
  選出一點後，先用 GP 的「預測平均值」當作假的觀測值餵回模型，再選下一點。
  這樣同一批次內的點才會彼此分散，而不是全部擠在同一個 EI 峰值上。

GP 的 O(n^3) 成本在 n=512 時仍可接受（~0.1s 等級），不會成為瓶頸——
真正的成本永遠是量子電路取樣。

放置位置：optimizers/bayesopt.py
==============================================================================
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import BaseOptimizer


def _fit_gp(X: np.ndarray, y: np.ndarray, seed: int, tune: bool = True,
            kernel=None):
    """
    配一個 Matérn GP。

    ★ 用「等向性」（isotropic）Matérn，而非每維一個 length-scale 的 ARD：
      D=134 而預算只有數百點，ARD 的 134 個 length-scale 在統計上根本無法辨識，
      徒然把超參最佳化變成高維非凸問題（實測會慢到讓 Batch BO 跑不完）。
      等向性核在這個 n≪D 的情境下既快又是比較誠實的模型假設。

    `tune=False` 時沿用既有超參數、只重新條件化資料——
    Kriging Believer 在同一批次內加入假觀測時就用這個模式，
    避免每加一點就重跑一次超參最佳化。
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

    if kernel is None:
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(1e-4, (1e-8, 1e-1)))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1 if tune else 0,
        optimizer="fmin_l_bfgs_b" if tune else None,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


def _expected_improvement(gp, Xc: np.ndarray, y_best: float,
                          xi: float = 0.01) -> np.ndarray:
    """EI（最大化版本）。xi 為探索係數。"""
    from scipy.stats import norm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = gp.predict(Xc, return_std=True)
    sd = np.maximum(sd, 1e-12)
    imp = mu - y_best - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


class BayesianOptimization(BaseOptimizer):
    """
    序列 BO：GP + EI，每次提出 1 個點。

    初始設計用 Sobol，點數取 `min(2*batch_size, max_evals//4)`——
    太少 GP 配不出東西，太多就變成隨機搜尋。
    """
    name = "bo"

    def __init__(self, *args, n_init: int = None, n_candidates: int = 4000,
                 xi: float = 0.01, max_gp_points: int = 400,
                 tune_every: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_init = n_init if n_init is not None else min(
            2 * self.batch_size, max(8, self.max_evals // 4))
        self.n_candidates = n_candidates
        self.xi = xi
        self.q = 1          # 每次提出的點數（Batch BO 覆寫）

        # ★ GP 訓練點數上限。
        #   GP 的成本是 O(n³)：n=9,664 時單次配模型要數分鐘，而 BO 每次迭代都要配一次
        #   → 大預算下 BO 會慢到跑不完（實測 18.5 s/eval，9,664 次要 50 小時）。
        #   文獻上 GP-based BO 本來就很少跑到上萬次評估；超過時的標準做法是
        #   只用「最好的一部分 + 最近的一部分」點來配 GP。
        #   這是 BO 在大預算下的**真實限制**，不是我們對它不公平——
        #   報告時應說明 BO 的樣本效率優勢主要體現在數百次評估的區間。
        self.max_gp_points = max_gp_points

        # ★ 超參數重調頻率。
        #   GP 的成本分兩塊：(a) 超參最佳化（貴，L-BFGS 要反覆做 Cholesky）、
        #   (b) 條件化＋預測（便宜）。每次迭代都重調 (a) 的話，實測 1,200 次評估
        #   都跑不完。標準做法是讓超參數「慢慢跟上」——每 tune_every 次迭代才重調，
        #   其餘迭代沿用既有超參數、只把新觀測條件化進去。
        #   對 BO 的行為影響很小（超參數本來就變化緩慢），但速度差一個數量級。
        self.tune_every = tune_every
        self._kernel_cache = None

    def _gp_subset(self, X: np.ndarray, y: np.ndarray):
        """超過上限時，取『最佳一半 + 最近一半』作為 GP 訓練集。"""
        n = len(X)
        if n <= self.max_gp_points:
            return X, y
        k = self.max_gp_points // 2
        best_idx = np.argsort(-y)[:k]              # 最好的 k 個
        recent_idx = np.arange(n - k, n)           # 最近的 k 個
        idx = np.unique(np.concatenate([best_idx, recent_idx]))
        return X[idx], y[idx]

    def _optimize(self) -> None:
        # ── 初始設計（或從斷點還原）──────────────────────────────────────
        if self._resumed_X is not None and len(self._resumed_X):
            X, y = self._resumed_X, self._resumed_y
            self.logger.info(f"  [bo] 從斷點續跑，已有 {len(X)} 個觀測點。")
        else:
            X = self._sobol(self.n_init)
            y = self._evaluate(X)
            X = X[: len(y)]
            self.iteration += 1
            self._save_state(X, y)

        # ── BO 主迴圈 ────────────────────────────────────────────────────
        it = 0
        while True:
            # 訓練集受 max_gp_points 限制；超參數每 tune_every 次才重調一次
            Xg, yg = self._gp_subset(X, y)
            do_tune = (self._kernel_cache is None) or (it % self.tune_every == 0)
            gp = _fit_gp(Xg, yg, self.seed, tune=do_tune,
                         kernel=None if do_tune else self._kernel_cache)
            if do_tune:
                self._kernel_cache = gp.kernel_
            it += 1

            # Kriging Believer：批次內逐點挑選，已選點先用 GP 預測值頂替。
            # 加入假觀測後只重新條件化（tune=False），不重跑超參最佳化。
            X_aug, y_aug = Xg.copy(), yg.copy()
            picked = []
            for _ in range(self.q):
                cand = self.rng.random((self.n_candidates, self.D))
                ei = _expected_improvement(gp, cand, y_aug.max(), self.xi)
                x_new = cand[int(np.argmax(ei))]
                picked.append(x_new)

                if self.q > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y_fake = float(gp.predict(x_new[None, :])[0])
                    X_aug = np.vstack([X_aug, x_new])
                    y_aug = np.append(y_aug, y_fake)
                    gp = _fit_gp(X_aug, y_aug, self.seed, tune=False,
                                 kernel=self._kernel_cache)

            X_new = np.array(picked)
            y_new = self._evaluate(X_new)

            X = np.vstack([X, X_new[: len(y_new)]])
            y = np.append(y, y_new)
            self.iteration += 1
            # 每輪存檔：撞到時間上限時可從這裡續跑
            if self.iteration % 10 == 0:
                self._save_state(X, y)


class BatchBayesianOptimization(BayesianOptimization):
    """
    批次 BO：同一個 GP + EI，但每次用 Kriging Believer 一口氣提出
    q = batch_size 個點，讓整批可以在 48 GPU 上同時評估。

    與序列 BO 的唯一差別就是 q。這是刻意的：兩者共用完全相同的代理模型與
    採集函數，因此實驗結果的差異可以乾淨地歸因於「批次化」本身，
    而不是模型或超參數不同。
    """
    name = "batch_bo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = self.batch_size
