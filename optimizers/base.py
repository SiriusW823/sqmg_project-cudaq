"""
==============================================================================
optimizers/base.py — 所有最適化器的共同基底（v12.0）
==============================================================================

這個檔案存在的唯一理由是「讓 8 種演算法可以被公平比較」。

公平比較的三個支柱
------------------
1. **相同的評估預算**：所有演算法共用同一個 `max_evals`（＝ M × T）。
   預算由基底類別強制執行，而不是交給各演算法自律——`_evaluate()` 會在
   超出預算時截斷該批次並丟出 `BudgetExhausted`，因此像 BO 這種「一次一點」
   與 CMA-ES 這種「一次一族群」的演算法，最終吃到的目標函數次數完全相同。
   這是黑箱最適化 benchmark 的標準做法：橫軸是評估次數，不是迭代次數。

2. **相同的搜尋空間與目標函數**：D=134 的 [0,1]^D，fitness = V × U
   （或由 `fitness_fn` 覆寫，供 HBA/HBD 多目標版使用）。

3. **相同的紀錄格式**：所有演算法寫出欄位完全一致的 CSV，
   因此比較圖表可以用同一支分析程式產生，不需要為每個演算法寫 parser。

刻意「不」統一的部分
--------------------
初始化策略保持各演算法原生的做法（RR-QPSO 用 Sobol、CMA-ES 用高斯、
BO 用 Sobol 初始設計……），因為初始化本來就是演算法的一部分——
RR-QPSO 的 Sobol 初始化正是它宣稱的貢獻之一，把它抽掉就不是在比同一件事了。
初始化造成的變異由「同一實驗跑 5 個 seed」來吸收。

放置位置：optimizers/base.py
==============================================================================
"""
from __future__ import annotations

import csv
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np


class BudgetExhausted(Exception):
    """評估預算用盡。由 `_evaluate()` 丟出，`run()` 攔截後正常收尾。"""


class BaseOptimizer(ABC):
    """
    所有最適化器的基底。

    子類別只需要實作 `_optimize()`，在裡面自由呼叫 `self._evaluate(X)`；
    預算控制、CSV 紀錄、最佳解追蹤、計時都由基底處理。
    """

    # 所有演算法共用的 CSV 欄位（分析程式只認這一組）
    CSV_FIELDS = [
        "eval_index",      # 第幾次目標函數評估（1-based，跨迭代連續）
        "iteration",       # 演算法自己的迭代編號（BO 等序列法可為 eval_index）
        "algo",
        "seed",
        "validity",
        "uniqueness",
        "fitness",
        "best_fitness",    # 到此為止的最佳（best-so-far，畫收斂曲線用）
        "best_validity",
        "best_uniqueness",
        "elapsed_s",
    ]

    # 子類別覆寫：演算法在註冊表中的名字
    name: str = "base"

    def __init__(
        self,
        n_params:          int,
        max_evals:         int,
        batch_evaluate_fn: Callable[[np.ndarray], List[Tuple[float, float]]],
        logger:            logging.Logger,
        seed:              int = 0,
        data_dir:          str = "results_benchmark",
        task_name:         str = "run",
        batch_size:        int = 32,
        fitness_fn:        Optional[Callable[[Tuple[float, ...]], float]] = None,
        resume:            bool = False,
    ):
        self.D          = n_params
        self.max_evals  = max_evals
        self.batch_eval = batch_evaluate_fn
        self.logger     = logger
        self.seed       = seed
        self.batch_size = batch_size
        self.fitness_fn = fitness_fn
        self.rng        = np.random.default_rng(seed)

        self.lb = np.zeros(self.D, dtype=np.float64)
        self.ub = np.ones(self.D,  dtype=np.float64)

        # 追蹤狀態
        self.n_evals   = 0
        self.iteration = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f    = -np.inf
        self.best_v    = 0.0
        self.best_u    = 0.0
        self.history: List[dict] = []

        os.makedirs(data_dir, exist_ok=True)
        self._csv_path   = os.path.join(data_dir, f"{task_name}.csv")
        self._state_path = os.path.join(data_dir, f"{task_name}_state.npz")
        self.resume = resume

        # ── 斷點續跑 ─────────────────────────────────────────────────────
        #   長跑（近萬次評估）有可能撞上 SLURM 的時間上限；序列型的 BO 尤其
        #   （9,664 次 × ~18.5s ≈ 50h > nchc 的 48h）。
        #   resume=True 時從既有 CSV 還原進度並「附加」而非覆寫，
        #   讓同一個 run 可以跨多個作業完成。
        self._resumed_X: Optional[np.ndarray] = None
        self._resumed_y: Optional[np.ndarray] = None
        if resume and os.path.exists(self._csv_path):
            self._restore()
        else:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writeheader()

        self._t0 = None

    # ---------------------------------------------------------- checkpoint
    def _restore(self) -> None:
        """從既有 CSV（與 state 檔）還原 n_evals / best / 已評估點。"""
        with open(self._csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        self.n_evals = len(rows)
        last = rows[-1]
        self.best_f = float(last["best_fitness"])
        self.best_v = float(last["best_validity"])
        self.best_u = float(last["best_uniqueness"])
        self.iteration = int(last["iteration"])

        if os.path.exists(self._state_path):
            d = np.load(self._state_path)
            self._resumed_X = d["X"]
            self._resumed_y = d["y"]
            if len(self._resumed_X):
                self.best_x = self._resumed_X[int(np.argmax(self._resumed_y))]

        self.logger.info(
            f"[resume] 從 {self._csv_path} 還原：已完成 {self.n_evals}/{self.max_evals} "
            f"次評估，目前最佳 {self.best_f:.4f}")

    def _save_state(self, X: np.ndarray, y: np.ndarray) -> None:
        """儲存已評估的點，供斷點續跑重建代理模型（BO 用）。"""
        try:
            np.savez_compressed(self._state_path, X=X, y=y)
        except Exception as e:                        # noqa: BLE001
            self.logger.warning(f"[checkpoint] 儲存 state 失敗（不影響執行）：{e}")

    # ---------------------------------------------------------------- utils
    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _uniform(self, n: int) -> np.ndarray:
        return self.rng.random((n, self.D))

    def _sobol(self, n: int) -> np.ndarray:
        """Owen-scrambled Sobol 初始設計。n 非 2 的冪次時取前 n 個點。"""
        from scipy.stats import qmc
        m = int(np.ceil(np.log2(max(n, 2))))
        sampler = qmc.Sobol(d=self.D, scramble=True, seed=self.seed)
        return sampler.random_base2(m=m)[:n]

    def _to_fitness(self, metrics: Tuple[float, ...]) -> float:
        if self.fitness_fn is None:
            return float(metrics[0]) * float(metrics[1])
        return float(self.fitness_fn(metrics))

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.n_evals)

    # ------------------------------------------------------------- evaluate
    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """評估一批候選解，回傳 fitness 陣列（預算與紀錄見 `_evaluate_metrics`）。"""
        return np.array([self._to_fitness(m) for m in self._evaluate_metrics(X)],
                        dtype=np.float64)

    def _evaluate_metrics(self, X: np.ndarray) -> List[Tuple[float, ...]]:
        """
        評估一批候選解，回傳「原始 metrics」串列（(V, U, ...)）。

        ★ 預算在這裡強制執行：
          - 若這批會超出預算，先截斷到剛好用完為止（回傳的長度會比 X 短）；
          - 用完後丟出 BudgetExhausted，讓演算法立刻停止。
          子類別只要不去攔截這個例外，就自動獲得「精確用滿預算」的行為。

        回傳原始 metrics（而非 fitness）是為了讓既有的 RR-QPSO 最適化器
        可以原封不動地被包起來——它需要拿到 (V, U) 才能做 V-U 解耦 mbest。
        """
        if self.remaining <= 0:
            raise BudgetExhausted()

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        truncated = False
        if X.shape[0] > self.remaining:
            X = X[: self.remaining]
            truncated = True

        if self._t0 is None:
            self._t0 = time.time()

        results = self.batch_eval(X)

        fits = np.empty(len(results), dtype=np.float64)
        for i, metrics in enumerate(results):
            v, u = float(metrics[0]), float(metrics[1])
            f = self._to_fitness(metrics)
            fits[i] = f

            self.n_evals += 1
            if f > self.best_f:
                self.best_f = f
                self.best_v = v
                self.best_u = u
                self.best_x = X[i].copy()

            row = {
                "eval_index":      self.n_evals,
                "iteration":       self.iteration,
                "algo":            self.name,
                "seed":            self.seed,
                "validity":        round(v, 6),
                "uniqueness":      round(u, 6),
                "fitness":         round(f, 6),
                "best_fitness":    round(self.best_f, 6),
                "best_validity":   round(self.best_v, 6),
                "best_uniqueness": round(self.best_u, 6),
                "elapsed_s":       round(time.time() - self._t0, 2),
            }
            self.history.append(row)
            with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self.CSV_FIELDS).writerow(row)

        self.logger.info(
            f"  [{self.name}] iter={self.iteration:3d}  "
            f"evals={self.n_evals}/{self.max_evals}  "
            f"batch_best={fits.max():.4f}  best={self.best_f:.4f} "
            f"(V={self.best_v:.3f} U={self.best_u:.3f})  "
            f"t={time.time()-self._t0:.0f}s"
        )

        # 若 truncated 或預算已滿，先讓呼叫端拿到這批結果，
        # 下一次呼叫 _evaluate_metrics 時才丟 BudgetExhausted。
        return list(results)

    # ------------------------------------------------------------------ run
    def run(self) -> Tuple[np.ndarray, float]:
        self._t0 = time.time()
        self.logger.info("=" * 70)
        self.logger.info(f"最適化器：{self.name}")
        self.logger.info(f"  維度 D        : {self.D}")
        self.logger.info(f"  評估預算      : {self.max_evals}")
        self.logger.info(f"  批次大小      : {self.batch_size}")
        self.logger.info(f"  seed          : {self.seed}")
        self.logger.info(f"  CSV           : {self._csv_path}")
        self.logger.info("=" * 70)

        try:
            self._optimize()
        except BudgetExhausted:
            self.logger.info(f"  [{self.name}] 預算用盡，正常結束。")

        elapsed = time.time() - self._t0
        self.logger.info("=" * 70)
        self.logger.info(
            f"[{self.name}] 完成  評估 {self.n_evals}/{self.max_evals}  "
            f"最佳 fitness={self.best_f:.6f}  V={self.best_v:.4f}  U={self.best_u:.4f}  "
            f"耗時 {elapsed/60:.1f} 分"
        )
        self.logger.info("=" * 70)
        return self.best_x, self.best_f

    @abstractmethod
    def _optimize(self) -> None:
        """子類別實作：自由呼叫 self._evaluate(X)，不必自己管預算。"""
