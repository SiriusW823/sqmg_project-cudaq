"""
==============================================================================
optimizers/ — SQMG 參數搜尋演算法套件（v12.0）
==============================================================================

八個可互換的最適化器，共用 `BaseOptimizer` 的介面、預算控制與 CSV 格式，
因此可以在相同評估預算下公平比較。

    名稱          類別                          類型
    ------------  ----------------------------  ----------------------------
    sobol         SobolRandomSearch             低差異隨機搜尋（地板基準）
    bo            BayesianOptimization          GP + EI，序列（一次 1 點）
    batch_bo      BatchBayesianOptimization     GP + EI，q-EI 批次
    cmaes         CMAES                         演化策略（共變異數自適應）
    de            DifferentialEvolution         DE/rand/1/bin
    spsa          SPSA                          同時擾動隨機逼近
    qpso          QPSO                          標準量子行為粒子群（消融基準）
    rr_qpso       RRQPSO                        本專案方法

用法：
    from optimizers import get_optimizer
    cls = get_optimizer("cmaes")
    opt = cls(n_params=134, max_evals=512, batch_evaluate_fn=fn, logger=log, ...)
    best_x, best_f = opt.run()

放置位置：optimizers/__init__.py
==============================================================================
"""
from .base import BaseOptimizer, BudgetExhausted
from .baselines import SobolRandomSearch, DifferentialEvolution, CMAES, SPSA
from .bayesopt import BayesianOptimization, BatchBayesianOptimization
from .qpso import QPSO, RRQPSO

REGISTRY = {
    "sobol":    SobolRandomSearch,
    "bo":       BayesianOptimization,
    "batch_bo": BatchBayesianOptimization,
    "cmaes":    CMAES,
    "de":       DifferentialEvolution,
    "spsa":     SPSA,
    "qpso":     QPSO,
    "rr_qpso":  RRQPSO,
}

# ★ Ax/BoTorch GPEI（論文原始 BO baseline 的忠實移植）。
#   延遲註冊：它需要 ax-platform 與 torch，而那些只裝在獨立的 ax-bo 環境裡
#   （ax 會拉 CUDA 13，與 cudaq-v071 的 CUDA 12 衝突）。在 cudaq-v071 底下
#   import 失敗是預期行為，不應讓整個 REGISTRY 無法載入。
#   註：axbo.py 把 ax/torch 的 import 放在 _optimize() 內（避免載入 REGISTRY
#   就付出 torch 的啟動成本），因此模組層級的 import 一定會成功。要讓 REGISTRY
#   如實反映可用性，必須主動探測 ax 本身是否存在。
try:
    import importlib.util as _ilu
    if _ilu.find_spec("ax") is None or _ilu.find_spec("torch") is None:
        raise ImportError("ax / torch 不在此環境（需 ax-bo 環境）")
    from .axbo import AxBO
    REGISTRY["ax_bo"] = AxBO
except Exception:      # noqa: BLE001
    AxBO = None

# 圖表與表格的顯示名稱（順序即為預設的呈現順序：由弱到強、本方法殿後）
DISPLAY_NAMES = {
    "sobol":    "Sobol random search",
    "spsa":     "SPSA",
    "de":       "Differential Evolution",
    "cmaes":    "CMA-ES",
    "bo":       "Bayesian Opt. (sequential)",
    "batch_bo": "Batch BO (q-EI)",
    "ax_bo":    "Bayesian Opt. (Ax/BoTorch GPEI)",
    "qpso":     "QPSO",
    "rr_qpso":  "RR-QPSO (ours)",
}

PLOT_ORDER = ["sobol", "spsa", "de", "cmaes", "bo", "batch_bo", "qpso", "rr_qpso"]

# 每次評估只用 1 顆 GPU、無法填滿批次的演算法（排程時要特別處理）
SEQUENTIAL = {"bo", "ax_bo"}


def get_optimizer(name: str):
    key = name.strip().lower()
    if key not in REGISTRY:
        raise KeyError(
            f"未知的最適化器 '{name}'。可用：{sorted(REGISTRY)}"
        )
    return REGISTRY[key]


__all__ = [
    "BaseOptimizer", "BudgetExhausted",
    "SobolRandomSearch", "DifferentialEvolution", "CMAES", "SPSA",
    "BayesianOptimization", "BatchBayesianOptimization",
    "QPSO", "RRQPSO",
    "REGISTRY", "DISPLAY_NAMES", "PLOT_ORDER", "SEQUENTIAL", "get_optimizer",
]
