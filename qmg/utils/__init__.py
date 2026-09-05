"""qmg.utils — 化學/線路工具。

★ 為何電路建構器要延遲載入
--------------------------
`build_dynamic_circuit_cudaq` 需要 `cudaq`，而 cudaq 只裝在 cudaq-v071 環境。
Ax/BoTorch baseline（optimizers/axbo.py）必須跑在獨立的 ax-bo 環境（ax 會拉
CUDA 13，與 cudaq 0.7.1 的 CUDA 12 衝突），該環境沒有 cudaq。

但 ax-bo 那一側的父行程只需要 `ConditionalWeightsGenerator`（純 rdkit/numpy）
——實際的線路模擬是由 subprocess worker 在 cudaq-v071 底下執行的。若在此處
無條件 import cudaq，父行程會在 import 階段就死掉。

因此：cudaq 相關的名稱改為延遲代理。**沒有靜默降級**——只要真的去碰
`DynamicCircuitBuilder`，就會原封不動拋出當初的 ImportError，訊息說明是哪個
環境缺什麼。這樣「該有 cudaq 卻沒有」的情境仍會立刻爆，不會變成難查的錯誤。
"""
from .chemistry_data_processing import MoleculeQuantumStateGenerator
from .weight_generator import ConditionalWeightsGenerator
from .fitness_calculator import FitnessCalculator, FitnessCalculatorWrapper

try:
    from .build_dynamic_circuit_cudaq import (
        DynamicCircuitBuilderCUDAQ as DynamicCircuitBuilder)
    CUDAQ_IMPORT_ERROR = None
except ImportError as _e:          # 只吞 ImportError；其他錯誤照常往上拋
    CUDAQ_IMPORT_ERROR = _e

    class _MissingCudaq:
        """碰到就爆，且爆得清楚。"""

        def __init__(self, *a, **k):
            raise ImportError(
                "DynamicCircuitBuilder 需要 cudaq，但這個 Python 環境沒有它。"
                f"（原始錯誤：{CUDAQ_IMPORT_ERROR}）\n"
                "  → 模擬工作必須在 cudaq-v071 底下執行；若父行程跑在 ax-bo，"
                "請確認 QMG_WORKER_PYTHON 指向 cudaq-v071 的直譯器。"
            ) from CUDAQ_IMPORT_ERROR

    DynamicCircuitBuilder = _MissingCudaq

__all__ = [
    "MoleculeQuantumStateGenerator", "ConditionalWeightsGenerator",
    "FitnessCalculator", "FitnessCalculatorWrapper",
    "DynamicCircuitBuilder", "CUDAQ_IMPORT_ERROR",
]
