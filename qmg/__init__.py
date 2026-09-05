"""qmg — 量子分子生成器。

`generator_cudaq` 需要 cudaq，而 cudaq 只在 cudaq-v071 環境裡。Ax/BoTorch
baseline 的父行程跑在沒有 cudaq 的 ax-bo 環境，只用得到 `qmg.utils` 裡的純
rdkit 工具（見 qmg/utils/__init__.py 的說明）。此處同樣採「延遲代理」：
沒有 cudaq 時 import 成功，但一旦真的去實例化生成器就會拋出原始 ImportError。
"""
try:
    from .generator_cudaq import MoleculeGeneratorCUDAQ, MoleculeGenerator
    CUDAQ_IMPORT_ERROR = None
except ImportError as _e:
    CUDAQ_IMPORT_ERROR = _e

    class _MissingCudaq:
        def __init__(self, *a, **k):
            raise ImportError(
                "MoleculeGenerator 需要 cudaq，但這個 Python 環境沒有它。"
                f"（原始錯誤：{CUDAQ_IMPORT_ERROR}）\n"
                "  → 分子模擬必須在 cudaq-v071 底下；若父行程跑在 ax-bo，"
                "請設定 QMG_WORKER_PYTHON 指向 cudaq-v071 的直譯器。"
            ) from CUDAQ_IMPORT_ERROR

    MoleculeGeneratorCUDAQ = MoleculeGenerator = _MissingCudaq

__all__ = ["MoleculeGeneratorCUDAQ", "MoleculeGenerator", "CUDAQ_IMPORT_ERROR"]
