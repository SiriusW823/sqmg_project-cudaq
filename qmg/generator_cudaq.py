"""
==============================================================================
generator_cudaq.py  (修正版)
CUDA-Q 版本的 MoleculeGenerator，對應 Qiskit 版的 qmg/generator.py
==============================================================================

修正清單：
  [BUG-1] 相對 import 錯誤 → 改為完整套件路徑
  [BUG-2] uniqueness 計算：len(smiles_dict)-1 在 validity=100% 時少算 1
  [BUG-3] _CUDAQ_TARGET_MAP 補齊常用 alias，移除不存在的 nvidia-mgpu
  [BUG-4] cudaq.set_random_seed 加 try/except 兼容不同版本 API
  [BUG-5] result.get_bitstrings() 改用 result.items() 避免大量記憶體複製
==============================================================================
"""
from __future__ import annotations

import numpy as np
from typing import List, Union, Tuple

import cudaq

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ── [BUG-1 修正] 使用完整套件路徑，而非裸 import ─────────────────────────────
from qmg.utils.chemistry_data_processing import MoleculeQuantumStateGenerator
from qmg.utils.weight_generator import ConditionalWeightsGenerator
from qmg.utils.build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ


# ===========================================================================
# Backend 映射表
# ===========================================================================
# [BUG-3 修正] 補齊 alias、移除不穩定的 nvidia-mgpu（單機不需要）
_CUDAQ_TARGET_MAP = {
    # CPU 模擬
    "cudaq_qpp":         "qpp-cpu",     # CPU 模擬（無 GPU 時 fallback）
    "qpp-cpu":           "qpp-cpu",

    # 單 GPU cuStateVec（V100 推薦）
    "cudaq_nvidia":      "nvidia",      # FP32（預設，速度最快）
    "nvidia":            "nvidia",
    "cudaq_nvidia_fp64": "nvidia-fp64", # FP64（精度較高，較慢）
    "nvidia-fp64":       "nvidia-fp64",

    # Qiskit-Aer 向後相容 alias
    "qiskit_aer":        "qpp-cpu",
}


def _set_target_safe(target_name: str) -> str:
    """
    嘗試設定 CUDA-Q target；若不可用則 fallback 到 qpp-cpu。
    回傳實際使用的 target 名稱。
    """
    try:
        cudaq.set_target(target_name)
        return target_name
    except Exception as e:
        import warnings
        warnings.warn(
            f"[CUDAQ] target='{target_name}' 不可用（{e}），"
            f"自動 fallback 到 qpp-cpu。"
        )
        cudaq.set_target("qpp-cpu")
        return "qpp-cpu"


# ===========================================================================
# MoleculeGeneratorCUDAQ
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器。
    公開介面與 Qiskit MoleculeGenerator 完全相同：
        __init__(num_heavy_atom, all_weight_vector, backend_name, ...)
        update_weight_vector(w)
        sample_molecule(num_sample) → (smiles_dict, validity, uniqueness)
    """

    def __init__(
        self,
        num_heavy_atom:            int,
        all_weight_vector:         Union[List[float], np.ndarray, None] = None,
        backend_name:              str   = "cudaq_nvidia",
        temperature:               float = 0.2,
        dynamic_circuit:           bool  = True,
        remove_bond_disconnection: bool  = True,
        chemistry_constraint:      bool  = True,
    ):
        if not dynamic_circuit:
            raise NotImplementedError("CUDA-Q 版目前僅支援 dynamic_circuit=True。")

        self.num_heavy_atom            = num_heavy_atom
        self.all_weight_vector         = (
            np.array(all_weight_vector, dtype=np.float64)
            if all_weight_vector is not None else None
        )
        self.backend_name              = backend_name
        self.temperature               = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint

        # 電路建構器（提供 kernel + bond fix post-processing）
        self.circuit_builder = DynamicCircuitBuilderCUDAQ(
            num_heavy_atom            = num_heavy_atom,
            temperature               = temperature,
            remove_bond_disconnection = remove_bond_disconnection,
            chemistry_constraint      = chemistry_constraint,
        )
        self.kernel = self.circuit_builder.get_kernel()

        # 分子解碼器（複用 QMG 原版）
        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size = num_heavy_atom,
            ncpus           = 1,
            sanitize_method = "strict",
        )

        # 設定 CUDA-Q backend
        actual_target = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(actual_target)
        print(f"[CUDAQ] Generator initialized. target='{self._active_target}'  "
              f"N={num_heavy_atom}  weight_dim={self.circuit_builder.length_all_weight_vector}")

    # ----------------------------------------------------------------
    def update_weight_vector(
        self, all_weight_vector: Union[List[float], np.ndarray]
    ) -> None:
        self.all_weight_vector = np.array(all_weight_vector, dtype=np.float64)

    # ----------------------------------------------------------------
    def sample_molecule(
        self,
        num_sample:  int,
        random_seed: int = 0,
    ) -> Tuple[dict, float, float]:
        """
        執行量子電路採樣並解碼為 SMILES 字典。

        Returns:
            smiles_dict  : {smiles_str: shot_count}（含 None key 代表無效分子）
            validity     : 有效分子 shots / 總 shots
            uniqueness   : 獨特有效 SMILES 數 / 有效 shots
        """
        assert self.all_weight_vector is not None, \
            "請先呼叫 update_weight_vector() 或在建構時傳入 all_weight_vector。"

        w = self.all_weight_vector
        assert len(w) == self.circuit_builder.length_all_weight_vector, (
            f"weight 長度不符：{len(w)} != {self.circuit_builder.length_all_weight_vector}"
        )

        # [BUG-4 修正] set_random_seed 跨版本 API 保護
        try:
            cudaq.set_random_seed(random_seed)
        except AttributeError:
            pass  # 部分舊版 CUDA-Q 無此函式，略過

        # ── 量子採樣 ──────────────────────────────────────────────────
        result = cudaq.sample(
            self.kernel,
            w.tolist(),              # @cudaq.kernel 需要 list[float]
            shots_count=num_sample,
        )

        # ── Bit-string 解碼 ────────────────────────────────────────────
        # [BUG-5 修正] 使用 result.items() 取得 (bitstring, count) 對
        #   避免 get_bitstrings() 展開所有 shots 導致記憶體暴增
        smiles_dict:   dict[str, int] = {}
        num_valid_shots = 0

        for bs, count in result.items():
            # Bond disconnection post-processing
            bs_fixed      = self.circuit_builder.apply_bond_disconnection_correction(bs)

            # bit reorder → canonical quantum state
            quantum_state = self.data_generator.post_process_quantum_state(
                bs_fixed, reverse=False   # CUDA-Q mz() 順序 = clbit 0→N，不需反轉
            )

            smiles = self.data_generator.QuantumStateToSmiles(quantum_state)
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + count

            if smiles and smiles != "None":
                num_valid_shots += count

        # ── 指標計算 ──────────────────────────────────────────────────
        validity   = num_valid_shots / num_sample

        # [BUG-2 修正] 直接數 valid SMILES key，不依賴 None key 是否存在
        num_unique_valid = len([
            k for k in smiles_dict
            if k and k != "None"
        ])
        uniqueness = (
            num_unique_valid / num_valid_shots
            if num_valid_shots > 0 else 0.0
        )

        return smiles_dict, validity, uniqueness


# 相容性別名（給舊版 import 使用）
MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=== MoleculeGeneratorCUDAQ 功能驗證 ===")
    N = 5
    cwg = ConditionalWeightsGenerator(N, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    gen = MoleculeGeneratorCUDAQ(N, all_weight_vector=w, backend_name="cudaq_qpp")
    t0  = time.time()
    smiles_dict, validity, uniqueness = gen.sample_molecule(1000)
    print(f"Validity  : {validity:.3f}")
    print(f"Uniqueness: {uniqueness:.3f}")
    print(f"V×U       : {validity*uniqueness:.4f}")
    print(f"Elapsed   : {time.time()-t0:.1f}s")
    top5 = [(s, c) for s, c in sorted(smiles_dict.items(), key=lambda x: -x[1]) if s and s != "None"][:5]
    print("Top-5:", top5)