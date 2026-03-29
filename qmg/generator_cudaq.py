"""
==============================================================================
generator_cudaq.py  (完整修正版)
CUDA-Q 版本的 MoleculeGenerator
==============================================================================
修正清單：
  [FIX-1] result.items() 在不同 CUDA-Q 版本行為不穩定
          → 改用 for bs in result / result[bs] 最保守 iteration
  [FIX-2] uniqueness 分母使用 num_valid_shots，分子直接 count valid keys
          → 避免 None key 存在與否影響計算
  [FIX-3] bitstring 長度保護：kernel 若提前 return 產生短 bs 直接歸入 invalid
  [FIX-4] cudaq.set_random_seed try/except 兼容不同版本 API
  [FIX-5] _set_target_safe fallback 邏輯與警告訊息強化
==============================================================================
"""
from __future__ import annotations

import warnings
import numpy as np
from typing import List, Union, Tuple

import cudaq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from qmg.utils.chemistry_data_processing import MoleculeQuantumStateGenerator
from qmg.utils.weight_generator import ConditionalWeightsGenerator
from qmg.utils.build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ


# ===========================================================================
# Backend 映射表
# ===========================================================================
_CUDAQ_TARGET_MAP = {
    # CPU
    "cudaq_qpp":         "qpp-cpu",
    "qpp-cpu":           "qpp-cpu",
    # 單 GPU cuStateVec FP32（V100 推薦）
    "cudaq_nvidia":      "nvidia",
    "nvidia":            "nvidia",
    # 單 GPU cuStateVec FP64
    "cudaq_nvidia_fp64": "nvidia-fp64",
    "nvidia-fp64":       "nvidia-fp64",
    # 多 GPU
    "cudaq_mqpu":        "nvidia-mqpu",
    "nvidia-mqpu":       "nvidia-mqpu",
    # 向後相容
    "qiskit_aer":        "qpp-cpu",
}


def _set_target_safe(target_name: str) -> str:
    """
    嘗試設定 CUDA-Q target；若不可用則自動 fallback 到 qpp-cpu。
    回傳實際使用的 target 名稱。
    """
    try:
        cudaq.set_target(target_name)
        return target_name
    except Exception as e:
        warnings.warn(
            f"[CUDAQ] target='{target_name}' 不可用（{e}），"
            f"自動 fallback 到 qpp-cpu。"
        )
        cudaq.set_target("qpp-cpu")
        return "qpp-cpu"


def _sample_result_to_counts(result) -> dict:
    """
    將 cudaq.SampleResult 安全轉換為 {bitstring: count} dict。

    CUDA-Q 各版本 SampleResult API 差異：
      - result.items()       → 部分版本不支援或行為不穩定（不用）
      - for bs in result     → 所有版本支援，iterate unique bitstrings
      - result[bs]           → 所有版本支援，回傳 int count
      - result.get_bitstrings() → 展開全部 shots，記憶體暴增（僅 fallback 用）
    """
    counts: dict[str, int] = {}
    try:
        for bitstring in result:
            counts[bitstring] = result[bitstring]
    except Exception:
        # 最終保底：展開所有 shots（shots 多時記憶體大，但一定能跑）
        for bitstring in result.get_bitstrings():
            counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


# ===========================================================================
# MoleculeGeneratorCUDAQ
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器。
    公開介面與 Qiskit MoleculeGenerator 完全相同：
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

        # 電路建構器（kernel + bond disconnection 後處理）
        self.circuit_builder = DynamicCircuitBuilderCUDAQ(
            num_heavy_atom            = num_heavy_atom,
            temperature               = temperature,
            remove_bond_disconnection = remove_bond_disconnection,
            chemistry_constraint      = chemistry_constraint,
        )
        self.kernel = self.circuit_builder.get_kernel()

        # 分子解碼器
        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size = num_heavy_atom,
            ncpus           = 1,
            sanitize_method = "strict",
        )

        # 設定 CUDA-Q backend
        cudaq_target = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(cudaq_target)

        # 預期 bitstring 長度：N*(N+1)，N=9 時為 90
        self._expected_bs_len = num_heavy_atom * (num_heavy_atom + 1)

        print(
            f"[CUDAQ] Generator initialized. "
            f"target='{self._active_target}'  "
            f"N={num_heavy_atom}  "
            f"weight_dim={self.circuit_builder.length_all_weight_vector}  "
            f"expected_bs_len={self._expected_bs_len}"
        )

    # ------------------------------------------------------------------ #

    def update_weight_vector(
        self, all_weight_vector: Union[List[float], np.ndarray]
    ) -> None:
        """更新電路權重向量（每次 QPSO 評估前呼叫）。"""
        self.all_weight_vector = np.array(all_weight_vector, dtype=np.float64)

    # ------------------------------------------------------------------ #

    def sample_molecule(
        self,
        num_sample:  int,
        random_seed: int = 0,
    ) -> Tuple[dict, float, float]:
        """
        執行量子電路採樣，解碼為 SMILES 字典並計算指標。

        Returns
        -------
        smiles_dict : dict[str | None, int]
            {smiles_str: shot_count}，無效 bitstring 對應 key=None
        validity : float
            有效分子 shots / 總 shots
        uniqueness : float
            獨特有效 SMILES 數 / 有效 shots
        """
        if self.all_weight_vector is None:
            raise RuntimeError(
                "請先呼叫 update_weight_vector() 或在建構時傳入 all_weight_vector。"
            )

        w = self.all_weight_vector
        expected_dim = self.circuit_builder.length_all_weight_vector
        if len(w) != expected_dim:
            raise ValueError(
                f"weight 長度不符：got {len(w)}, expected {expected_dim}"
            )

        # [FIX-4] 跨版本 API 保護
        try:
            cudaq.set_random_seed(random_seed)
        except AttributeError:
            pass

        # ── 量子電路採樣 ──────────────────────────────────────────────
        result = cudaq.sample(
            self.kernel,
            w.tolist(),              # @cudaq.kernel 只接受 list[float]
            shots_count=num_sample,
        )

        # ── Bit-string 解碼 ────────────────────────────────────────────
        raw_counts = _sample_result_to_counts(result)  # [FIX-1]

        smiles_dict:    dict = {}
        num_valid_shots = 0

        for bs, count in raw_counts.items():

            # [FIX-3] bitstring 長度保護
            if len(bs) != self._expected_bs_len:
                smiles_dict[None] = smiles_dict.get(None, 0) + count
                continue

            # Bond disconnection 後處理（純 Python，不影響電路）
            bs_fixed = self.circuit_builder.apply_bond_disconnection_correction(bs)

            # bit reorder → canonical quantum state
            # CUDA-Q mz() 順序 = clbit index 0→N，不需反轉（reverse=False）
            quantum_state = self.data_generator.post_process_quantum_state(
                bs_fixed, reverse=False
            )

            smiles = self.data_generator.QuantumStateToSmiles(quantum_state)
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + count

            if smiles and smiles != "None":
                num_valid_shots += count

        # ── 指標計算 ──────────────────────────────────────────────────
        validity = num_valid_shots / num_sample

        # [FIX-2] 直接 count valid keys，不依賴 None key 是否存在
        num_unique_valid = sum(
            1 for k in smiles_dict if k and k != "None"
        )
        uniqueness = (
            num_unique_valid / num_valid_shots
            if num_valid_shots > 0 else 0.0
        )

        return smiles_dict, validity, uniqueness


# 相容性別名
MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證（python qmg/generator_cudaq.py）
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=== MoleculeGeneratorCUDAQ 功能驗證 ===")
    N = 5
    cwg = ConditionalWeightsGenerator(N, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    gen = MoleculeGeneratorCUDAQ(N, all_weight_vector=w, backend_name="cudaq_nvidia")
    t0  = time.time()
    smiles_dict, validity, uniqueness = gen.sample_molecule(1000)
    print(f"Validity  : {validity:.3f}")
    print(f"Uniqueness: {uniqueness:.3f}")
    print(f"V×U       : {validity * uniqueness:.4f}")
    print(f"Elapsed   : {time.time() - t0:.1f}s")
    top5 = [
        (s, c) for s, c in sorted(smiles_dict.items(), key=lambda x: -x[1])
        if s and s != "None"
    ][:5]
    print("Top-5 SMILES:", top5)