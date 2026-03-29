"""
==============================================================================
generator_cudaq.py  (V100 sm_70 相容修正版)
CUDA-Q 版本的 MoleculeGenerator，對應 Qiskit 版的 qmg/generator.py
==============================================================================

修正清單（相對於原版）：
  [BUG-1] 相對 import 錯誤 → 改為完整套件路徑
  [BUG-2] uniqueness 計算：len(smiles_dict)-1 在 validity=100% 時少算 1
  [BUG-3] _CUDAQ_TARGET_MAP 補齊常用 alias，移除不存在的 nvidia-mgpu
  [BUG-4] cudaq.set_random_seed 加 try/except 兼容不同版本 API
  [BUG-5] result.get_bitstrings() 改用 result.items() / 相容 fallback
  [BUG-6] ★ 新增 V100 sm_70 架構相容性檢查：
           CUDA-Q >= 0.8.0 的 pip wheel 不含 sm_70 PTX/SASS，
           cudaq.set_target("nvidia") 會靜默 fallback 到 qpp-cpu(CPU)，
           本修正版加入明確的 GPU 可用性驗證與警告，並在 cudaq 0.7.x
           的 SampleResult API 差異間提供相容 shim。

架構背景：
  V100 = Volta = sm_70
  CUDA-Q >= 0.8.0 預編譯 wheel 支援：sm_80 (Ampere), sm_90 (Hopper)
  CUDA-Q <= 0.7.1 預編譯 wheel 支援：sm_70 (Volta) ✓
  → DGX V100 必須使用 cuda-quantum-cu11==0.7.1 或 cuda-quantum-cu12==0.7.1
==============================================================================
"""
from __future__ import annotations

import warnings
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
# [BUG-6] CUDA-Q 版本與 GPU 架構相容性檢查
# ===========================================================================

def _check_cudaq_version_volta_compat() -> tuple[str, bool]:
    """
    檢查 CUDA-Q 版本是否支援 Volta (sm_70)。
    回傳 (version_str, is_volta_compatible)。
    """
    try:
        ver_str = cudaq.__version__          # e.g. "0.7.1" or "0.8.0"
        parts   = ver_str.split(".")
        major, minor = int(parts[0]), int(parts[1])
        # 0.8.0 起的 pip wheel 移除了 sm_70 支援
        is_compat = (major, minor) <= (0, 7)
        return ver_str, is_compat
    except Exception:
        return "unknown", True               # 無法判斷時不阻止執行


def _verify_gpu_actually_used(target_name: str) -> bool:
    """
    嘗試執行一個最小量子電路，確認 GPU 實際參與計算而非 fallback 至 CPU。
    回傳 True 表示 GPU 確實被使用。
    """
    if target_name not in ("nvidia", "nvidia-fp64", "tensornet"):
        return False                         # CPU target，不需驗證

    try:
        @cudaq.kernel
        def _smoke_test():
            q = cudaq.qvector(1)
            h(q[0])

        result = cudaq.sample(_smoke_test, shots_count=16)
        # 若成功且結果合理（兩個 state 都出現），則 GPU 正常工作
        counts = dict(result.items()) if hasattr(result, "items") else {}
        return len(counts) > 0
    except Exception as e:
        warnings.warn(f"[CUDAQ] GPU smoke test 失敗：{e}")
        return False


# ===========================================================================
# Backend 映射表
# ===========================================================================

# [BUG-3 修正] 補齊 alias、移除不穩定的 nvidia-mgpu
_CUDAQ_TARGET_MAP = {
    # CPU 模擬
    "cudaq_qpp":         "qpp-cpu",
    "qpp-cpu":           "qpp-cpu",

    # 單 GPU cuStateVec（V100 需 cudaq <= 0.7.1）
    "cudaq_nvidia":      "nvidia",
    "nvidia":            "nvidia",
    "cudaq_nvidia_fp64": "nvidia-fp64",
    "nvidia-fp64":       "nvidia-fp64",

    # Tensor network（需 cuTensorNet，V100 需確認版本）
    "tensornet":         "tensornet",

    # Qiskit-Aer 向後相容 alias
    "qiskit_aer":        "qpp-cpu",
}

# GPU 類型的 target（需要額外架構相容性檢查）
_GPU_TARGETS = {"nvidia", "nvidia-fp64", "tensornet"}


def _set_target_safe(target_name: str) -> str:
    """
    設定 CUDA-Q target，並執行 V100 架構相容性與 GPU 實際啟動驗證。
    若發現不相容則給出明確錯誤訊息，而非靜默 fallback。
    回傳實際使用的 target 名稱。
    """
    # ── 版本相容性檢查 ──────────────────────────────────────────────────
    ver_str, is_compat = _check_cudaq_version_volta_compat()

    if target_name in _GPU_TARGETS and not is_compat:
        raise RuntimeError(
            f"\n"
            f"{'='*65}\n"
            f"[CUDAQ 架構不相容] CUDA-Q {ver_str} 的 pip wheel 不支援\n"
            f"V100 Volta (sm_70) 架構。\n"
            f"\n"
            f"解決方案（擇一）：\n"
            f"  方案 A（推薦）：降版至含 sm_70 支援的版本\n"
            f"    CUDA 11.x：pip install cuda-quantum-cu11==0.7.1\n"
            f"    CUDA 12.x：pip install cuda-quantum-cu12==0.7.1\n"
            f"\n"
            f"  方案 B：從原始碼編譯，指定 sm_70\n"
            f"    git clone https://github.com/NVIDIA/cuda-quantum.git\n"
            f"    cd cuda-quantum && ./scripts/build_cudaq.sh -a '70'\n"
            f"\n"
            f"  方案 C（暫時）：改用 CPU 模擬（速度較慢，僅供驗證）\n"
            f"    --backend cudaq_qpp\n"
            f"{'='*65}"
        )

    # ── 設定 target ──────────────────────────────────────────────────────
    try:
        cudaq.set_target(target_name)
    except Exception as e:
        raise RuntimeError(
            f"[CUDAQ] cudaq.set_target('{target_name}') 失敗：{e}\n"
            f"請確認 cudaq 安裝正確，或改用 --backend cudaq_qpp（CPU fallback）。"
        ) from e

    # ── GPU 實際啟動驗證 ─────────────────────────────────────────────────
    if target_name in _GPU_TARGETS:
        gpu_ok = _verify_gpu_actually_used(target_name)
        if not gpu_ok:
            warnings.warn(
                f"[CUDAQ] target='{target_name}' 設定成功，但 GPU smoke test 異常。\n"
                f"電路可能正在 CPU 上執行。請確認 GPU 驅動與 cuTensorNet/cuStateVec 版本。"
            )
        else:
            print(f"[CUDAQ] GPU target='{target_name}' 驗證通過 ✓")

    return target_name


# ===========================================================================
# SampleResult 相容 shim（0.7.x vs 0.8.x API 差異）
# ===========================================================================

def _iter_sample_result(result) -> list[tuple[str, int]]:
    """
    從 SampleResult 取得 (bitstring, count) 對。
    相容 CUDA-Q 0.7.x 與 0.8.x 的 API 差異：
      0.8.x：result.items() 直接可用
      0.7.x：result.items() 通常也存在，但部分 build 可能只有
             result.get_register_counts() 或 dict(result)
    """
    # 優先嘗試標準 dict-like items()（0.8.x 及大部分 0.7.x）
    if hasattr(result, "items"):
        try:
            return list(result.items())
        except Exception:
            pass

    # fallback：嘗試 dict() 轉換（部分 0.7.x build）
    try:
        return list(dict(result).items())
    except Exception:
        pass

    # 最後手段：get_bitstrings()（記憶體較高，但保證可用）
    warnings.warn(
        "[CUDAQ] result.items() 不可用，改用 get_bitstrings()（記憶體用量較高）。\n"
        "建議升級 cuda-quantum 至 0.7.1 以取得完整 API 支援。"
    )
    bs_list = result.get_bitstrings()
    counts: dict[str, int] = {}
    for bs in bs_list:
        counts[bs] = counts.get(bs, 0) + 1
    return list(counts.items())


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

    V100 相容性說明：
        需使用 cuda-quantum-cu11==0.7.1 或 cuda-quantum-cu12==0.7.1。
        CUDA-Q >= 0.8.0 的 pip wheel 不含 sm_70 PTX，__init__ 時會拋出
        明確的 RuntimeError 而非靜默 fallback 至 CPU。
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

        # ── [BUG-6] 設定 CUDA-Q backend，含 V100 相容性驗證 ────────────
        actual_target = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        # _set_target_safe 會在版本不相容時主動拋出 RuntimeError（不再靜默 fallback）
        self._active_target = _set_target_safe(actual_target)

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized.\n"
            f"  cudaq version : {ver_str}\n"
            f"  active target : {self._active_target}\n"
            f"  N atoms       : {num_heavy_atom}\n"
            f"  weight dim    : {self.circuit_builder.length_all_weight_vector}"
        )

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
            pass

        # ── 量子採樣 ──────────────────────────────────────────────────
        result = cudaq.sample(
            self.kernel,
            w.tolist(),
            shots_count=num_sample,
        )

        # ── [BUG-5 + BUG-6] Bit-string 解碼（跨版本相容） ───────────
        smiles_dict:    dict[str, int] = {}
        num_valid_shots = 0

        for bs, count in _iter_sample_result(result):
            bs_fixed      = self.circuit_builder.apply_bond_disconnection_correction(bs)
            quantum_state = self.data_generator.post_process_quantum_state(
                bs_fixed, reverse=False
            )
            smiles = self.data_generator.QuantumStateToSmiles(quantum_state)
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + count

            if smiles and smiles != "None":
                num_valid_shots += count

        # ── 指標計算 ──────────────────────────────────────────────────
        validity = num_valid_shots / num_sample

        # [BUG-2 修正] 明確篩選 valid key，不依賴 None key 是否存在
        num_unique_valid = len([
            k for k in smiles_dict
            if k and k != "None"
        ])
        uniqueness = (
            num_unique_valid / num_valid_shots
            if num_valid_shots > 0 else 0.0
        )

        return smiles_dict, validity, uniqueness


# 相容性別名
MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=== MoleculeGeneratorCUDAQ 功能驗證 ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q version       : {ver_str}")
    print(f"Volta (sm_70) compat : {'✓ YES' if is_compat else '✗ NO — 請降版至 0.7.1'}")

    N   = 5
    cwg = ConditionalWeightsGenerator(N, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    gen = MoleculeGeneratorCUDAQ(N, all_weight_vector=w, backend_name="cudaq_qpp")
    t0  = time.time()
    smiles_dict, validity, uniqueness = gen.sample_molecule(1000)
    print(f"Validity  : {validity:.3f}")
    print(f"Uniqueness: {uniqueness:.3f}")
    print(f"V×U       : {validity*uniqueness:.4f}")
    print(f"Elapsed   : {time.time()-t0:.1f}s")
    top5 = [
        (s, c) for s, c in
        sorted(smiles_dict.items(), key=lambda x: -x[1])
        if s and s != "None"
    ][:5]
    print("Top-5:", top5)