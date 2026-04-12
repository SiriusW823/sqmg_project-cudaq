"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v7)
==============================================================================

根本解法（基於官方文件）：
  cudaq 0.7.1 官方的正確模式是 closure/capture，不是 list[float] parameter。
  sample_molecule() 每次呼叫 build_kernel_from_weights(w) 建立新 kernel，
  再以 cudaq.sample(k, shots_count=N)（無額外參數）採樣。
  無參數 → __isBroadcast = False → 正確的 shot-by-shot 路徑。
==============================================================================
"""
from __future__ import annotations

import re
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
# 模組層級 smoke test kernel（無參數，不觸發 broadcast）
# ===========================================================================

@cudaq.kernel
def _smoke_kernel():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q[0])


# ===========================================================================
# CUDA-Q 版本與 V100 相容性
# ===========================================================================

def _check_cudaq_version_volta_compat() -> tuple[str, bool]:
    try:
        ver_str = cudaq.__version__
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', ver_str)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return ver_str, (major, minor) <= (0, 7)
        return ver_str, True
    except Exception:
        return "unknown", True


def _verify_gpu_actually_used(target_name: str) -> bool:
    if target_name not in ("nvidia", "nvidia-fp64"):
        return False
    try:
        result = cudaq.sample(_smoke_kernel, shots_count=16)
        return len(dict(result.items())) > 0
    except Exception as e:
        warnings.warn(f"[CUDAQ] GPU smoke test 失敗：{e}")
        return False


# ===========================================================================
# Backend 映射
# ===========================================================================

_CUDAQ_TARGET_MAP = {
    "cudaq_qpp":         "qpp-cpu",
    "qpp-cpu":           "qpp-cpu",
    "cudaq_nvidia":      "nvidia",
    "nvidia":            "nvidia",
    "cudaq_nvidia_fp64": "nvidia-fp64",
    "nvidia-fp64":       "nvidia-fp64",
    "tensornet":         "tensornet",
    "qiskit_aer":        "qpp-cpu",
}
_GPU_TARGETS = {"nvidia", "nvidia-fp64"}


def _set_target_safe(target_name: str) -> str:
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    if target_name in _GPU_TARGETS and not is_compat:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"[CUDAQ] CUDA-Q {ver_str} 不支援 V100 (sm_70)。\n"
            f"  請安裝：pip install cuda-quantum-cu11==0.7.1\n"
            f"{'='*60}"
        )
    try:
        cudaq.set_target(target_name)
    except Exception as e:
        raise RuntimeError(f"[CUDAQ] set_target('{target_name}') 失敗：{e}") from e

    if target_name in _GPU_TARGETS:
        if _verify_gpu_actually_used(target_name):
            print(f"[CUDAQ] GPU target '{target_name}' 驗證通過 ✓")
        else:
            warnings.warn(f"[CUDAQ] GPU smoke test 異常，可能在 CPU 執行。")
    return target_name


# ===========================================================================
# 90-bit bitstring 重建（具名暫存器策略）
# ===========================================================================

_N9_NAMED_REGS: list[str] = [
    'a1_0', 'a1_1',
    'a2_0', 'a2_1',
    'b21_0', 'b21_1',
    'a3_0', 'a3_1',
    'b31_0', 'b31_1', 'b32_0', 'b32_1',
    'a4_0', 'a4_1',
    'b41_0', 'b41_1', 'b42_0', 'b42_1', 'b43_0', 'b43_1',
    'a5_0', 'a5_1',
    'b51_0', 'b51_1', 'b52_0', 'b52_1',
    'b53_0', 'b53_1', 'b54_0', 'b54_1',
    'a6_0', 'a6_1',
    'b61_0', 'b61_1', 'b62_0', 'b62_1', 'b63_0', 'b63_1',
    'b64_0', 'b64_1', 'b65_0', 'b65_1',
    'a7_0', 'a7_1',
    'b71_0', 'b71_1', 'b72_0', 'b72_1', 'b73_0', 'b73_1',
    'b74_0', 'b74_1', 'b75_0', 'b75_1', 'b76_0', 'b76_1',
    'a8_0', 'a8_1',
    'b81_0', 'b81_1', 'b82_0', 'b82_1', 'b83_0', 'b83_1',
    'b84_0', 'b84_1', 'b85_0', 'b85_1', 'b86_0', 'b86_1',
    'b87_0', 'b87_1',
    'a9_0', 'a9_1',
]  # 共 74 個具名暫存器


def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    90-bit bitstring 重建：
      bits[ 0:74] — 74 個具名 mz() → get_sequential_data(reg)
      bits[74:90] — 16 個無名 mz()（鍵 9-{1..8}）→ __global__[4:20]
    """
    try:
        reg_data = {reg: result.get_sequential_data(reg) for reg in _N9_NAMED_REGS}
    except AttributeError:
        warnings.warn("[CUDAQ] get_sequential_data() 不存在，使用 items() fallback。")
        counts: dict[str, int] = {}
        for bs_raw, cnt in result.items():
            if len(bs_raw) == 90:
                counts[bs_raw] = counts.get(bs_raw, 0) + cnt
        return counts

    n_shots = len(reg_data['a1_0'])
    if n_shots == 0:
        warnings.warn("[CUDAQ] get_sequential_data('a1_0') 回傳空列表，射次為 0。")
        return {}

    try:
        global_data = result.get_sequential_data('__global__')
    except Exception:
        global_data = None

    counts: dict[str, int] = {}
    warned_global = False
    for i in range(n_shots):
        named_bits = ''.join(reg_data[reg][i] for reg in _N9_NAMED_REGS)

        if global_data and len(global_data) > i and len(global_data[i]) >= 20:
            bond9_bits = global_data[i][4:20]
        else:
            bond9_bits = '0' * 16
            if not warned_global:
                warnings.warn(
                    "[CUDAQ] __global__ 不可用，鍵 9-{1..8} 補零。"
                    "apply_bond_disconnection_correction 確保原子9至少有一鍵。"
                )
                warned_global = True

        bs = named_bits + bond9_bits
        if len(bs) != 90:
            raise RuntimeError(
                f"Shot {i}: bitstring 長度 {len(bs)} != 90"
            )
        counts[bs] = counts.get(bs, 0) + 1

    return counts


# ===========================================================================
# MoleculeGeneratorCUDAQ
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70 相容）。

    核心設計：每次 sample_molecule() 建立一個 closure kernel，
    weights 已 bake 進 kernel，cudaq.sample(k) 無參數，不觸發 broadcasting。
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
        if num_heavy_atom != 9:
            raise NotImplementedError(
                f"目前僅支援 num_heavy_atom=9（N={num_heavy_atom} 尚未實作）。"
            )

        self.num_heavy_atom            = num_heavy_atom
        self.all_weight_vector         = (
            np.array(all_weight_vector, dtype=np.float64)
            if all_weight_vector is not None else None
        )
        self.backend_name              = backend_name
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint
        self.expected_bits             = num_heavy_atom * (num_heavy_atom + 1)  # 90

        self.circuit_builder = DynamicCircuitBuilderCUDAQ(
            num_heavy_atom            = num_heavy_atom,
            temperature               = temperature,
            remove_bond_disconnection = remove_bond_disconnection,
            chemistry_constraint      = chemistry_constraint,
        )

        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size = num_heavy_atom,
            ncpus           = 1,
            sanitize_method = "strict",
        )

        actual_target       = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(actual_target)

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized.\n"
            f"  cudaq version : {ver_str}\n"
            f"  active target : {self._active_target}\n"
            f"  N atoms       : {num_heavy_atom}\n"
            f"  weight dim    : {self.circuit_builder.length_all_weight_vector}\n"
            f"  expected_bits : {self.expected_bits}\n"
            f"  design        : closure/capture (no-arg kernel)"
        )

    def update_weight_vector(
        self, all_weight_vector: Union[List[float], np.ndarray]
    ) -> None:
        self.all_weight_vector = np.array(all_weight_vector, dtype=np.float64)

    def sample_molecule(
        self,
        num_sample:  int,
        random_seed: int = 0,
    ) -> Tuple[dict, float, float]:
        assert self.all_weight_vector is not None, "請先呼叫 update_weight_vector()。"

        w = self.all_weight_vector
        assert len(w) == self.circuit_builder.length_all_weight_vector, (
            f"weight 長度不符：{len(w)} != "
            f"{self.circuit_builder.length_all_weight_vector}"
        )

        try:
            cudaq.set_random_seed(random_seed)
        except AttributeError:
            pass

        # ★ v7 核心：closure kernel，無參數，不觸發 broadcasting
        kernel = self.circuit_builder.build_kernel_from_weights(w)

        # cudaq.sample(kernel) — no args → __isBroadcast = False
        # → 正確的 conditionalOnMeasure shot-by-shot 路徑
        result = cudaq.sample(kernel, shots_count=num_sample)

        raw_counts = _reconstruct_bitstrings_n9(result)

        if not raw_counts:
            warnings.warn("[CUDAQ] raw_counts 為空，回傳 validity=0。")
            return {}, 0.0, 0.0

        # 解碼 bitstring → SMILES
        smiles_dict: dict[str, int] = {}
        num_valid_shots = 0

        for bs, count in raw_counts.items():
            bs_fixed      = self.circuit_builder.apply_bond_disconnection_correction(bs)
            quantum_state = self.data_generator.post_process_quantum_state(
                bs_fixed, reverse=False
            )
            smiles = self.data_generator.QuantumStateToSmiles(quantum_state)
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + count
            if smiles and smiles != "None":
                num_valid_shots += count

        validity = num_valid_shots / num_sample

        num_unique_valid = len([k for k in smiles_dict if k and k != "None"])
        uniqueness = (
            num_unique_valid / num_valid_shots if num_valid_shots > 0 else 0.0
        )

        return smiles_dict, validity, uniqueness


MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v7 closure/capture) ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q : {ver_str}  Volta compat: {'✓' if is_compat else '⚠ >=0.8'}")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    # ── Test 1：CPU ─────────────────────────────────────────────────────────
    print("\n[Test 1] qpp-cpu, 200 shots")
    gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
    t0  = time.time()
    smiles_dict, v, u = gen.sample_molecule(200)
    elapsed = time.time() - t0
    print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({elapsed:.1f}s)")
    valid_smiles = [k for k in smiles_dict if k and k != "None"]
    print(f"  有效分子數: {len(valid_smiles)}")
    print(f"  範例 SMILES: {valid_smiles[:5]}")
    if v > 0:
        print("  [Test 1] ✓ CPU 解碼正常，可啟動正式實驗")
    else:
        print("  [Test 1] ✗ validity=0")
        # Debug：測試 closure kernel 的 metadata
        k_test = gen.circuit_builder.build_kernel_from_weights(w)
        print(f"  [DEBUG] kernel type     = {type(k_test)}")
        print(f"  [DEBUG] kernel metadata = {k_test.metadata}")
        print(f"  [DEBUG] kernel arguments= {k_test.arguments}")

    # ── Test 2：GPU ─────────────────────────────────────────────────────────
    print("\n[Test 2] nvidia GPU, 500 shots")
    try:
        gen_gpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w,
                                         backend_name="cudaq_nvidia")
        t0 = time.time()
        smiles_dict_gpu, v_gpu, u_gpu = gen_gpu.sample_molecule(500)
        elapsed = time.time() - t0
        print(f"  V={v_gpu:.3f}  U={u_gpu:.3f}  V×U={v_gpu*u_gpu:.4f}  ({elapsed:.1f}s)")
        valid_smiles_gpu = [k for k in smiles_dict_gpu if k and k != "None"]
        print(f"  有效分子數: {len(valid_smiles_gpu)}")
        print(f"  範例 SMILES: {valid_smiles_gpu[:5]}")
        if v_gpu > 0:
            print("  [Test 2] ✓ GPU 解碼正常，可啟動正式實驗")
        else:
            print("  [Test 2] ✗ GPU validity=0")
    except Exception as e:
        print(f"  [Test 2] GPU 失敗：{e}")
        import traceback
        traceback.print_exc()