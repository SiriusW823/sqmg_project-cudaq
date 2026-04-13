"""
==============================================================================
build_dynamic_circuit_cudaq.py  (v8)
CUDA-Q 0.7.1 — Closure/Capture + 記憶體管理
==============================================================================

v7 問題：
  每次 make_qmg_n9_kernel() 建立新的 @cudaq.kernel，
  MLIR module 累積在記憶體中不會被 GC 釋放，
  155 次後 OOM → Killed。

v8 修正：
  build_kernel_from_weights() 每次回傳新 kernel 後，
  呼叫方負責 del kernel + gc.collect() 釋放 MLIR 記憶體。
  generator 的 sample_molecule() 已加入正確的清理邏輯。

效能說明：
  qpp-cpu：supportsConditionalFeedback=False
    → Python shot-by-shot loop，10000 shots = 10000 次 kernel 呼叫
    → ~90s/eval，10000 shots，太慢

  nvidia GPU：supportsConditionalFeedback=True
    → cuStateVec 原生處理 conditional，一次呼叫完成所有 shots
    → 預計 1-5s/eval（速度提升 20~90x）

  建議：使用 --backend cudaq_nvidia
==============================================================================
"""
from __future__ import annotations

import math
import gc
import numpy as np
from typing import Union, List

import cudaq


def make_qmg_n9_kernel(weights: list):
    """
    工廠函式：建立捕捉 weights 的 N=9 QMG 動態電路 kernel（無參數）。

    使用後請呼叫 del kernel; gc.collect() 釋放 MLIR 記憶體。

    Args:
        weights: 長度 134 的 Python float list

    Returns:
        無參數的 @cudaq.kernel
    """
    assert len(weights) == 134, f"weights 長度 {len(weights)} != 134"
    w = [float(x) for x in weights]

    @cudaq.kernel
    def _qmg_n9():
        q = cudaq.qvector(20)

        # ================================================================
        # Phase 1: build_two_atoms   w[0:8]
        # ================================================================
        ry(math.pi * w[0], q[0])
        x(q[1])
        ry(math.pi * w[2], q[2])
        ry(math.pi * w[4], q[3])
        x.ctrl(q[0], q[1])
        ry.ctrl(math.pi * w[3], q[1], q[2])
        x.ctrl(q[2], q[3])
        ry.ctrl(math.pi * w[1], q[0], q[1])
        x.ctrl(q[1], q[2])
        ry.ctrl(math.pi * w[5], q[2], q[3])

        a1_0 = mz(q[0])
        a1_1 = mz(q[1])
        a2_0 = mz(q[2])
        a2_1 = mz(q[3])

        if a2_0 or a2_1:
            ry(math.pi * w[6], q[4])
            x(q[5])
            x.ctrl(q[4], q[5])
            ry.ctrl(math.pi * w[7], q[4], q[5])
        b21_0 = mz(q[4])
        b21_1 = mz(q[5])

        # ================================================================
        # Phase 2: atom 3   w[8:17]
        # ================================================================
        if a2_0:
            x(q[2])
        if a2_1:
            x(q[3])
        if b21_0:
            x(q[4])
        if b21_1:
            x(q[5])

        if a2_0 or a2_1:
            ry(math.pi * w[8],  q[2])
            ry(math.pi * w[9],  q[3])
            ry.ctrl(math.pi * w[10], q[2], q[3])
        a3_0 = mz(q[2])
        a3_1 = mz(q[3])

        if a3_0 or a3_1:
            ry(math.pi * w[11], q[5])
            ry.ctrl(math.pi * w[13], q[5], q[4])
            ry.ctrl(math.pi * w[14], q[4], q[5])
            ry(math.pi * w[12], q[7])
            ry.ctrl(math.pi * w[15], q[7], q[6])
            ry.ctrl(math.pi * w[16], q[6], q[7])
        b31_0 = mz(q[4])
        b31_1 = mz(q[5])
        b32_0 = mz(q[6])
        b32_1 = mz(q[7])

        # ================================================================
        # Phase 3: atom 4   w[17:29]
        # ================================================================
        if a3_0:
            x(q[2])
        if a3_1:
            x(q[3])
        if b31_0:
            x(q[4])
        if b31_1:
            x(q[5])
        if b32_0:
            x(q[6])
        if b32_1:
            x(q[7])

        if a3_0 or a3_1:
            ry(math.pi * w[17], q[2])
            ry(math.pi * w[18], q[3])
            ry.ctrl(math.pi * w[19], q[2], q[3])
        a4_0 = mz(q[2])
        a4_1 = mz(q[3])

        if a4_0 or a4_1:
            ry(math.pi * w[20], q[5])
            ry.ctrl(math.pi * w[23], q[5], q[4])
            ry.ctrl(math.pi * w[24], q[4], q[5])
            ry(math.pi * w[21], q[7])
            ry.ctrl(math.pi * w[25], q[7], q[6])
            ry.ctrl(math.pi * w[26], q[6], q[7])
            ry(math.pi * w[22], q[9])
            ry.ctrl(math.pi * w[27], q[9], q[8])
            ry.ctrl(math.pi * w[28], q[8], q[9])
        b41_0 = mz(q[4])
        b41_1 = mz(q[5])
        b42_0 = mz(q[6])
        b42_1 = mz(q[7])
        b43_0 = mz(q[8])
        b43_1 = mz(q[9])

        # ================================================================
        # Phase 4: atom 5   w[29:44]
        # ================================================================
        if a4_0:
            x(q[2])
        if a4_1:
            x(q[3])
        if b41_0:
            x(q[4])
        if b41_1:
            x(q[5])
        if b42_0:
            x(q[6])
        if b42_1:
            x(q[7])
        if b43_0:
            x(q[8])
        if b43_1:
            x(q[9])

        if a4_0 or a4_1:
            ry(math.pi * w[29], q[2])
            ry(math.pi * w[30], q[3])
            ry.ctrl(math.pi * w[31], q[2], q[3])
        a5_0 = mz(q[2])
        a5_1 = mz(q[3])

        if a5_0 or a5_1:
            ry(math.pi * w[32], q[5])
            ry.ctrl(math.pi * w[36], q[5], q[4])
            ry.ctrl(math.pi * w[37], q[4], q[5])
            ry(math.pi * w[33], q[7])
            ry.ctrl(math.pi * w[38], q[7], q[6])
            ry.ctrl(math.pi * w[39], q[6], q[7])
            ry(math.pi * w[34], q[9])
            ry.ctrl(math.pi * w[40], q[9], q[8])
            ry.ctrl(math.pi * w[41], q[8], q[9])
            ry(math.pi * w[35], q[11])
            ry.ctrl(math.pi * w[42], q[11], q[10])
            ry.ctrl(math.pi * w[43], q[10], q[11])
        b51_0 = mz(q[4])
        b51_1 = mz(q[5])
        b52_0 = mz(q[6])
        b52_1 = mz(q[7])
        b53_0 = mz(q[8])
        b53_1 = mz(q[9])
        b54_0 = mz(q[10])
        b54_1 = mz(q[11])

        # ================================================================
        # Phase 5: atom 6   w[44:62]
        # ================================================================
        if a5_0:
            x(q[2])
        if a5_1:
            x(q[3])
        if b51_0:
            x(q[4])
        if b51_1:
            x(q[5])
        if b52_0:
            x(q[6])
        if b52_1:
            x(q[7])
        if b53_0:
            x(q[8])
        if b53_1:
            x(q[9])
        if b54_0:
            x(q[10])
        if b54_1:
            x(q[11])

        if a5_0 or a5_1:
            ry(math.pi * w[44], q[2])
            ry(math.pi * w[45], q[3])
            ry.ctrl(math.pi * w[46], q[2], q[3])
        a6_0 = mz(q[2])
        a6_1 = mz(q[3])

        if a6_0 or a6_1:
            ry(math.pi * w[47], q[5])
            ry.ctrl(math.pi * w[52], q[5], q[4])
            ry.ctrl(math.pi * w[53], q[4], q[5])
            ry(math.pi * w[48], q[7])
            ry.ctrl(math.pi * w[54], q[7], q[6])
            ry.ctrl(math.pi * w[55], q[6], q[7])
            ry(math.pi * w[49], q[9])
            ry.ctrl(math.pi * w[56], q[9], q[8])
            ry.ctrl(math.pi * w[57], q[8], q[9])
            ry(math.pi * w[50], q[11])
            ry.ctrl(math.pi * w[58], q[11], q[10])
            ry.ctrl(math.pi * w[59], q[10], q[11])
            ry(math.pi * w[51], q[13])
            ry.ctrl(math.pi * w[60], q[13], q[12])
            ry.ctrl(math.pi * w[61], q[12], q[13])
        b61_0 = mz(q[4])
        b61_1 = mz(q[5])
        b62_0 = mz(q[6])
        b62_1 = mz(q[7])
        b63_0 = mz(q[8])
        b63_1 = mz(q[9])
        b64_0 = mz(q[10])
        b64_1 = mz(q[11])
        b65_0 = mz(q[12])
        b65_1 = mz(q[13])

        # ================================================================
        # Phase 6: atom 7   w[62:83]
        # ================================================================
        if a6_0:
            x(q[2])
        if a6_1:
            x(q[3])
        if b61_0:
            x(q[4])
        if b61_1:
            x(q[5])
        if b62_0:
            x(q[6])
        if b62_1:
            x(q[7])
        if b63_0:
            x(q[8])
        if b63_1:
            x(q[9])
        if b64_0:
            x(q[10])
        if b64_1:
            x(q[11])
        if b65_0:
            x(q[12])
        if b65_1:
            x(q[13])

        if a6_0 or a6_1:
            ry(math.pi * w[62], q[2])
            ry(math.pi * w[63], q[3])
            ry.ctrl(math.pi * w[64], q[2], q[3])
        a7_0 = mz(q[2])
        a7_1 = mz(q[3])

        if a7_0 or a7_1:
            ry(math.pi * w[65], q[5])
            ry.ctrl(math.pi * w[71], q[5], q[4])
            ry.ctrl(math.pi * w[72], q[4], q[5])
            ry(math.pi * w[66], q[7])
            ry.ctrl(math.pi * w[73], q[7], q[6])
            ry.ctrl(math.pi * w[74], q[6], q[7])
            ry(math.pi * w[67], q[9])
            ry.ctrl(math.pi * w[75], q[9], q[8])
            ry.ctrl(math.pi * w[76], q[8], q[9])
            ry(math.pi * w[68], q[11])
            ry.ctrl(math.pi * w[77], q[11], q[10])
            ry.ctrl(math.pi * w[78], q[10], q[11])
            ry(math.pi * w[69], q[13])
            ry.ctrl(math.pi * w[79], q[13], q[12])
            ry.ctrl(math.pi * w[80], q[12], q[13])
            ry(math.pi * w[70], q[15])
            ry.ctrl(math.pi * w[81], q[15], q[14])
            ry.ctrl(math.pi * w[82], q[14], q[15])
        b71_0 = mz(q[4])
        b71_1 = mz(q[5])
        b72_0 = mz(q[6])
        b72_1 = mz(q[7])
        b73_0 = mz(q[8])
        b73_1 = mz(q[9])
        b74_0 = mz(q[10])
        b74_1 = mz(q[11])
        b75_0 = mz(q[12])
        b75_1 = mz(q[13])
        b76_0 = mz(q[14])
        b76_1 = mz(q[15])

        # ================================================================
        # Phase 7: atom 8   w[83:107]
        # ================================================================
        if a7_0:
            x(q[2])
        if a7_1:
            x(q[3])
        if b71_0:
            x(q[4])
        if b71_1:
            x(q[5])
        if b72_0:
            x(q[6])
        if b72_1:
            x(q[7])
        if b73_0:
            x(q[8])
        if b73_1:
            x(q[9])
        if b74_0:
            x(q[10])
        if b74_1:
            x(q[11])
        if b75_0:
            x(q[12])
        if b75_1:
            x(q[13])
        if b76_0:
            x(q[14])
        if b76_1:
            x(q[15])

        if a7_0 or a7_1:
            ry(math.pi * w[83], q[2])
            ry(math.pi * w[84], q[3])
            ry.ctrl(math.pi * w[85], q[2], q[3])
        a8_0 = mz(q[2])
        a8_1 = mz(q[3])

        if a8_0 or a8_1:
            ry(math.pi * w[86], q[5])
            ry.ctrl(math.pi * w[93], q[5], q[4])
            ry.ctrl(math.pi * w[94], q[4], q[5])
            ry(math.pi * w[87], q[7])
            ry.ctrl(math.pi * w[95], q[7], q[6])
            ry.ctrl(math.pi * w[96], q[6], q[7])
            ry(math.pi * w[88], q[9])
            ry.ctrl(math.pi * w[97], q[9], q[8])
            ry.ctrl(math.pi * w[98], q[8], q[9])
            ry(math.pi * w[89], q[11])
            ry.ctrl(math.pi * w[99], q[11], q[10])
            ry.ctrl(math.pi * w[100], q[10], q[11])
            ry(math.pi * w[90], q[13])
            ry.ctrl(math.pi * w[101], q[13], q[12])
            ry.ctrl(math.pi * w[102], q[12], q[13])
            ry(math.pi * w[91], q[15])
            ry.ctrl(math.pi * w[103], q[15], q[14])
            ry.ctrl(math.pi * w[104], q[14], q[15])
            ry(math.pi * w[92], q[17])
            ry.ctrl(math.pi * w[105], q[17], q[16])
            ry.ctrl(math.pi * w[106], q[16], q[17])
        b81_0 = mz(q[4])
        b81_1 = mz(q[5])
        b82_0 = mz(q[6])
        b82_1 = mz(q[7])
        b83_0 = mz(q[8])
        b83_1 = mz(q[9])
        b84_0 = mz(q[10])
        b84_1 = mz(q[11])
        b85_0 = mz(q[12])
        b85_1 = mz(q[13])
        b86_0 = mz(q[14])
        b86_1 = mz(q[15])
        b87_0 = mz(q[16])
        b87_1 = mz(q[17])

        # ================================================================
        # Phase 8: atom 9   w[107:134]
        # ================================================================
        if a8_0:
            x(q[2])
        if a8_1:
            x(q[3])
        if b81_0:
            x(q[4])
        if b81_1:
            x(q[5])
        if b82_0:
            x(q[6])
        if b82_1:
            x(q[7])
        if b83_0:
            x(q[8])
        if b83_1:
            x(q[9])
        if b84_0:
            x(q[10])
        if b84_1:
            x(q[11])
        if b85_0:
            x(q[12])
        if b85_1:
            x(q[13])
        if b86_0:
            x(q[14])
        if b86_1:
            x(q[15])
        if b87_0:
            x(q[16])
        if b87_1:
            x(q[17])

        if a8_0 or a8_1:
            ry(math.pi * w[107], q[2])
            ry(math.pi * w[108], q[3])
            ry.ctrl(math.pi * w[109], q[2], q[3])

        a9_0 = mz(q[2])
        a9_1 = mz(q[3])

        if a9_0 or a9_1:
            ry(math.pi * w[110], q[5])
            ry.ctrl(math.pi * w[118], q[5], q[4])
            ry.ctrl(math.pi * w[119], q[4], q[5])
            ry(math.pi * w[111], q[7])
            ry.ctrl(math.pi * w[120], q[7], q[6])
            ry.ctrl(math.pi * w[121], q[6], q[7])
            ry(math.pi * w[112], q[9])
            ry.ctrl(math.pi * w[122], q[9], q[8])
            ry.ctrl(math.pi * w[123], q[8], q[9])
            ry(math.pi * w[113], q[11])
            ry.ctrl(math.pi * w[124], q[11], q[10])
            ry.ctrl(math.pi * w[125], q[10], q[11])
            ry(math.pi * w[114], q[13])
            ry.ctrl(math.pi * w[126], q[13], q[12])
            ry.ctrl(math.pi * w[127], q[12], q[13])
            ry(math.pi * w[115], q[15])
            ry.ctrl(math.pi * w[128], q[15], q[14])
            ry.ctrl(math.pi * w[129], q[14], q[15])
            ry(math.pi * w[116], q[17])
            ry.ctrl(math.pi * w[130], q[17], q[16])
            ry.ctrl(math.pi * w[131], q[16], q[17])
            ry(math.pi * w[117], q[19])
            ry.ctrl(math.pi * w[132], q[19], q[18])
            ry.ctrl(math.pi * w[133], q[18], q[19])

        mz(q[4]);  mz(q[5])
        mz(q[6]);  mz(q[7])
        mz(q[8]);  mz(q[9])
        mz(q[10]); mz(q[11])
        mz(q[12]); mz(q[13])
        mz(q[14]); mz(q[15])
        mz(q[16]); mz(q[17])
        mz(q[18]); mz(q[19])

    return _qmg_n9


# ===========================================================================
# DynamicCircuitBuilderCUDAQ
# ===========================================================================

class DynamicCircuitBuilderCUDAQ:
    """CUDA-Q 0.7.1 版 QMG 動態電路建構器（Closure/Capture 模式）。"""

    def __init__(
        self,
        num_heavy_atom:            int,
        temperature:               float = 0.2,
        remove_bond_disconnection: bool  = True,
        chemistry_constraint:      bool  = True,
    ):
        if num_heavy_atom != 9:
            raise NotImplementedError(f"目前僅支援 N=9（N={num_heavy_atom} 尚未實作）。")
        self.num_heavy_atom            = num_heavy_atom
        self.temperature               = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint
        self.num_clbits                = num_heavy_atom * (num_heavy_atom + 1)  # 90
        self.length_all_weight_vector  = int(
            8 + (num_heavy_atom - 2) * (num_heavy_atom + 3) * 3 / 2
        )  # 134

    def build_kernel_from_weights(self, weights) -> "cudaq.PyKernelDecorator":
        """
        建立捕捉 weights 的無參數 closure kernel。

        使用後請呼叫:
            del kernel
            gc.collect()
        釋放 MLIR 記憶體，避免 OOM。
        """
        w_list = [float(x) for x in (weights.tolist() if hasattr(weights, 'tolist') else weights)]
        return make_qmg_n9_kernel(w_list)

    def apply_bond_disconnection_correction(self, bitstring: str) -> str:
        if not self.remove_bond_disconnection:
            return bitstring
        n    = self.num_heavy_atom
        bits = list(bitstring)
        for k in range(3, n + 1):
            atom_start  = (k - 1) ** 2 + (k - 1)
            atom_exists = bits[atom_start] == '1' or bits[atom_start + 1] == '1'
            if not atom_exists:
                continue
            bond_start = k * k - k + 2
            bond_end   = bond_start + 2 * (k - 1)
            if all(b == '0' for b in bits[bond_start:bond_end]):
                bits[bond_end - 1] = '1'
        return ''.join(bits)