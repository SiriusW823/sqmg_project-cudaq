"""
==============================================================================
worker_eval.py — 子行程評估工作者（v10.2 backend 修正版）
==============================================================================

v10.1 → v10.2 修正：

  ★ [BUG-FIX 1] 預設 backend 由 cudaq_tensornet 改為 cudaq_nvidia：
      v10.1 中 --backend 預設值為 cudaq_tensornet。
      parent script（run_qpso_qmg_cudaq.py）呼叫時永遠顯式傳入
      --backend 參數，所以正常流程不受影響。
      但若直接執行 worker_eval.py 進行 debug / smoke test 而未指定
      backend，會靜默掛住（tensornet + dynamic circuit 不相容），
      造成診斷困難。修正：預設改為 cudaq_nvidia。

  ★ [BUG-FIX 2] 從 SUPPORTED_BACKENDS 移除 cudaq_tensornet 系列：
      README 與 docs 明確記載 tensornet 後端與含 mid-circuit
      measurement 的動態電路不相容（silent hang，非報錯），
      但 v10.1 的 SUPPORTED_BACKENDS 仍列出此選項，argparse choices
      不會阻止誤用。移除後，若誤傳 --backend cudaq_tensornet
      會立刻得到 argparse 錯誤，而非靜默掛住。

  v10.1 保留（不變）：
    - 雙重 chemistry constraint 防止（chemistry_constraint=False）
    - 預設失敗保護（np.save [0.0, 0.0] 先行）

說明：
  此檔案供 run_qpso_qmg_cudaq.py（v10.x subprocess 版）使用。
  若使用 run_qpso_qmg_mpi.py（MPI 版），不需要此檔案。

放置位置：worker_eval.py（專案根目錄）
==============================================================================
"""
import argparse
import sys
import os

import numpy as np

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


# ★ v10.2 修正：移除 cudaq_tensornet / cudaq_tensornet_mps
#   理由：tensornet 後端與動態電路（mid-circuit measurement）不相容，
#         會造成 silent hang。保留在此列表會讓 argparse choices 允許誤用。
#   若未來 cudaq 修正此問題，再重新加回。
SUPPORTED_BACKENDS = [
    "cudaq_nvidia",           # V100 cuStateVec GPU（推薦，預設）
    "cudaq_qpp",              # CPU 模擬（僅用於除錯）
    "cudaq_nvidia_fp64",      # cuStateVec FP64（較慢，高精度）
    "cudaq_nvidia_mqpu",      # multi-GPU shots（MPI 架構）
    "cudaq_nvidia_mgpu",      # multi-GPU statevec
]


def main():
    p = argparse.ArgumentParser(description="QMG worker_eval (v10.2)")
    p.add_argument("--weight_path",    type=str, required=True)
    p.add_argument("--result_path",    type=str, required=True)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=10000)
    # ★ v10.2 修正：預設由 cudaq_tensornet 改為 cudaq_nvidia
    p.add_argument("--backend",        type=str, default="cudaq_nvidia",
                   choices=SUPPORTED_BACKENDS)
    args = p.parse_args()

    # 預設失敗輸出（確保 result_path 在任何錯誤情況下都存在）
    np.save(args.result_path, np.array([0.0, 0.0], dtype=np.float64))

    try:
        from qmg.generator_cudaq import MoleculeGeneratorCUDAQ

        # ★ v10.1 保留（不變）：
        #   主行程（run_qpso_qmg_cudaq.py）在儲存前已呼叫：
        #     w_constrained = cwg.apply_chemistry_constraint(pos.copy())
        #     np.save(weight_path, w_constrained)
        #   因此此處直接載入，不再重複套用 chemistry constraint。
        #
        #   正確流程（v10.1+）：
        #     w = np.load(...)                               # 已 constrained
        #     gen = MoleculeGeneratorCUDAQ(..., chemistry_constraint=False)
        w = np.load(args.weight_path)
        assert len(w) == 134, f"weight 長度錯誤：{len(w)}，期待 134"

        gen = MoleculeGeneratorCUDAQ(
            num_heavy_atom            = args.num_heavy_atom,
            all_weight_vector         = w,
            backend_name              = args.backend,
            remove_bond_disconnection = True,
            # ★ v10.1 關鍵修正：設為 False，避免 generator 內部再套用一次
            #   主行程的 evaluate_fn 已套用，此處不需要重複
            chemistry_constraint      = False,
        )

        _, validity, uniqueness = gen.sample_molecule(args.num_sample)

        np.save(args.result_path, np.array([validity, uniqueness], dtype=np.float64))
        sys.exit(0)

    except Exception as e:
        print(f"[worker_eval] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()