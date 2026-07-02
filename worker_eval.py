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


# ===========================================================================
# HBA / HBD 計測工具（v10.4 新增，純量測，不影響 V×U 最適化）
# ===========================================================================
#   目的：對齊 qiskit 參考 log（chemistry_constraint 4HBA/3HBD）的量測定義，
#         在完全不改動 QPSO 演算法與 V×U 目標的前提下，額外回報生成分子群的
#         平均 HBA（氫鍵受體數）與 HBD（氫鍵供體數）。
#
#   定義：使用 RDKit Lipinski.NumHAcceptors / NumHDonors，
#         此為 QMG / Chen 等分子生成論文最常用的 HBA/HBD 定義。
#         若參考實作使用 rdMolDescriptors.CalcNumHBA/CalcNumHBD，
#         只需切換下方 _HBA_FN / _HBD_FN 兩行（見註解）。
#
#   統計基礎：對「有效分子」以出現次數加權求平均（distribution-learning 均值），
#            與 validity / uniqueness 的採樣基礎一致，因此 qiskit log 的
#            「HBA (close to 4)」/「HBD (close to 3)」語意可直接對比。
# ===========================================================================

def compute_mean_hba_hbd(smiles_dict: dict):
    """
    從 sample_molecule 回傳的 smiles_dict（{SMILES: count}）計算
    加權平均 HBA / HBD（僅計入可解析的有效分子）。

    回傳：(hba_mean, hbd_mean)；若無有效分子回傳 (0.0, 0.0)。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ImportError:
        return 0.0, 0.0

    # ── HBA/HBD 定義（如需對齊其他參考實作，改這兩行即可）──
    _HBA_FN = Lipinski.NumHAcceptors    # 或 rdMolDescriptors.CalcNumHBA
    _HBD_FN = Lipinski.NumHDonors       # 或 rdMolDescriptors.CalcNumHBD

    total = 0
    hba_sum = 0.0
    hbd_sum = 0.0
    for smi, cnt in smiles_dict.items():
        if smi is None or smi == "None":
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        total += cnt
        hba_sum += float(_HBA_FN(mol)) * cnt
        hbd_sum += float(_HBD_FN(mol)) * cnt

    if total == 0:
        return 0.0, 0.0
    return hba_sum / total, hbd_sum / total


def main():
    p = argparse.ArgumentParser(description="QMG worker_eval (v10.2)")
    p.add_argument("--weight_path",    type=str, required=True)
    p.add_argument("--result_path",    type=str, required=True)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=10000)
    # ★ v10.2 修正：預設由 cudaq_tensornet 改為 cudaq_nvidia
    p.add_argument("--backend",        type=str, default="cudaq_nvidia",
                   choices=SUPPORTED_BACKENDS)
    # ★ v10.4 新增：opt-in 的 HBA/HBD 量測旗標。
    #   未指定時，行為與舊版完全相同（HBA/HBD 欄位輸出 0.0，不做 rdkit 計算，
    #   不增加任何執行時間）。指定時才對 smiles_dict 計算平均 HBA/HBD。
    p.add_argument("--report_hbahbd", action="store_true", default=False,
                   help="額外計算並輸出生成分子群的平均 HBA / HBD（純量測，"
                        "不影響 V×U 目標）。")
    args = p.parse_args()

    # 預設失敗輸出（確保 result_path 在任何錯誤情況下都存在）
    # ★ v10.4：欄位擴為 4 個 [validity, uniqueness, HBA, HBD]，向後相容
    #   （父行程只讀取 arr[0], arr[1] 作為 (V, U)；arr[2], arr[3] 為量測欄位）。
    np.save(args.result_path, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))

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

        smiles_dict, validity, uniqueness = gen.sample_molecule(args.num_sample)

        # ★ v10.4：opt-in HBA/HBD 量測（不影響 V×U 目標與 QPSO 演算法）
        hba_mean, hbd_mean = (0.0, 0.0)
        if args.report_hbahbd:
            hba_mean, hbd_mean = compute_mean_hba_hbd(smiles_dict)

        np.save(
            args.result_path,
            np.array([validity, uniqueness, hba_mean, hbd_mean], dtype=np.float64),
        )
        sys.exit(0)

    except Exception as e:
        print(f"[worker_eval] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()