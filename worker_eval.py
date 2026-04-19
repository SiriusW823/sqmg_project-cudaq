"""
==============================================================================
worker_eval.py — 子行程評估工作者
==============================================================================
由 run_qpso_qmg_cudaq.py 的 evaluate_fn 透過 subprocess.run() 呼叫。

每次呼叫都在獨立的 Python 行程中執行，行程結束時 CUDA context 被
kernel 強制銷毀，所有 cuStateVec pinned memory 完全釋放。

這是解決 CUDA-Q 0.7.1 pinned memory 洩漏的唯一有效方案：
  - glibc malloc_trim：無效（pinned memory 不走 glibc heap）
  - del generator + gc.collect()：無效（CUDA context 仍存在）
  - generator 重建：無效（CUDA context 在同一行程中累積）
  - subprocess 隔離：✓ 行程結束 → CUDA driver 強制釋放所有資源

使用方式（由 run_qpso_qmg_cudaq.py 內部呼叫，不直接執行）：
  CUDA_VISIBLE_DEVICES=0 python worker_eval.py \
      --weight_path /tmp/w_xyz.npy \
      --num_heavy_atom 9 \
      --num_sample 10000 \
      --backend cudaq_nvidia \
      --result_path /tmp/result_xyz.npy

輸出：
  result_path.npy 包含 [validity, uniqueness] 兩個 float64
  若執行失敗，寫入 [0.0, 0.0] 並以 exit code 1 結束
==============================================================================
"""
import argparse
import sys
import os

import numpy as np

# 靜音 RDKit
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight_path",    type=str, required=True,
                   help="輸入 weight vector 的 .npy 檔路徑（shape=(134,)）")
    p.add_argument("--result_path",    type=str, required=True,
                   help="輸出 [validity, uniqueness] 的 .npy 檔路徑")
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=10000)
    p.add_argument("--backend",        type=str, default="cudaq_nvidia")
    args = p.parse_args()

    # 預設失敗輸出
    np.save(args.result_path, np.array([0.0, 0.0], dtype=np.float64))

    try:
        from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
        from qmg.utils.weight_generator import ConditionalWeightsGenerator

        # 載入 weight vector
        w = np.load(args.weight_path)
        assert len(w) == 134, f"weight 長度錯誤：{len(w)}"

        # 套用化學約束
        cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=None)
        w_constrained = cwg.apply_chemistry_constraint(w.copy())

        # 建立 generator 並採樣
        gen = MoleculeGeneratorCUDAQ(
            num_heavy_atom            = args.num_heavy_atom,
            all_weight_vector         = w_constrained,
            backend_name              = args.backend,
            remove_bond_disconnection = True,
            chemistry_constraint      = True,
        )

        _, validity, uniqueness = gen.sample_molecule(args.num_sample)

        # 寫入結果
        np.save(args.result_path, np.array([validity, uniqueness], dtype=np.float64))
        sys.exit(0)

    except Exception as e:
        # 失敗時印出訊息（會被主行程的 stderr 捕捉）
        print(f"[worker_eval] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # result_path 已在最開頭寫入 [0.0, 0.0]
        sys.exit(1)


if __name__ == "__main__":
    main()
