#!/usr/bin/env python3
"""
==============================================================================
persistent_worker.py — 常駐評估 worker（v12.1）
==============================================================================

為什麼需要它
------------
原本的 `worker_eval.py` 是「一次評估一個行程」：每次評估都要重新
    Python 啟動 → import cudaq → 驗證 GPU target → 建 MoleculeGeneratorCUDAQ
實測（2026-07-31，1000 shots、N=9）：
    整體每次評估 33.8 s，但 generator 建好後單次 sample_molecule 只要 10.0 s
→ **約 24 秒（71%）是純粹的行程啟動開銷。**

這支常駐 worker 只做一次初始化，然後在 stdin/stdout 上跑一個任務迴圈，
把那 24 秒攤提掉。

協定（極簡，行為單位）
----------------------
父行程 → worker（stdin，每行一個 JSON）：
    {"w": "<weight .npy 路徑>", "r": "<result .npy 路徑>", "seed": 0}
worker → 父行程（stdout，每行一個 JSON）：
    {"ok": true,  "r": "<result 路徑>"}
    {"ok": false, "err": "<訊息>"}
stdin 關閉即結束。

結果檔格式與 worker_eval.py 完全相同：[validity, uniqueness, HBA, HBD]，
因此下游解析不需要任何改動。

關於 random_seed
----------------
`sample_molecule(num_sample, random_seed=s)` 內部會呼叫 `cudaq.set_random_seed(s)`，
**預設 s=0**——這就是為什麼過去每次評估都給出完全相同的結果。
本 worker 把 seed 透過協定傳入：
  - 最適化時傳 0（維持與歷史資料一致的確定性目標函數）
  - 重新評估最佳解時傳不同值（才能量出真正的取樣不確定性）

放置位置：persistent_worker.py（專案根目錄）
==============================================================================
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=5000)
    p.add_argument("--backend",        type=str, default="cudaq_nvidia")
    p.add_argument("--report_hbahbd",  action="store_true", default=False)
    args = p.parse_args()

    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    # ── 一次性初始化（這是我們要攤提掉的成本）────────────────────────────
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
    from qmg.utils.weight_generator import ConditionalWeightsGenerator

    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=None,
                                      disable_connectivity_position=[])
    dummy = cwg.apply_chemistry_constraint(
        cwg.generate_conditional_random_weights(random_seed=0))

    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        all_weight_vector         = dummy,
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = False,   # 權重在父行程已套用
    )

    if args.report_hbahbd:
        from worker_eval import compute_mean_hba_hbd
    else:
        compute_mean_hba_hbd = None

    print(json.dumps({"ready": True}), flush=True)

    # ── 任務迴圈 ─────────────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "err": f"bad json: {e}"}), flush=True)
            continue

        rpath = task.get("r", "")
        try:
            w = np.load(task["w"])
            assert len(w) == gen.length_all_weight_vector, \
                f"weight 長度 {len(w)} != {gen.length_all_weight_vector}"

            gen.update_weight_vector(w)
            smiles_dict, validity, uniqueness = gen.sample_molecule(
                args.num_sample, random_seed=int(task.get("seed", 0)))

            hba = hbd = 0.0
            if compute_mean_hba_hbd is not None:
                hba, hbd = compute_mean_hba_hbd(smiles_dict)

            np.save(rpath, np.array([validity, uniqueness, hba, hbd],
                                    dtype=np.float64))
            print(json.dumps({"ok": True, "r": rpath}), flush=True)

        except Exception as e:                            # noqa: BLE001
            # 失敗時仍寫出 0 結果檔，維持與 worker_eval.py 相同的容錯語意
            try:
                np.save(rpath, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
            except Exception:
                pass
            print(json.dumps({"ok": False, "err": str(e)[:300]}), flush=True)


if __name__ == "__main__":
    main()
