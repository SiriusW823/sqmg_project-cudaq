#!/usr/bin/env python3
"""
==============================================================================
benchmark/shot_seed_test.py — 取樣種子敏感度測試（v12.0）
==============================================================================

背景
----
`revalidate.py` 發現：同一組 best_params 重複評估 24 次，V×U 的標準差是 **0.0000**。
評估是確定性的——CUDA-Q 預設的取樣亂數種子固定，每次 `cudaq.sample()`
都跑出同一組 shots。

這件事的意義
------------
好消息：報告值沒有 winner's curse（沒有評估雜訊就沒有選擇偏差），數字可完全重現。

壞消息：整個最佳化其實是在對**一條固定的亂數流**做最佳化。
report 的 V×U 是「這組參數 × 這條 shot 序列」的值，而不是
「這組參數的期望表現」。若換一條 shot 序列數字就掉，那就代表最佳化器
（部分地）過擬合到那條特定亂數流——這正是「剛好的特定情況」的另一種形式。

本測試
------
固定參數向量，只改 `cudaq.set_random_seed(s)`，跑 S 個不同的取樣種子。
得到的分佈就是該組參數的**真實表現分佈**：
  - 平均值 = 期望 V×U（論文該報的數字）
  - 標準差 = 取樣不確定性
  - 原報告值在此分佈中的位置 = 它有多「剛好」

用法：
    python benchmark/shot_seed_test.py \\
        --params results_qpso_aeqts/unconditional_9_qpso_aeqts_M64T150_best_params.npy \\
        --seeds 20 --num_sample 5000
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--seeds", type=int, default=20, help="取樣種子數量")
    p.add_argument("--num_sample", type=int, default=5000)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--backend", default="cudaq_nvidia")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import cudaq
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
    from qmg.utils import ConditionalWeightsGenerator

    path = args.params if os.path.isabs(args.params) else os.path.join(REPO, args.params)
    w_raw = np.load(path)

    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=None,
                                      disable_connectivity_position=[])
    w = cwg.apply_chemistry_constraint(w_raw.copy())

    print("=" * 78)
    print(f"取樣種子敏感度測試：{args.params}")
    print(f"  {args.seeds} 個取樣種子 × {args.num_sample} shots")
    print("=" * 78)

    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom=args.num_heavy_atom,
        all_weight_vector=w,
        backend_name=args.backend,
        remove_bond_disconnection=True,
        chemistry_constraint=False,     # 已在上面套用
    )

    # ★ v12.1 修正：種子必須「傳給 sample_molecule」，不能在外面呼叫
    #   cudaq.set_random_seed()——因為 sample_molecule 內部第一件事就是
    #       cudaq.set_random_seed(random_seed)   # 預設 0
    #   會把外面設的種子直接覆蓋掉。先前版本正是踩到這點，導致 20 個「不同種子」
    #   全部跑出一模一樣的結果，錯誤地推論出「評估是確定性的、沒有 winner's curse」。
    rows = []
    for s in range(args.seeds):
        _, v, u = gen.sample_molecule(args.num_sample, random_seed=s)
        rows.append((s, float(v), float(u), float(v) * float(u)))
        print(f"  seed={s:3d}  V={v:.4f}  U={u:.4f}  V×U={v*u:.4f}", flush=True)

    vu = np.array([r[3] for r in rows])
    V  = np.array([r[1] for r in rows])
    U  = np.array([r[2] for r in rows])

    mean, sd = vu.mean(), vu.std(ddof=1)
    se = sd / np.sqrt(len(vu))

    print("\n" + "=" * 78)
    print(f"V×U  平均 = {mean:.4f}   標準差 = {sd:.4f}")
    print(f"     95% CI（平均值）= [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}]")
    print(f"     範圍 = [{vu.min():.4f}, {vu.max():.4f}]")
    print(f"V    平均 = {V.mean():.4f} ± {V.std(ddof=1):.4f}")
    print(f"U    平均 = {U.mean():.4f} ± {U.std(ddof=1):.4f}")

    # seed=0 就是最適化當時實際看到的那條 shot 序列，也就是論文報的那個數字。
    vu0 = vu[0]
    if sd > 0:
        z = (vu0 - mean) / sd
        pct = (vu < vu0).mean() * 100
        print(f"\n[winner's curse 檢查]")
        print(f"  最適化時使用的 seed=0 值 : {vu0:.4f}")
        print(f"  跨 {len(vu)} 個取樣種子的期望值 : {mean:.4f}  ({vu0-mean:+.4f})")
        print(f"  seed=0 的 z-score        : {z:+.2f}   (百分位 {pct:.0f}%)")
        if vu0 > mean + sd:
            print("  → seed=0 明顯偏高：報告值含選擇偏差，論文應改報期望值 ± CI。")
        elif vu0 < mean - sd:
            print("  → seed=0 偏低：報告值反而低估了這組參數。")
        else:
            print("  → seed=0 落在一個標準差內：報告值可視為該參數的代表性表現。")
    print("=" * 78)

    out = args.out or os.path.join(
        REPO, "results_revalidate",
        f"shotseed_{os.path.basename(os.path.dirname(args.params))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "params": args.params, "n_seeds": args.seeds,
            "num_sample": args.num_sample,
            "vu_mean": float(mean), "vu_std": float(sd),
            "vu_min": float(vu.min()), "vu_max": float(vu.max()),
            "v_mean": float(V.mean()), "u_mean": float(U.mean()),
            "per_seed": [{"seed": r[0], "V": r[1], "U": r[2], "VU": r[3]} for r in rows],
        }, f, indent=2, ensure_ascii=False)
    print(f"已寫出 {out}")


if __name__ == "__main__":
    main()
