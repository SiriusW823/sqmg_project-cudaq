#!/usr/bin/env python3
"""
==============================================================================
benchmark/test_score.py — 時間檢查點 incumbent 的獨立重評分
==============================================================================

預登記：docs/PREREGISTRATION_anytime_vs_bo.md §4.6

對 `<data_dir>/*_ckpt_<h>h.npz` 的每一個 incumbent，用 K 個**從未在最佳化中
出現過**的取樣種子（TEST_SEED_BASE + j，j = 0..K-1）重新評估，回報平均 V×U。
這是主要終點：衡量的是「這組參數有多好」，不是「最佳化當時哪一次抽樣運氣好」。

所有 arm、所有檢查點用**同一組**測試種子（配對），種子空間與最佳化期間的
shot_seed_for(base, k) 不相交（evaluator.TEST_SEED_BASE）。

多 GPU 平行：每張 GPU 跑一個 shard。

    CUDA_VISIBLE_DEVICES=0 python benchmark/test_score.py --data_dir results_anytime \\
        --shard 0 --n_shards 8 --out results_anytime/test_scores/shard0.csv

結果檔逐列附加；已評過的 (task, ckpt) 會跳過，可中斷續跑。
==============================================================================
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIELDS = ["task", "arm", "seed", "ckpt_h", "carried", "n_evals", "f_observed",
          "test_mean", "test_sd", "test_min", "test_max", "v_mean", "u_mean",
          "K", "num_sample", "test_seed_base", "elapsed_s"]

CKPT_RE = re.compile(r"^(?P<task>.+)_ckpt_(?P<h>[0-9.]+)h\.npz$")
TASK_RE = re.compile(r"^any_(?P<arm>.+)_s(?P<seed>\d+)$")


def main() -> None:
    from evaluator import TEST_SEED_BASE

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--K", type=int, default=16, help="每個 incumbent 的測試種子數")
    p.add_argument("--num_sample", type=int, default=5000)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--backend", default="cudaq_nvidia")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--hours", type=str, default="",
                   help="只評這些檢查點（逗號分隔，預設全部）")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    want_h = {float(x) for x in args.hours.split(",") if x.strip()}
    files = []
    for f in sorted(glob.glob(os.path.join(args.data_dir, "*_ckpt_*h.npz"))):
        m = CKPT_RE.match(os.path.basename(f))
        if not m:
            continue
        h = float(m.group("h"))
        if want_h and h not in want_h:
            continue
        files.append((m.group("task"), h, f))
    # 已評過的 (task, ckpt)：讀輸出目錄裡「所有」shard 的 CSV，而不只自己的那一份。
    # 分兩次評分時（先評已完成的配對、最後一波結束再補），檔案清單會變，
    # 若只看自己的 CSV，shard 指派一變就會重評別的 shard 已評過的檢查點。
    # shard 指派用 (task, ckpt) 的雜湊，與檔案清單和啟動時間無關：
    # 20 個 shard 不會同時開跑，若用清單索引，晚開跑的 shard 看到的清單不同，
    # 會漏評或重評。
    done = set()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    for prev in glob.glob(os.path.join(out_dir, "*.csv")):
        with open(prev, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(r["K"]) == args.K and int(r["num_sample"]) == args.num_sample:
                    done.add((r["task"], float(r["ckpt_h"])))
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    remaining = [x for x in files if (x[0], x[1]) not in done]
    todo = [x for x in remaining
            if zlib.crc32(f"{x[0]}|{x[1]:g}".encode()) % args.n_shards == args.shard]
    print(f"shard {args.shard}/{args.n_shards}: 檢查點共 {len(files)}，"
          f"已評 {len(files) - len(remaining)}，本 shard 待評 {len(todo)}  "
          f"(K={args.K}, shots={args.num_sample}, 測試種子 {TEST_SEED_BASE}+j)", flush=True)
    if not todo:
        return

    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
    from qmg.utils import ConditionalWeightsGenerator
    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=None,
                                      disable_connectivity_position=[])

    for task, h, f in todo:
        t0 = time.time()
        d = np.load(f)
        x = d["x"]
        mt = TASK_RE.match(task)
        arm, seed = (mt.group("arm"), int(mt.group("seed"))) if mt else ("?", -1)
        row = {"task": task, "arm": arm, "seed": seed, "ckpt_h": h,
               "carried": bool(d["carried"]), "n_evals": int(d["n_evals"]),
               "f_observed": float(d["f"]), "K": args.K,
               "num_sample": args.num_sample, "test_seed_base": TEST_SEED_BASE}
        if np.isnan(x).any():
            # 檢查點時還沒有任何完成的評估（只可能發生在極早的時間點）
            row.update(test_mean=0.0, test_sd=0.0, test_min=0.0, test_max=0.0,
                       v_mean=0.0, u_mean=0.0)
        else:
            w = cwg.apply_chemistry_constraint(x.copy())
            gen = MoleculeGeneratorCUDAQ(
                num_heavy_atom=args.num_heavy_atom, all_weight_vector=w,
                backend_name=args.backend, remove_bond_disconnection=True,
                chemistry_constraint=False)            # 已在上面套用
            V, U = [], []
            for j in range(args.K):
                # ★ 種子必須傳給 sample_molecule；在外面 set_random_seed 會被覆蓋。
                _, v, u = gen.sample_molecule(args.num_sample,
                                              random_seed=TEST_SEED_BASE + j)
                V.append(float(v)); U.append(float(u))
            V, U = np.array(V), np.array(U)
            vu = V * U
            row.update(test_mean=float(vu.mean()), test_sd=float(vu.std(ddof=1)),
                       test_min=float(vu.min()), test_max=float(vu.max()),
                       v_mean=float(V.mean()), u_mean=float(U.mean()))
        row["elapsed_s"] = round(time.time() - t0, 1)
        with open(args.out, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
        print(f"  {task} @ {h:g}h  observed={row['f_observed']:.4f}  "
              f"test={row['test_mean']:.4f} ± {row['test_sd']:.4f}  "
              f"({row['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
