#!/usr/bin/env python3
"""
==============================================================================
benchmark/launch_benchmark.py — 提交 8 演算法 × N seeds 的比較實驗（v12.0）
==============================================================================

排程策略：跨 run 平行，而不是 run 內平行
----------------------------------------
這個比較實驗有 40 個**彼此獨立**的 run（8 演算法 × 5 seeds）。
關鍵的實測發現（2026-07-25，5000 shots）：

    每節點並行 worker 數    單批 wall    有效評估吞吐量
    1                         47.6s        0.0210 /s
    2                         58.6s        0.0341 /s   ← 最佳
    4                        261.1s        0.0153 /s
    8                        280.9s        0.0285 /s
    （4 節點 × 8 worker 的真實批次：32 個評估中有 15 個因逾時作廢）

也就是說「每節點塞 8 個 worker」不但沒有比較快，還會掉一半的評估。
單節點最佳並行度是 **2**。

由此得到的排程方式：
  - 每個 run 只要 1 個節點的 2 顆 GPU（`--gres=gpu:2`），用 local 派工。
    SLURM 可以在同一個節點上塞 4 個這種 job（8 GPU / 2），
    6 個節點就能同時跑 24 個 run。
  - 序列型的 `bo` 每次只提 1 個點，只需 1 顆 GPU（`--gres=gpu:1`），
    一個節點可塞 8 個。

跨 run 平行遠優於 run 內平行：後者每個 run 都要獨占多個節點、
還要承受 srun step 的偶發失敗與 straggler 節點拖累。

用法
----
    # 先看排程計畫，不實際提交
    python benchmark/launch_benchmark.py --dry_run

    # 正式提交
    python benchmark/launch_benchmark.py --M 32 --T 16 --seeds 5

    # 先跑一個小規模 pilot 校正單次評估時間
    python benchmark/launch_benchmark.py --pilot

放置位置：benchmark/launch_benchmark.py
==============================================================================
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizers import PLOT_ORDER, SEQUENTIAL       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="提交 SQMG 演算法比較實驗")
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--shots", type=int, default=5000)
    p.add_argument("--objective", default="vu", choices=["vu", "hbahbd"])
    p.add_argument("--partition", default="nchc")
    p.add_argument("--time", default="24:00:00")
    p.add_argument("--data_dir", default="results_benchmark")
    p.add_argument("--algos", default=",".join(PLOT_ORDER),
                   help="逗號分隔；預設全部 8 個")
    p.add_argument("--gpus_per_job", type=int, default=2,
                   help="每個批次型 run 使用的 GPU 數。實測 2 是單節點吞吐量最佳點；"
                        "調大不會更快，反而會因逾時掉評估。")
    p.add_argument("--exclude", default="",
                   help="排除的節點（逗號分隔）。DGX102 實測比其他節點慢 14 倍。")
    p.add_argument("--chain", type=int, default=1,
                   help="序列型演算法(bo)串接的作業數。單一作業受 SLURM 時間上限，"
                        "9,664 次序列評估約需 50h > 48h，需拆成多個作業以 --resume 續跑。")
    p.add_argument("--dry_run", action="store_true", help="只印出計畫，不提交")
    p.add_argument("--pilot", action="store_true",
                   help="小規模校正：M=8 T=2 shots=1000，每個演算法 1 seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.pilot:
        args.M, args.T, args.seeds, args.shots = 8, 2, 1, 1000
        args.time = "01:00:00"
        args.data_dir = "results_benchmark_pilot"

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    n_runs = len(algos) * args.seeds

    print("=" * 74)
    print(f"SQMG 演算法比較實驗  M={args.M} T={args.T} "
          f"→ 每 run {args.M*args.T} 次評估   shots={args.shots}")
    print(f"演算法 {len(algos)} 個 × seeds {args.seeds} = {n_runs} runs")
    print(f"每個批次型 run：1 節點 / {args.gpus_per_job} GPU（可與其他 run 共用節點）")
    print(f"序列型 bo：1 節點 / 1 GPU")
    print(f"輸出目錄 {args.data_dir}")
    print("=" * 74)

    os.makedirs(os.path.join(REPO, args.data_dir), exist_ok=True)
    submitted = []

    # ★ seed-major 交錯提交順序（v12.1）
    #   先前用 algo-major（把某個演算法的 5 個 seed 一次送完）提交，結果佇列
    #   照順序消化，前 4 個演算法吃光了配額，qpso / rr_qpso / bo / batch_bo
    #   一個都沒排到——中途停止時拿不到任何橫向比較。
    #   改成 seed-major：先送所有演算法的 seed 0，再送 seed 1……
    #   這樣任何時間點停下來，都是「8 個演算法各 N 個 seed」的完整比較。
    order = [(algo, seed) for seed in range(args.seeds) for algo in algos]

    for algo, seed in order:
        is_seq = algo in SEQUENTIAL
        n_chain = args.chain if is_seq else 1
        if True:
            task = f"{algo}_{args.objective}_M{args.M}T{args.T}_s{seed}"
            ngpus = 1 if is_seq else args.gpus_per_job

            prev_jid = None
            for link in range(n_chain):
                resume = 1 if link > 0 else 0
                common = (f"ALL,ALGO={algo},SEED={seed},M={args.M},T={args.T},"
                          f"SHOTS={args.shots},OBJECTIVE={args.objective},"
                          f"DATA_DIR={args.data_dir},DISPATCH=local,"
                          f"NGPUS={ngpus},RESUME={resume}")

                cmd = ["sbatch", "--parsable",
                       "-N", "1", "-p", args.partition, "--time", args.time,
                       f"--gres=gpu:{ngpus}",
                       "-J", f"bm.{algo}.s{seed}" + (f".{link}" if n_chain > 1 else ""),
                       "-o", f"{args.data_dir}/console_{task}"
                             + (f"_{link}" if n_chain > 1 else "") + ".log",
                       "--export", common]
                if args.exclude:
                    cmd += [f"--exclude={args.exclude}"]
                if prev_jid:
                    # afterany：不論前一段是正常結束還是撞到時間上限，都接著跑
                    cmd += [f"--dependency=afterany:{prev_jid}"]
                cmd += ["benchmark/benchmark.slurm"]

                kind = "序列" if is_seq else f"批次 M={args.M}"
                link_s = f" [chain {link+1}/{n_chain}]" if n_chain > 1 else ""
                desc = f"{algo:<9} seed={seed}  {ngpus} GPU（{kind}）{link_s}"

                if args.dry_run:
                    print(f"  [DRY] {desc}")
                    continue
                try:
                    jid = subprocess.check_output(cmd, cwd=REPO, text=True).strip()
                    submitted.append(jid)
                    prev_jid = jid
                    print(f"  [{jid}] {desc}")
                except subprocess.CalledProcessError as e:
                    print(f"  [FAIL] {desc}\n         {e}")
                    break

    plan = submitted if not args.dry_run else []

    print("=" * 74)
    if args.dry_run:
        print("（dry run）未實際提交。")
    else:
        print(f"已提交 {len(submitted)} 個作業。")
        print(f"監控：squeue -u $USER")
        print(f"分析：python benchmark/analyze_benchmark.py --data_dir {args.data_dir}")
        print(f"檢定：python benchmark/stats_test.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
