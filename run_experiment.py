#!/usr/bin/env python3
"""
==============================================================================
run_experiment.py — SQMG 參數搜尋的統一入口（v12.0）
==============================================================================

取代先前散落在根目錄的多支 runner。一個指令跑任何演算法、任何目標函數、
任何派工模式：

    # 單節點 8 GPU，RR-QPSO，V×U 目標
    python run_experiment.py --optimizer rr_qpso --M 32 --T 16

    # 6 節點 48 GPU（需在 sbatch 配額內），CMA-ES
    python run_experiment.py --optimizer cmaes --M 32 --T 16 \
        --dispatch slurm --nodes 6

    # HBA/HBD 多目標
    python run_experiment.py --optimizer rr_qpso --objective hbahbd

演算法（--optimizer）：
    sobol  bo  batch_bo  cmaes  de  spsa  qpso  rr_qpso

目標函數（--objective）：
    vu      fitness = V × U（預設）
    hbahbd  fitness = (V×U) × ((1-w) + w·chem_closeness)
            chem_closeness = exp(-0.5·((|HBA-4|/σ)² + (|HBD-3|/σ)²))

公平比較的關鍵
--------------
`--M` × `--T` 就是**總評估預算**，對所有演算法一致，由 BaseOptimizer 強制執行。
M 同時作為批次大小（BO 除外，它一次只提 1 點——這正是要量的東西）。

放置位置：run_experiment.py（專案根目錄）
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np


def setup_logger(path: str) -> logging.Logger:
    logger = logging.getLogger(f"sqmg_{os.path.basename(path)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout);            sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def make_fitness_fn(args):
    """回傳 fitness_fn；vu 目標回傳 None（基底預設 V×U）。"""
    if args.objective == "vu":
        return None

    w, hs, ds = args.chem_weight, args.hba_sigma, args.hbd_sigma
    hba_t, hbd_t = args.hba_target, args.hbd_target

    def fitness(metrics):
        v, u = float(metrics[0]), float(metrics[1])
        hba = float(metrics[2]) if len(metrics) > 2 else 0.0
        hbd = float(metrics[3]) if len(metrics) > 3 else 0.0
        closeness = np.exp(-0.5 * (((abs(hba - hba_t) / hs) ** 2)
                                   + ((abs(hbd - hbd_t) / ds) ** 2)))
        return (v * u) * ((1.0 - w) + w * closeness)

    return fitness


def parse_args():
    p = argparse.ArgumentParser(
        description="SQMG 統一實驗入口（v12.0）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--optimizer", required=True,
                   help="sobol | bo | batch_bo | cmaes | de | spsa | qpso | rr_qpso")
    p.add_argument("--objective", default="vu", choices=["vu", "hbahbd"])
    p.add_argument("--M", type=int, default=32, help="批次大小（族群大小）")
    p.add_argument("--T", type=int, default=16, help="迭代數；預算 = M × T")
    p.add_argument("--max_evals", type=int, default=None,
                   help="直接指定評估預算，覆寫 M×T")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample", type=int, default=5000, help="每次評估的 shots")
    p.add_argument("--backend", type=str, default="cudaq_nvidia")
    p.add_argument("--subprocess_timeout", type=int, default=600)

    # 派工
    p.add_argument("--dispatch", default="local", choices=["local", "slurm"])
    p.add_argument("--pool", action="store_true", default=True,
                   help="使用常駐 worker pool（預設開啟）。實測可省下每次評估 "
                        "~24s 的行程啟動開銷（33.8 → ~10 s/eval）。")
    p.add_argument("--no_pool", action="store_false", dest="pool",
                   help="關閉常駐 worker，改用 v12.0 的每次開新行程模式。")
    p.add_argument("--shot_seed", type=int, default=0,
                   help="傳給 sample_molecule(random_seed=)。0 = 與歷史資料一致的"
                        "確定性目標函數；改成其他值會換一條 shot 序列。")

    # ── 超參數覆寫（用於「參數公平」比較）────────────────────────────────
    #   ★ 為什麼需要：原版 QPSO 與 RR-QPSO 的 α 排程原本不同
    #     （QPSO 用文獻預設 [0.5, 1.0]；RR-QPSO 用調校過的 [0.3, 1.2]），
    #     兩者的差異因此混淆了「RR 機制的效果」與「α 排程的效果」。
    #     要隔離出 RR 機制的貢獻，必須把 α 對齊後再比。
    p.add_argument("--alpha_max", type=float, default=None,
                   help="QPSO / RR-QPSO 的收縮擴張係數上界（不指定則用各自預設）")
    p.add_argument("--alpha_min", type=float, default=None,
                   help="QPSO / RR-QPSO 的收縮擴張係數下界（不指定則用各自預設）")
    # ── BO 的 GP 訓練點上限 ────────────────────────────────────────────
    #   預設 400 是為了讓 BO 在牆鐘內跑完（GP 是 O(n³)），但在大預算下
    #   會讓 GP 看不到絕大多數觀測：9,664 次評估時被丟掉約 96%。
    #   實測 BO 在 1,984→9,664 只進步 +0.0300，停滯很可能就來自這個上限，
    #   而不是 BO 本身的性質。開成參數以便驗證：把它設得 >= max_evals
    #   等於不設限，可用來區分「BO 的極限」與「我們的實作限制」。
    p.add_argument("--max_gp_points", type=int, default=None,
                   help="BO / Batch BO 的 GP 訓練點數上限（預設 400；"
                        "設為 >= max_evals 等於不設限）")
    p.add_argument("--tune_every", type=int, default=None,
                   help="BO 超參數重調頻率（預設每 25 次迭代）")
    p.add_argument("--ablate", type=str, default=None,
                   choices=["none", "sobol", "obl", "ae", "vu", "mc"],
                   help="RR-QPSO 組件消融：關閉指定的單一組件。"
                        "sobol=Sobol初始化 obl=對立式學習 "
                        "ae=AE-QTS有符號mbest vu=V-U解耦 mc=mode collapse回收")
    p.add_argument("--n_gpus", type=int, default=8)
    p.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--nodes", type=int, default=None)
    p.add_argument("--gpus_per_node", type=int, default=8)
    p.add_argument("--job_dir", type=str, default=None)
    p.add_argument("--srun_retries", type=int, default=3)

    # HBA/HBD 目標參數
    p.add_argument("--chem_weight", type=float, default=0.40)
    p.add_argument("--hba_target", type=float, default=4.0)
    p.add_argument("--hbd_target", type=float, default=3.0)
    p.add_argument("--hba_sigma", type=float, default=1.0)
    p.add_argument("--hbd_sigma", type=float, default=1.0)

    # 輸出
    p.add_argument("--data_dir", type=str, default="results_benchmark")
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="從既有 CSV/state 續跑（供長跑撞到 SLURM 時間上限時接續）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from optimizers import get_optimizer
    from qmg.utils import ConditionalWeightsGenerator

    task = args.task_name or (
        f"{args.optimizer}_{args.objective}_M{args.M}T{args.T}_s{args.seed}")
    os.makedirs(args.data_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.data_dir, f"{task}.log"))

    max_evals = args.max_evals if args.max_evals else args.M * args.T

    logger.info("=" * 70)
    logger.info(f"SQMG 實驗 v12.0  task={task}")
    logger.info(f"  演算法        : {args.optimizer}")
    logger.info(f"  目標函數      : {args.objective}")
    logger.info(f"  預算 M×T      : {args.M} × {args.T} = {max_evals} 次評估")
    logger.info(f"  shots         : {args.num_sample}")
    logger.info(f"  seed          : {args.seed}")
    logger.info(f"  heavy atoms   : {args.num_heavy_atom}")
    logger.info("=" * 70)

    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom, smarts=None, disable_connectivity_position=[])
    D = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"  參數維度 D    : {D}")

    # ── 建立評估器 ───────────────────────────────────────────────────────
    report_hbahbd = (args.objective == "hbahbd")
    if args.dispatch == "slurm":
        from evaluator import make_slurm_evaluator
        env_n = os.environ.get("SLURM_NNODES") or os.environ.get("SLURM_JOB_NUM_NODES")
        nodes = args.nodes or (int(env_n) if env_n and env_n.isdigit() else 1)
        job_dir = args.job_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".mn_jobs", task)
        logger.info(f"  派工          : slurm  {nodes} 節點 × "
                    f"{args.gpus_per_node} GPU = {nodes*args.gpus_per_node} 並行")
        batch_fn = make_slurm_evaluator(
            cwg=cwg, logger=logger, job_dir=job_dir, nodes=nodes,
            gpus_per_node=args.gpus_per_node,
            num_heavy_atom=args.num_heavy_atom, num_sample=args.num_sample,
            backend=args.backend, timeout=args.subprocess_timeout,
            srun_retries=args.srun_retries)
    else:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()][: args.n_gpus]
        if args.pool:
            from evaluator import make_pooled_evaluator
            logger.info(f"  派工          : local pool（常駐 worker）GPU={gpu_ids}")
            batch_fn = make_pooled_evaluator(
                cwg=cwg, logger=logger, gpu_ids=gpu_ids,
                num_heavy_atom=args.num_heavy_atom, num_sample=args.num_sample,
                backend=args.backend, timeout=args.subprocess_timeout,
                report_hbahbd=report_hbahbd, shot_seed=args.shot_seed)
        else:
            from evaluator import make_local_evaluator
            logger.info(f"  派工          : local（每次評估開新行程）GPU={gpu_ids}")
            batch_fn = make_local_evaluator(
                cwg=cwg, logger=logger, gpu_ids=gpu_ids,
                num_heavy_atom=args.num_heavy_atom, num_sample=args.num_sample,
                backend=args.backend, timeout=args.subprocess_timeout)

    if report_hbahbd:
        logger.info("  ⚠ hbahbd 目標需要 worker 回報 HBA/HBD 欄位。")

    # ── 執行 ─────────────────────────────────────────────────────────────
    cls = get_optimizer(args.optimizer)

    # 只有接受 α 參數的最適化器（qpso / rr_qpso）才傳入；其餘忽略。
    import inspect
    extra = {}
    accepted = inspect.signature(cls.__init__).parameters
    for k, v in (("alpha_max", args.alpha_max), ("alpha_min", args.alpha_min),
                 ("ablate", args.ablate),
                 ("max_gp_points", args.max_gp_points),
                 ("tune_every", args.tune_every)):
        if v is not None and k in accepted:
            extra[k] = v
    if extra:
        logger.info(f"  超參數覆寫    : {extra}")
    elif args.alpha_max is not None or args.alpha_min is not None:
        logger.info(f"  ⚠ {args.optimizer} 不接受 alpha 參數，忽略覆寫")

    opt = cls(
        n_params          = D,
        max_evals         = max_evals,
        batch_evaluate_fn = batch_fn,
        logger            = logger,
        seed              = args.seed,
        data_dir          = args.data_dir,
        task_name         = task,
        batch_size        = args.M,
        fitness_fn        = make_fitness_fn(args),
        resume            = args.resume,
        **extra,
    )

    if opt.n_evals >= max_evals:
        logger.info(f"已完成 {opt.n_evals}/{max_evals} 次評估，無需再跑。")
        return

    t0 = time.time()
    best_x, best_f = opt.run()
    elapsed = time.time() - t0

    if best_x is not None:
        np.save(os.path.join(args.data_dir, f"{task}_best_params.npy"), best_x)
    with open(os.path.join(args.data_dir, f"{task}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "task": task, "algo": args.optimizer, "objective": args.objective,
            "seed": args.seed, "M": args.M, "T": args.T,
            "max_evals": max_evals, "n_evals": opt.n_evals,
            "num_sample": args.num_sample,
            "best_fitness": opt.best_f,
            "best_validity": opt.best_v, "best_uniqueness": opt.best_u,
            "elapsed_s": round(elapsed, 1),
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"完成。best={best_f:.6f}  耗時 {elapsed/60:.1f} 分")


if __name__ == "__main__":
    main()
