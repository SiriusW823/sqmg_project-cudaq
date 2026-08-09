#!/usr/bin/env python3
"""
==============================================================================
benchmark/revalidate.py — 重新評估已儲存的 best_params（v12.0）
==============================================================================

要回答的問題
------------
「我報出來的 V×U 最大值，是這組參數真正的實力，還是剛好抽到運氣好的那一次？」

為什麼一定要做這件事
--------------------
論文裡的 V×U 是「N 次評估中的最大值」，而 N 動輒上萬（例如 M=64,T=150 → 9,664 次）。
每一次評估都是 5000 shots 的**抽樣估計**，本身帶雜訊。
在上萬個帶雜訊的估計中取最大值，會系統性地挑中「雜訊往上偏」的那一次——
這就是 winner's curse。因此**報出的最大值必定是真實值的上偏估計**，
偏多少取決於雜訊大小與 N。

這支程式的做法很直接：把 best_params 載回來，用**全新的隨機 shots** 重複評估 R 次，
report 平均值與 95% 信賴區間。
  - 若重評的平均值 ≈ 原報告值 → 原數字站得住。
  - 若明顯低於原報告值 → 原數字含選擇偏差，論文應改報「重評平均 ± CI」。

這同時也給出**單次評估的雜訊大小**（重評的標準差），
是後續一切統計推論的基礎量。

用法：
    python benchmark/revalidate.py --params results_v8/unconditional_9_ae_v8_best_params.npy \\
                                   --repeats 20 --num_sample 5000
    python benchmark/revalidate.py --auto --repeats 20      # 自動挑主要的幾組
==============================================================================
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# 主要的 headline 結果（--auto 時使用）
AUTO_TARGETS = [
    "results_qpso_aeqts/unconditional_9_qpso_aeqts_M64T150_best_params.npy",
    "results_sweep_M128/unconditional_9_ae_M128T150_best_params.npy",
    "results_v8/unconditional_9_ae_v8_best_params.npy",
    "results_qpso_nosobol/unconditional_9_qpso_nosobol_M64T150_best_params.npy",
    "results_qpso_pure/unconditional_9_qpso_pure_M64T150_best_params.npy",
]


def setup_logger(path: str) -> logging.Logger:
    lg = logging.getLogger("reval"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh)
    return lg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params", nargs="*", default=None,
                   help="要重新評估的 best_params .npy（可多個）")
    p.add_argument("--auto", action="store_true", help="自動挑主要的 headline 結果")
    p.add_argument("--repeats", type=int, default=20, help="每組參數重複評估次數")
    p.add_argument("--num_sample", type=int, default=5000)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--backend", default="cudaq_nvidia")
    p.add_argument("--n_gpus", type=int, default=2,
                   help="單節點並行度。實測 2 是吞吐量最佳點。")
    p.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7")
    p.add_argument("--dispatch", default="local", choices=["local", "slurm"])
    p.add_argument("--nodes", type=int, default=None)
    p.add_argument("--out_dir", default="results_revalidate")
    p.add_argument("--reported", type=float, default=None,
                   help="原論文報告的 V×U，用來對比偏差")
    args = p.parse_args()

    targets = args.params or (AUTO_TARGETS if args.auto else None)
    if not targets:
        print("需要 --params 或 --auto"); sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.out_dir, "revalidate.log"))

    from qmg.utils import ConditionalWeightsGenerator
    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=None,
                                      disable_connectivity_position=[])

    if args.dispatch == "slurm":
        from evaluator import make_slurm_evaluator
        env_n = os.environ.get("SLURM_NNODES")
        nodes = args.nodes or (int(env_n) if env_n and env_n.isdigit() else 1)
        batch_fn = make_slurm_evaluator(
            cwg=cwg, logger=logger,
            job_dir=os.path.join(REPO, ".mn_jobs", "revalidate"),
            nodes=nodes, gpus_per_node=2,
            num_heavy_atom=args.num_heavy_atom, num_sample=args.num_sample,
            backend=args.backend, timeout=900)
    else:
        from evaluator import make_local_evaluator
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()][: args.n_gpus]
        batch_fn = make_local_evaluator(
            cwg=cwg, logger=logger, gpu_ids=gpu_ids,
            num_heavy_atom=args.num_heavy_atom, num_sample=args.num_sample,
            backend=args.backend, timeout=900)

    summary = []
    logger.info("=" * 78)
    logger.info(f"重新評估 best_params：{len(targets)} 組 × {args.repeats} 次  "
                f"shots={args.num_sample}")
    logger.info("=" * 78)

    for rel in targets:
        path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        if not os.path.exists(path):
            logger.warning(f"跳過（找不到）：{rel}")
            continue

        w = np.load(path)
        # 注意：best_params 儲存的是「未套用 chemistry constraint」的原始向量，
        # evaluator 內部會再套一次，與最適化當時的流程完全一致。
        X = np.tile(w, (args.repeats, 1))

        logger.info(f"\n--- {rel}  (D={len(w)}) ---")
        t0 = time.time()
        res = batch_fn(X)
        dt = time.time() - t0

        V = np.array([r[0] for r in res], dtype=float)
        U = np.array([r[1] for r in res], dtype=float)
        VU = V * U
        ok = VU > 0
        if ok.sum() == 0:
            logger.warning("  全部評估失敗，跳過。")
            continue
        VU, V, U = VU[ok], V[ok], U[ok]

        mean, sd = VU.mean(), VU.std(ddof=1)
        se = sd / np.sqrt(len(VU))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se

        logger.info(f"  n={len(VU)}  耗時 {dt:.0f}s")
        logger.info(f"  V×U  平均={mean:.4f}  標準差={sd:.4f}  "
                    f"95% CI=[{lo:.4f}, {hi:.4f}]")
        logger.info(f"  V×U  最小={VU.min():.4f}  最大={VU.max():.4f}")
        logger.info(f"  V    平均={V.mean():.4f}±{V.std(ddof=1):.4f}")
        logger.info(f"  U    平均={U.mean():.4f}±{U.std(ddof=1):.4f}")

        rec = {
            "params": rel, "n": int(len(VU)), "num_sample": args.num_sample,
            "vu_mean": float(mean), "vu_std": float(sd),
            "vu_ci95": [float(lo), float(hi)],
            "vu_min": float(VU.min()), "vu_max": float(VU.max()),
            "v_mean": float(V.mean()), "u_mean": float(U.mean()),
            "elapsed_s": round(dt, 1),
        }
        summary.append(rec)

    # ── 總結表 ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 100)
    logger.info("總結：重新評估 vs 原報告最大值")
    logger.info("=" * 100)
    logger.info(f"{'參數來源':<52}{'重評平均':>10}{'單次雜訊σ':>11}{'95% CI':>22}")
    logger.info("-" * 100)
    for r in summary:
        name = os.path.dirname(r["params"])
        lo, hi = r["vu_ci95"]
        ci = f"[{lo:.4f}, {hi:.4f}]"
        logger.info(f"{name:<52}{r['vu_mean']:>10.4f}{r['vu_std']:>11.4f}{ci:>22}")
    logger.info("=" * 100)
    logger.info("判讀：重評平均 ≈ 原報告最大值 → 原數字可信；")
    logger.info("      重評平均 明顯低於 原報告最大值 → 原數字含 winner's curse 偏差，")
    logger.info("      論文應改報『重評平均 ± 95% CI』並註明取樣次數。")

    out = os.path.join(args.out_dir, "revalidate_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n已寫出 {out}")


if __name__ == "__main__":
    main()
