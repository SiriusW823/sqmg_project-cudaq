#!/usr/bin/env python3
"""
==============================================================================
benchmark/analyze_benchmark.py — 彙整比較結果並產圖（v12.0）
==============================================================================

讀取 `results_benchmark/*.csv`（每個 (演算法, seed) 一個檔，格式由
`optimizers.BaseOptimizer.CSV_FIELDS` 保證一致），輸出：

  1. `benchmark_convergence.png`
     收斂曲線：橫軸「評估次數」（不是迭代數——這才是公平的比較軸），
     縱軸 best-so-far fitness。實線為 5 個 seed 的中位數，
     陰帶為 min–max 全距。用中位數而非平均，是因為 seed 數只有 5，
     平均值容易被單一次失敗的 run 拉走。

  2. `benchmark_final_box.png`
     各演算法最終 fitness 的箱型圖，直接呈現「跨 seed 的變異」——
     這正是跑 5 次要回答的問題：某個演算法贏，是穩定贏還是碰運氣。

  3. `benchmark_summary.csv` / 終端表格
     每個演算法的 median / mean ± std / best / worst，以及達到某門檻
     所需的評估次數（樣本效率）。

用法：
    python benchmark/analyze_benchmark.py --data_dir results_benchmark
==============================================================================
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizers import DISPLAY_NAMES, PLOT_ORDER      # noqa: E402


def load_runs(data_dir: str):
    """回傳 {algo: {seed: (evals, best_so_far)}}"""
    import csv as _csv
    runs: dict = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if os.path.basename(path).startswith("benchmark_"):
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            continue
        if not rows or "best_fitness" not in rows[0]:
            continue
        algo = rows[0].get("algo", "?")
        seed = int(rows[0].get("seed", 0))
        ev   = np.array([int(r["eval_index"]) for r in rows])
        bf   = np.array([float(r["best_fitness"]) for r in rows])
        runs.setdefault(algo, {})[seed] = (ev, bf)
    return runs


def to_grid(runs_for_algo: dict, n_max: int) -> np.ndarray:
    """把各 seed 的 best-so-far 對齊到 1..n_max 的共同網格。"""
    grid = np.arange(1, n_max + 1)
    out = []
    for _, (ev, bf) in sorted(runs_for_algo.items()):
        # best-so-far 是階梯函數 → 用 previous-value 補值
        idx = np.searchsorted(ev, grid, side="right") - 1
        idx = np.clip(idx, 0, len(bf) - 1)
        out.append(bf[idx])
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="results_benchmark")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="樣本效率門檻；預設取所有 run 最終中位數的最大值 × 0.95")
    args = p.parse_args()

    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.data_dir)
    if not runs:
        print(f"[ERROR] {args.data_dir} 下找不到任何結果 CSV。")
        sys.exit(1)

    algos = [a for a in PLOT_ORDER if a in runs] + \
            [a for a in runs if a not in PLOT_ORDER]
    n_max = min(max(ev[-1] for ev, _ in r.values()) for r in runs.values())

    # ── 統計表 ───────────────────────────────────────────────────────────
    finals = {a: to_grid(runs[a], n_max)[:, -1] for a in algos}
    med_best = max(np.median(v) for v in finals.values())
    thr = args.threshold if args.threshold else 0.95 * med_best

    rows = []
    print("=" * 96)
    print(f"SQMG 演算法比較   評估預算 {n_max}   門檻 {thr:.4f}")
    print("=" * 96)
    print(f"{'演算法':<28}{'seeds':>6}{'中位數':>10}{'平均±標準差':>18}"
          f"{'最佳':>9}{'最差':>9}{'達門檻評估數':>14}")
    print("-" * 96)

    for a in algos:
        g = to_grid(runs[a], n_max)
        f = g[:, -1]
        # 樣本效率：各 seed 首次達門檻的評估次數，取中位數（未達者記為 n_max+1）
        hits = []
        for r in g:
            w = np.argmax(r >= thr) if (r >= thr).any() else None
            hits.append((w + 1) if w is not None else n_max + 1)
        hit_med = int(np.median(hits))
        hit_str = str(hit_med) if hit_med <= n_max else "未達"

        print(f"{DISPLAY_NAMES.get(a,a):<28}{len(f):>6}{np.median(f):>10.4f}"
              f"{np.mean(f):>11.4f}±{np.std(f):<6.4f}"
              f"{f.max():>9.4f}{f.min():>9.4f}{hit_str:>14}")
        rows.append({
            "algo": a, "display": DISPLAY_NAMES.get(a, a), "n_seeds": len(f),
            "median": np.median(f), "mean": np.mean(f), "std": np.std(f),
            "best": f.max(), "worst": f.min(), "evals_to_threshold": hit_str,
        })
    print("=" * 96)

    import csv as _csv
    sp = os.path.join(out_dir, "benchmark_summary.csv")
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"已寫出 {sp}")

    # ── 繪圖 ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i % 10) for i, a in enumerate(algos)}

    # 1) 收斂曲線
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, n_max + 1)
    for a in algos:
        g = to_grid(runs[a], n_max)
        med = np.median(g, axis=0)
        lo, hi = g.min(axis=0), g.max(axis=0)
        lw = 2.6 if a == "rr_qpso" else 1.6
        ax.plot(x, med, label=DISPLAY_NAMES.get(a, a), color=colors[a], lw=lw)
        ax.fill_between(x, lo, hi, color=colors[a], alpha=0.12, linewidth=0)
    ax.set_xlabel("Number of objective evaluations")
    ax.set_ylabel("Best-so-far  V × U")
    ax.set_title(f"Convergence (median of {len(runs[algos[0]])} seeds, band = min–max)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "benchmark_convergence.png")
    fig.savefig(p1, dpi=180); plt.close(fig)
    print(f"已寫出 {p1}")

    # 2) 最終值箱型圖
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [finals[a] for a in algos]
    bp = ax.boxplot(data, labels=[DISPLAY_NAMES.get(a, a) for a in algos],
                    patch_artist=True, medianprops=dict(color="black"))
    for patch, a in zip(bp["boxes"], algos):
        patch.set_facecolor(colors[a]); patch.set_alpha(0.55)
    for i, a in enumerate(algos):      # 疊上個別 seed 的點
        y = finals[a]
        ax.scatter(np.full(len(y), i + 1) + np.random.uniform(-.07, .07, len(y)),
                   y, s=16, color="k", zorder=3, alpha=0.7)
    ax.set_ylabel("Final best  V × U")
    ax.set_title(f"Final performance across seeds (budget = {n_max} evaluations)")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "benchmark_final_box.png")
    fig.savefig(p2, dpi=180); plt.close(fig)
    print(f"已寫出 {p2}")


if __name__ == "__main__":
    main()
