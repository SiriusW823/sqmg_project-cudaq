#!/usr/bin/env python3
"""
==============================================================================
benchmark/stats_test.py — 演算法比較的推論統計（v12.0）
==============================================================================

`analyze_benchmark.py` 只做描述性統計（中位數、平均±標準差、箱型圖）。
這支補上「差異是否站得住腳」的推論部分。

方法選擇與理由
--------------
* **Mann–Whitney U（two-sided, exact）**
  每組只有 5 個 seed，樣本數太小、也無法驗證常態性，因此用無母數檢定而非 t 檢定。
  用 exact 版本（不是常態近似），因為 n=5 時近似不可靠。
  **不做配對**：不同演算法拿到同一個 seed 並不構成「同一個實驗條件」——
  seed 只是各自亂數流的起點，配對檢定的前提不成立。

* **Holm–Bonferroni 多重比較校正**
  8 個演算法 = 28 組兩兩比較。不校正的話，光靠運氣就會冒出假陽性。
  Holm 比 Bonferroni 有較高的檢定力，且同樣控制 family-wise error rate。

* **Cliff's delta 效果量**
  p 值只說「有沒有差」，不說「差多少」。Cliff's delta 是無母數的效果量，
  定義為 P(X>Y) − P(X<Y)，範圍 [-1, 1]。
  慣用門檻：|d|<0.147 可忽略、<0.33 小、<0.474 中、否則大。

* **Bootstrap 95% 信賴區間**
  對中位數與「兩演算法中位數之差」各做 10000 次 bootstrap，
  給出區間而非單一數字。區間包含 0 就是無法排除「沒有差異」。

檢定力的老實話
--------------
n=5 vs n=5 時，Mann–Whitney 的最小可能 p 值是 2/C(10,5) = 0.0079。
也就是說**即使兩組完全不重疊**，p 也只能低到 0.0079；經過 28 組 Holm 校正後
（最嚴的那組要乘以 28）就變成 0.22，**不可能達到顯著**。

這是本實驗設計的硬限制：**5 個 seed 不足以在校正後宣告任何兩兩差異顯著。**
因此下方同時report未校正的 p 值與效果量，並明確標示哪些結論需要更多 seed。

用法：
    python benchmark/stats_test.py --data_dir results_benchmark
    python benchmark/stats_test.py --data_dir results_benchmark --focus rr_qpso
==============================================================================
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizers import DISPLAY_NAMES, PLOT_ORDER      # noqa: E402


def load_finals(data_dir: str) -> dict:
    """{algo: {seed: final_best_fitness}}"""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if os.path.basename(path).startswith("benchmark_"):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if not rows or "best_fitness" not in rows[0]:
            continue
        algo = rows[0].get("algo")
        seed = int(rows[0].get("seed", 0))
        out.setdefault(algo, {})[seed] = float(rows[-1]["best_fitness"])
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple:
    """回傳 (delta, 量級標籤)。"""
    gt = sum((a > b) for a in x for b in y)
    lt = sum((a < b) for a in x for b in y)
    d = (gt - lt) / (len(x) * len(y))
    ad = abs(d)
    mag = ("negligible" if ad < 0.147 else
           "small"      if ad < 0.330 else
           "medium"     if ad < 0.474 else "large")
    return d, mag


def boot_ci(x: np.ndarray, stat=np.median, n_boot=10000, seed=0) -> tuple:
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff_ci(x, y, n_boot=10000, seed=0) -> tuple:
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(x, len(x), True)) - np.median(rng.choice(y, len(y), True))
          for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(pvals: list) -> list:
    """Holm–Bonferroni 校正，回傳校正後 p（順序與輸入相同）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return list(adj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="results_benchmark")
    p.add_argument("--focus", default="rr_qpso",
                   help="要重點比較的演算法")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    from scipy.stats import mannwhitneyu

    finals = load_finals(args.data_dir)
    algos = [a for a in PLOT_ORDER if a in finals]
    data = {a: np.array([finals[a][s] for s in sorted(finals[a])]) for a in algos}
    ns = {a: len(v) for a, v in data.items()}

    print("=" * 92)
    print("推論統計：Mann–Whitney U（exact, two-sided）+ Holm 校正 + Cliff's delta")
    print("=" * 92)

    # ── 各演算法的中位數與 bootstrap CI ───────────────────────────────────
    print(f"\n{'演算法':<28}{'n':>3}{'中位數':>9}{'95% CI (bootstrap)':>26}")
    print("-" * 92)
    for a in algos:
        lo, hi = boot_ci(data[a], seed=1)
        print(f"{DISPLAY_NAMES.get(a,a):<28}{ns[a]:>3}{np.median(data[a]):>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>26}")

    # ── 檢定力上限提醒 ───────────────────────────────────────────────────
    n = min(ns.values())
    from math import comb
    p_min = 2 / comb(2 * n, n)
    n_pairs = len(algos) * (len(algos) - 1) // 2
    print(f"\n[檢定力上限] n={n} vs n={n} 時 Mann–Whitney 的最小可能 p = {p_min:.4f}")
    print(f"             {n_pairs} 組兩兩比較，Holm 最嚴的一組需乘以 {n_pairs}"
          f" → 最小校正後 p = {min(p_min*n_pairs,1.0):.3f}")
    if p_min * n_pairs > args.alpha:
        print(f"             ⚠ 即使兩組完全不重疊也**無法**在 α={args.alpha} 下宣告顯著。")
        print(f"             要讓校正後檢定有機會顯著，每組至少需要 "
              f"{min([k for k in range(5,30) if 2/comb(2*k,k)*n_pairs <= args.alpha])} 個 seed。")

    # ── 全部兩兩比較 ─────────────────────────────────────────────────────
    pairs, praw = [], []
    for a, b in itertools.combinations(algos, 2):
        try:
            u, pv = mannwhitneyu(data[a], data[b], alternative="two-sided",
                                 method="exact")
        except Exception:
            u, pv = mannwhitneyu(data[a], data[b], alternative="two-sided")
        pairs.append((a, b))
        praw.append(float(pv))
    padj = holm(praw)

    print(f"\n{'比較':<46}{'p(raw)':>9}{'p(Holm)':>10}{'Cliff d':>10}{'量級':>12}")
    print("-" * 92)
    for (a, b), pr, pa in sorted(zip(pairs, praw, padj), key=lambda t: t[1]):
        d, mag = cliffs_delta(data[a], data[b])
        star = " *" if pr < args.alpha else ""
        print(f"{DISPLAY_NAMES.get(a,a)[:21]:<22} vs {DISPLAY_NAMES.get(b,b)[:21]:<22}"
              f"{pr:>9.4f}{pa:>10.3f}{d:>10.2f}{mag:>12}{star}")
    print("\n  * = 未校正 p < α（僅供參考，未通過多重比較校正）")

    # ── 聚焦比較 ─────────────────────────────────────────────────────────
    f = args.focus
    if f in data:
        print("\n" + "=" * 92)
        print(f"聚焦：{DISPLAY_NAMES.get(f,f)} vs 其他各演算法")
        print("=" * 92)
        print(f"{'對手':<28}{'中位數差':>10}{'95% CI of diff':>26}{'p(raw)':>9}{'Cliff d':>9}")
        print("-" * 92)
        for b in algos:
            if b == f:
                continue
            diff = np.median(data[f]) - np.median(data[b])
            lo, hi = boot_diff_ci(data[f], data[b], seed=2)
            try:
                _, pv = mannwhitneyu(data[f], data[b], alternative="two-sided",
                                     method="exact")
            except Exception:
                _, pv = mannwhitneyu(data[f], data[b], alternative="two-sided")
            d, _ = cliffs_delta(data[f], data[b])
            flag = "" if (lo <= 0 <= hi) else "  ← CI 不含 0"
            print(f"{DISPLAY_NAMES.get(b,b):<28}{diff:>+10.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>26}{pv:>9.4f}{d:>9.2f}{flag}")

    print("\n" + "=" * 92)
    print("判讀提醒：CI 包含 0 ⇒ 無法排除『兩者沒有差異』。")
    print("          n=5 時本實驗只能作為趨勢指標，不足以支撐『顯著優於』的主張。")
    print("=" * 92)


if __name__ == "__main__":
    main()
