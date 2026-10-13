#!/usr/bin/env python3
"""
==============================================================================
benchmark/rrfair2_analyze.py — PREREGISTRATION_rrqpso_fair.md §5 / §8 的分析
==============================================================================

Same statistics as the analysis that was run on the void batch (§8.2), moved
into the repository so the code that produces the corrected batch's numbers is
on record before it is run. It is split into the two stages §8.5 fixes:

  --stage h1        allowed once all 60 qpso + rr_qpso2 runs reached 2,000
                    evaluations. §5.1 checks, then H1 (@2,000) and H2 (@500).
                    Ablation arms are not summarised and not tested; their only
                    use is the §8.4 "not identical at 500" check, which compares
                    values for equality and prints nothing.
  --stage ablation  only once all 150 runs completed. Same §5.1 checks over
                    every arm, H1/H2 again (identical numbers: same RNG order),
                    then the three ablation arms, Holm-corrected, exploratory.

Nothing about hypotheses, endpoints, tests or decision rules differs from the
pre-registration; the stage split only keeps partial ablation data out of view.

    python benchmark/rrfair2_analyze.py --data_dir results_rrfair2 --stage h1
==============================================================================
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

M = 64
N_SEEDS = 30
PRIMARY = ("qpso", "rr2")
ABLATIONS = ("nosobol", "norr", "nofg")
WANT = {"qpso": {}, "rr2": {}, "nosobol": {"sobol_init": "False"},
        "norr": {"rank_refined": "False"}, "nofg": {"fitness_guided": "False"}}
rng = np.random.default_rng(20260922)


def load(d, end):
    out = collections.defaultdict(dict)
    heads = collections.defaultdict(set)
    for p in sorted(glob.glob(os.path.join(d, "rrf_*_s*.csv"))):
        arm, seed = os.path.basename(p)[4:].rsplit("_s", 1)
        seed = int(seed.split(".")[0])
        best, n, head = 0.0, 0, []
        for i, r in enumerate(csv.DictReader(open(p, newline="", errors="ignore"))):
            n += 1
            if i < M:
                head.append((r.get("validity"), r.get("uniqueness")))
            if n <= end:
                try:
                    best = max(best, float(r.get("best_fitness") or 0))
                except (TypeError, ValueError):
                    pass
        if n >= end:
            out[arm][seed] = best
        if len(head) == M:
            heads[arm].add(tuple(head))
    return out, heads


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def boot(dd, B=20000):
    idx = rng.integers(0, len(dd), size=(B, len(dd)))
    return np.percentile(np.median(np.asarray(dd)[idx], axis=1), [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def test(x, y, label):
    dd = x - y
    p = 1.0 if np.allclose(dd, 0) else wilcoxon(x, y, alternative="greater").pvalue
    lo, hi = boot(dd)
    dz = dd.mean() / dd.std(ddof=1) if dd.std(ddof=1) > 0 else 0.0
    print(f"  {label}")
    print(f"     Δ中位 {np.median(dd):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"δ {cliffs(x, y):+.3f}  d_z {dz:+.3f}")
    print(f"     勝/負 {int((dd > 0).sum())}/{int((dd < 0).sum())}  p={p:.4g}")
    return p, np.median(dd)


def gate_checks(d, stage):
    arms = PRIMARY if stage == "h1" else PRIMARY + ABLATIONS
    d2000, heads = load(d, 2000)
    print("=" * 74)
    print(f"§5.1 harness 檢查（強制，先於任何檢定）  stage={stage}")
    print("=" * 74)
    gate = True
    # (a) same number of completed runs — and all of them (§8.5)
    counts = {a: len(d2000.get(a, {})) for a in arms}
    ok_n = all(c == N_SEEDS for c in counts.values())
    gate &= ok_n
    print(f"  (a) 每個 arm 皆 {N_SEEDS} 個 run 達 2,000 次：{'✓' if ok_n else '✗'}  {counts}")
    # (b) distinct first-M rows per arm
    for a in arms:
        n_runs, n_start = len(d2000.get(a, {})), len(heads.get(a, set()))
        ok = n_runs > 0 and n_start == n_runs
        gate &= ok
        print(f"  (b) {a:9s} runs={n_runs:3d}  相異起點={n_start:3d}  "
              f"{'✓' if ok else '✗ 起點重複 → 作廢'}")
    # (c) §8.4 — the configuration was really applied (every log present, all arms)
    cfg_bad = []
    for p in sorted(glob.glob(os.path.join(d, "console_rrf_*_s*.log"))):
        arm = os.path.basename(p)[len("console_rrf_"):].rsplit("_s", 1)[0]
        txt = open(p, errors="ignore").read()
        m = re.findall(r"超參數覆寫\s*:\s*(\{.*?\})", txt)
        ok = (bool(m) and "忽略覆寫" not in txt
              and "'alpha_max': 1.2" in m[-1] and "'alpha_min': 0.3" in m[-1])
        for k, v in WANT.get(arm, {}).items():
            ok = ok and f"'{k}': {v}" in m[-1]
        if not ok:
            cfg_bad.append(os.path.basename(p))
    print(f"  (c) 組態（log 覆寫行）：{'✓' if not cfg_bad else '✗ ' + str(len(cfg_bad)) + ' 個不符'}")
    for b in cfg_bad[:5]:
        print(f"      ✗ {b}")
    gate &= not cfg_bad
    # (c) §8.4 — the rr-family arms must not be identical at 500 (equality only)
    d500, _ = load(d, 500)
    fam = [a for a in ("rr2",) + ABLATIONS if a in d500]
    common = sorted(set.intersection(*[set(d500[a]) for a in fam])) if len(fam) > 1 else []
    same = (all(all(d500[a][s] == d500["rr2"][s] for s in common) for a in fam if a != "rr2")
            if common else True)
    print(f"  (c) rr 家族 @500 不可全同（共同 seed {len(common)}）：{'✗ 全同或無法判斷 → 不檢定' if same else '✓'}")
    gate &= not same
    if not gate:
        print("\n★ harness 檢查未通過，依預登記不做任何假設檢定。")
        sys.exit(1)
    print("\n  ✓ 通過，繼續進行預登記的檢定\n")
    return d2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="results_rrfair2")
    ap.add_argument("--stage", choices=("h1", "ablation"), required=True)
    args = ap.parse_args()
    d = args.data_dir
    show = PRIMARY if args.stage == "h1" else PRIMARY + ABLATIONS

    d2000 = gate_checks(d, args.stage)

    for END, lab in ((2000, "H1 主要終點"), (500, "H2 次要終點（僅在 H1 成立時解讀）")):
        dd, _ = load(d, END)
        print("=" * 74)
        print(f"{lab}：第 {END} 次評估的 best-so-far V×U")
        print("=" * 74)
        for arm in show:
            if arm not in dd:
                continue
            v = np.array([dd[arm][s] for s in sorted(dd[arm])])
            print(f"  {arm:9s} n={len(v):2d}  中位 {np.median(v):.4f}  "
                  f"平均 {v.mean():.4f} ± {v.std(ddof=1):.4f}  [{v.min():.4f}, {v.max():.4f}]")
        seeds = sorted(set(dd["qpso"]) & set(dd["rr2"]))
        x = np.array([dd["rr2"][s] for s in seeds])
        y = np.array([dd["qpso"][s] for s in seeds])
        print(f"\n  配對 seed：{len(seeds)}")
        name = "H1" if END == 2000 else "H2"
        p, med = test(x, y, f"{name}  rr_qpso2 > qpso ?")
        print(f"     → {'支持' if p < 0.05 else '不支持'}（α=0.05，單尾）")
        if med < 0:
            pr = wilcoxon(y, x, alternative="greater").pvalue
            print(f"     ★ 實際方向相反，反向 p={pr:.4g}（描述用）")
        print()

    if args.stage == "h1":
        print("（消融臂依 §8.5 在 150 個 run 全部完成後才分析：--stage ablation）")
        return

    print("=" * 74)
    print("組件消融 @2,000（探索性；本研究未針對此設計檢定力）")
    print("=" * 74)
    seeds = sorted(set(d2000["rr2"]))
    labels = {"nosobol": "移除 Sobol 初始化", "norr": "移除 rank-refined mbest",
              "nofg": "移除 fitness-guided"}
    ps, rows = [], []
    for arm in ABLATIONS:
        ss = sorted(set(d2000[arm]) & set(seeds))
        full = np.array([d2000["rr2"][s] for s in ss])
        abl = np.array([d2000[arm][s] for s in ss])
        diff = full - abl          # >0：完整版較好 = 該組件有貢獻
        ps.append(wilcoxon(full, abl, alternative="greater").pvalue)
        rows.append((arm, full, abl, diff, ss))
    for (arm, full, abl, diff, ss), p, pa in zip(rows, ps, holm(ps)):
        lo, hi = boot(diff)
        print(f"  {labels[arm]}（n={len(ss)}）")
        print(f"     完整版 − 消融版：Δ中位 {np.median(diff):+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  δ {cliffs(full, abl):+.3f}")
        print(f"     勝/負 {int((diff > 0).sum())}/{int((diff < 0).sum())}  "
              f"p={p:.4g}  p_holm={pa:.4g}  → {'該組件有貢獻' if pa < 0.05 else '無證據'}")
        print()


if __name__ == "__main__":
    main()
