#!/usr/bin/env python3
"""
==============================================================================
benchmark/anytime_analyze.py — PREREGISTRATION_anytime_vs_bo.md §5 的分析
==============================================================================

Committed before any confirmatory data. Order of operations is the
pre-registered one:

  1. §5.1 mandatory checks per pair (a–e); failing pairs are excluded and
     counted; > 4 excluded → batch void, no test.
  2. H1  S_24(rr_qpso2) > S_24(ax_bo)   one-sided exact paired Wilcoxon
     H2  S_4(ax_bo)    > S_4(rr_qpso2)  one-sided exact paired Wilcoxon
     Holm across H1, H2. Cliff's δ, d_z, 95% bootstrap CI (20,000) on the
     median paired difference.
  3. Secondary, descriptive: S_h at every checkpoint, crossover, evaluations
     completed, fraction with S_h ≥ 0.95, and the qpso arm (E3).

    python benchmark/anytime_analyze.py --data_dir results_anytime
==============================================================================
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

SEEDS = list(range(500, 524))
HOURS = [2, 4, 8, 12, 16, 20, 24]
K, SHOTS = 16, 5000
MAX_EXCLUDED = 4
rng = np.random.default_rng(20261012)


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def boot_ci(d, B=20000):
    d = np.asarray(d)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def read(path):
    try:
        return open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def check_arm(d, arm, seed):
    """回傳 (ok, [失敗原因])。"""
    why = []
    task = f"any_{arm}_s{seed}"
    log = read(os.path.join(d, f"{task}.log"))
    sj = os.path.join(d, f"{task}_summary.json")
    summ = json.load(open(sj)) if os.path.exists(sj) else {}
    if not log:
        return False, ["no log"], summ
    # (b) configuration
    if arm == "ax_bo":
        if "ARD=是" not in log or "lengthscale 維度=134" not in log:
            why.append("(b) Ax kernel audit missing / not ARD-134")
        if "先前 Sobol 共提出 5 個 trial" not in log:
            why.append("(b) Sobol trials != 5")
    else:
        m = re.findall(r"超參數覆寫\s*:\s*(\{.*?\})", log)
        if not m or "'alpha_max': 1.2" not in m[-1] or "'alpha_min': 0.3" not in m[-1]:
            why.append("(b) alpha override missing")
        if "依牆鐘" not in log:
            why.append("(b) alpha not annealed by time")
    if "[FATAL]" in log or "忽略覆寫" in log:
        why.append("(b) rejected/ignored parameter")
    # (c) fresh shot noise, contiguous seed ranges
    if "fresh shot noise" not in log:
        why.append("(c) fresh shot noise not on")
    rngs = [(int(a), int(b)) for a, b in re.findall(r"shot seeds (\d+)–(\d+)", log)]
    if not rngs or any(rngs[i][0] != rngs[i - 1][1] + 1 for i in range(1, len(rngs))):
        why.append("(c) seed ranges not contiguous")
    if rngs and rngs[0][0] != seed * 1_000_000:
        why.append("(c) seed base mismatch")
    # (d) budget + checkpoints
    if summ.get("time_budget_h") != 24:
        why.append("(d) time budget != 24 h")
    have = {float(re.search(r"_ckpt_([0-9.]+)h\.npz$", f).group(1))
            for f in glob.glob(os.path.join(d, f"{task}_ckpt_*h.npz"))}
    if not set(map(float, HOURS)) <= have:
        why.append(f"(d) checkpoints missing: {sorted(set(map(float, HOURS)) - have)}")
    # (e) ended by the budget, not a crash
    if "預算用盡，正常結束" not in log:
        why.append("(e) did not end by budget")
    return not why, why, summ


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="results_anytime")
    args = p.parse_args()
    d = args.data_dir

    print("=" * 76)
    print("§5.1 mandatory checks (per pair)")
    print("=" * 76)
    kept, excluded = [], []
    for s in SEEDS:
        ok_r, why_r, sr = check_arm(d, "rr_qpso2", s)
        ok_b, why_b, sb = check_arm(d, "ax_bo", s)
        why = [f"rr: {w}" for w in why_r] + [f"bo: {w}" for w in why_b]
        if sr.get("host") is None or sr.get("host") != sb.get("host"):
            why.append(f"(a) hosts differ: {sr.get('host')} vs {sb.get('host')}")
        if why:
            excluded.append(s); print(f"  seed {s}: EXCLUDED — " + "; ".join(why))
        else:
            kept.append(s); print(f"  seed {s}: ok  host={sr.get('host')}  "
                                 f"evals rr={sr.get('n_evals')} bo={sb.get('n_evals')}")
    print(f"\n  kept {len(kept)}/24, excluded {len(excluded)}")
    if len(excluded) > MAX_EXCLUDED:
        print(f"\n★ more than {MAX_EXCLUDED} pairs excluded → batch VOID; no hypothesis tested.")
        sys.exit(1)

    # ── test scores ───────────────────────────────────────────────────────
    S = {}
    for f in glob.glob(os.path.join(d, "test_scores", "*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if int(r["K"]) != K or int(r["num_sample"]) != SHOTS:
                continue
            S[(r["arm"], int(r["seed"]), float(r["ckpt_h"]))] = float(r["test_mean"])
    missing = [(a, s, h) for a in ("rr_qpso2", "ax_bo") for s in kept for h in HOURS
               if (a, s, float(h)) not in S]
    if missing:
        print(f"\n★ {len(missing)} test scores missing (e.g. {missing[:3]}) — run test_score.py first.")
        sys.exit(2)

    def vec(arm, h, seeds):
        return np.array([S[(arm, s, float(h))] for s in seeds])

    print("\n" + "=" * 76)
    print(f"Primary hypotheses  (n = {len(kept)} pairs; test score = mean V×U over "
          f"{K} unseen seeds × {SHOTS} shots)")
    print("=" * 76)
    tests = []
    for name, h, x_arm, y_arm in (("H1", 24, "rr_qpso2", "ax_bo"), ("H2", 4, "ax_bo", "rr_qpso2")):
        x, y = vec(x_arm, h, kept), vec(y_arm, h, kept)
        dd = x - y
        p = 1.0 if np.allclose(dd, 0) else wilcoxon(x, y, alternative="greater").pvalue
        lo, hi = boot_ci(dd)
        dz = dd.mean() / dd.std(ddof=1) if dd.std(ddof=1) > 0 else 0.0
        tests.append((name, h, x_arm, y_arm, x, y, dd, p, lo, hi, dz))
    padj = holm([t[7] for t in tests])
    for (name, h, xa, ya, x, y, dd, p, lo, hi, dz), pa in zip(tests, padj):
        print(f"  {name}: S_{h}({xa}) > S_{h}({ya}) ?")
        print(f"      medians {np.median(x):.4f} vs {np.median(y):.4f}   Δ median {np.median(dd):+.4f}"
              f"  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"      wins {int((dd > 0).sum())}/{len(dd)}  Cliff δ {cliffs(x, y):+.3f}  d_z {dz:+.3f}"
              f"  p {p:.4g}  p_holm {pa:.4g}  → {'SUPPORTED' if pa < 0.05 else 'not supported'}")

    print("\n" + "=" * 76)
    print("Secondary (descriptive, no claims)")
    print("=" * 76)
    print(f"  {'h':>3} {'rr_qpso2':>9} {'ax_bo':>9} {'Δ med':>8} {'rr wins':>8} {'≥0.95 rr/bo':>12}")
    cross = None
    for h in HOURS:
        x, y = vec("rr_qpso2", h, kept), vec("ax_bo", h, kept)
        dm = np.median(x - y)
        if cross is None and dm > 0:
            cross = h
        print(f"  {h:>3} {np.median(x):9.4f} {np.median(y):9.4f} {dm:+8.4f} "
              f"{int((x > y).sum()):>4}/{len(x):<3} {int((x >= .95).sum()):>5}/{int((y >= .95).sum()):<5}")
    print(f"  first checkpoint with positive median paired difference: {cross} h")

    q = {s: [S.get(("qpso", s, float(h))) for h in HOURS] for s in SEEDS}
    if any(v[-1] is not None for v in q.values()):
        qs = [s for s in kept if q[s][-1] is not None]
        x, y = vec("rr_qpso2", 24, qs), np.array([q[s][-1] for s in qs])
        print(f"\n  E3 (cross-node, descriptive): S_24 rr_qpso2 {np.median(x):.4f} vs qpso "
              f"{np.median(y):.4f}  (n={len(qs)}; rr wins {int((x > y).sum())})")


if __name__ == "__main__":
    main()
