#!/usr/bin/env python3
"""Regenerate fig3_convergence (matches original style: single panel, viridis
M-colors, x in [0,150], y in [0.1,0.95]) but replace the flat 'BO baseline'
dashed line with the REAL BO growth curve reconstructed from Chen et al.'s
released log (results_reference_chen2025_unconditional_9.log), clipped to
iteration <=150 for visual consistency with the QPSO x-axis.
Run this ON DGX111 (needs local results_sweep_M*/ + results_v8/ CSVs).
"""
import os, re, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

ROOT = os.path.expanduser("~/sqmg_project-cudaq")
OUT  = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
BO_LOG = os.path.join(ROOT, "results_reference_chen2025_unconditional_9.log")

def curve(csv):
    df = pd.read_csv(csv)
    g = df.groupby("qpso_iter")["gbest_fitness"].max()
    return g.index.values, g.values, int(df["qpso_iter"].max())

def collect_sweep():
    runs = {}
    v8 = os.path.join(ROOT, "results_v8", "unconditional_9_ae_v8.csv")
    if os.path.exists(v8):
        runs[64] = curve(v8)
    for d in sorted(glob.glob(os.path.join(ROOT, "results_sweep_M*"))):
        try:
            M = int(os.path.basename(d).split("results_sweep_M")[1])
        except ValueError:
            continue
        cs = glob.glob(os.path.join(d, "*.csv"))
        if cs:
            runs[M] = curve(cs[0])
    return dict(sorted(runs.items()))

def parse_bo_log(path):
    it_re, val_re, uniq_re = (re.compile(r"Iteration number: (\d+)"),
        re.compile(r"validity \(maximize\): ([\d.]+)"),
        re.compile(r"uniqueness \(maximize\): ([\d.]+)"))
    recs = []; cur_it, cur_v = None, None
    with open(path) as f:
        for line in f:
            m = it_re.search(line)
            if m: cur_it, cur_v = int(m.group(1)), None; continue
            m = val_re.search(line)
            if m: cur_v = float(m.group(1)); continue
            m = uniq_re.search(line)
            if m and cur_v is not None:
                recs.append((cur_it, cur_v, float(m.group(1)))); cur_v = None
    return recs

runs = collect_sweep()
recs = parse_bo_log(BO_LOG)
bo_it  = np.array([r[0] for r in recs])
bo_vu  = np.array([r[1]*r[2] for r in recs])
bo_best = np.maximum.accumulate(bo_vu)
mask150 = bo_it <= 150
bo_it150, bo_best150 = bo_it[mask150], bo_best[mask150]
peak_all = bo_best[-1]
print(f"[BO] full-log final best-so-far: {peak_all:.4f}  |  best-so-far at iter<=150: {bo_best150[-1]:.4f}")

plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10.0, 7.2))
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(runs)))
for c, (M, (x, y, done)) in zip(cmap, runs.items()):
    tag = "" if done >= 150 else f" [iter {done}/150]"
    ax.plot(x, y, lw=2.2, color=c, label=f"M={M}{tag}", zorder=3)

# real BO growth curve (clipped to iter<=150), styled distinctly
ax.plot(bo_it150, bo_best150, color="#c0392b", lw=2.4, ls="-", zorder=4,
        label=f"BO (Chen 2025): {bo_best150[-1]:.3f} @ iter150")
ax.axhline(peak_all, color="#7f8c8d", ls="--", lw=1.5, zorder=2)
ax.text(4, peak_all + 0.010,
        f"BO full-log best-so-far  (V×U = {peak_all:.4f}, iter {int(bo_it[-1])}, 10000 shots)",
        ha="left", va="bottom", fontsize=10.5, color="#5d6d7e", fontstyle="italic")

ax.set_xlabel("QPSO iteration  $t$", fontsize=17)
ax.set_ylabel("Best  $V \\times U$  so far", fontsize=17)
ax.set_xlim(0, 152); ax.set_ylim(0.10, 0.97)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=12, ncol=2, framealpha=0.95)
ax.tick_params(labelsize=13)
plt.tight_layout()

png = os.path.join(OUT, "fig3_convergence.png")
pdf = os.path.join(OUT, "fig3_convergence.pdf")
plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf, bbox_inches="tight", facecolor="white")
print("saved", png, pdf, "| M values:", list(runs.keys()))
