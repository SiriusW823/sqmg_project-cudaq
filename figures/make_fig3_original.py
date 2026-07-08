#!/usr/bin/env python3
"""Regenerate fig3_convergence in the ORIGINAL style (single panel, viridis
M-colors, BO baseline as flat dashed line, NO extra BO growth curve).
The previously committed PNG/PDF were corrupted (truncated) since the very
first commit that added them -- this replaces them with a clean render.
Run this ON DGX111 (needs local results_sweep_M*/ + results_v8/ CSVs).
"""
import os, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

ROOT = os.path.expanduser("~/sqmg_project-cudaq")
OUT  = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
BO_BASELINE = 0.8834

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

runs = collect_sweep()

plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10.0, 7.2))
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(runs)))
for c, (M, (x, y, done)) in zip(cmap, runs.items()):
    tag = "" if done >= 150 else f" [iter {done}/150]"
    ax.plot(x, y, lw=2.2, color=c, label=f"M={M}{tag}", zorder=3)

ax.axhline(BO_BASELINE, color="#7f8c8d", ls="--", lw=1.8, zorder=2)
ax.text(2, BO_BASELINE + 0.012, f"BO baseline  (V×U = {BO_BASELINE:.3f})",
        ha="left", va="bottom", fontsize=14, color="#5d6d7e", fontstyle="italic")

ax.set_xlabel("QPSO iteration  $t$", fontsize=17)
ax.set_ylabel("Best  $V \\times U$  so far", fontsize=17)
ax.set_xlim(0, 152); ax.set_ylim(0.10, 0.97)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=13, ncol=2, framealpha=0.95)
ax.tick_params(labelsize=13)
plt.tight_layout()

png = os.path.join(OUT, "fig3_convergence.png")
pdf = os.path.join(OUT, "fig3_convergence.pdf")
plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf, bbox_inches="tight", facecolor="white")
print("saved", png, pdf, "| M values:", list(runs.keys()))

# integrity self-check
import struct
with open(png, "rb") as f:
    data = f.read()
print(f"png bytes on disk: {len(data)}")
assert data[-8:] == b'\x00\x00\x00\x00IEND\xaeB`\x82'[-8:] or data.endswith(b'IEND\xaeB`\x82'), "PNG missing IEND trailer!"
print("PNG integrity OK (IEND trailer present)")
