#!/usr/bin/env python3
"""Generate Figure 2 (optimizer comparison) and Figure 3 (particle-count sweep)
from local DGX111 result CSVs. Re-run anytime as more runs complete.
Outputs PNGs into figures/ next to this script.
"""
import os, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

ROOT = os.path.expanduser("~/sqmg_project-cudaq")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
BO_BASELINE = 0.8834   # Chen et al. 2025 (JCTC), ~355 BO evaluations

def curve(csv):
    df = pd.read_csv(csv)
    g = df.groupby("qpso_iter")["gbest_fitness"].max()
    return g.index.values, g.values, int(df["qpso_iter"].max())

AE_V8 = os.path.join(ROOT, "results_v8", "unconditional_9_ae_v8.csv")
QPSO  = os.path.join(ROOT, "results_qpso_pure", "unconditional_9_qpso_pure_M64T150.csv")

# ============================ Figure 2 ============================
def make_fig2():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhline(BO_BASELINE, color="#7f8c8d", ls="--", lw=1.8, zorder=2)
    ax.text(150, BO_BASELINE+0.006,
            f"BO baseline (Chen 2025): V$\\times$U = {BO_BASELINE:.4f}",
            ha="right", va="bottom", fontsize=8.8, color="#5d6d7e", fontstyle="italic")
    if os.path.exists(AE_V8):
        x, y, _ = curve(AE_V8)
        ax.plot(x, y, color="#2e6da4", lw=2.4, zorder=4,
                label=f"AE-QPSO (this work):  {y[-1]:.4f}")
        ax.scatter([x[-1]], [y[-1]], color="#2e6da4", s=42, zorder=5)
    if os.path.exists(QPSO):
        x, y, done = curve(QPSO)
        complete = done >= 150
        lbl = f"QPSO (no AE-QTS):  {y[-1]:.4f}" + ("" if complete else f"  [iter {done}/150]")
        ax.plot(x, y, color="#c0392b", lw=2.4, ls="-" if complete else "--", zorder=3, label=lbl)
        ax.scatter([x[-1]], [y[-1]], color="#c0392b", s=42, zorder=5)
    else:
        ax.plot([], [], color="#c0392b", lw=2.4, label="QPSO (no AE-QTS):  run in progress")
    ax.set_xlabel("QPSO iteration  $t$", fontsize=11)
    ax.set_ylabel("Best  V $\\times$ U  so far", fontsize=11)
    ax.set_title("Optimizer comparison  (9 heavy atoms, $M{=}64$, $T{=}150$, 5000 shots)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 152); ax.set_ylim(0.10, 0.97)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    p = os.path.join(OUT, "fig2_optimizer_comparison.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print("fig2 saved:", p)

# ============================ Figure 3 ============================
def collect_sweep():
    runs = {}  # M -> (x, y, done)
    if os.path.exists(AE_V8):
        runs[64] = curve(AE_V8)            # M=64 == V8
    for d in sorted(glob.glob(os.path.join(ROOT, "results_sweep_M*"))):
        try:
            M = int(os.path.basename(d).split("results_sweep_M")[1])
        except ValueError:
            continue
        cs = glob.glob(os.path.join(d, "*.csv"))
        if cs:
            runs[M] = curve(cs[0])
    return dict(sorted(runs.items()))

def make_fig3():
    runs = collect_sweep()
    if not runs:
        print("fig3: no data yet"); return
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(runs)))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.2, 5.1))

    # (a) convergence curves per M
    for c, (M, (x, y, done)) in zip(cmap, runs.items()):
        tag = "" if done >= 150 else f" [iter {done}/150]"
        axa.plot(x, y, lw=2.0, color=c, label=f"M = {M}{tag}")
    axa.axhline(BO_BASELINE, color="#7f8c8d", ls="--", lw=1.5)
    axa.text(150, BO_BASELINE+0.005, "BO baseline", ha="right", va="bottom",
             fontsize=8.2, color="#5d6d7e", fontstyle="italic")
    axa.set_xlabel("QPSO iteration  $t$", fontsize=11)
    axa.set_ylabel("Best  V $\\times$ U  so far", fontsize=11)
    axa.set_title("(a)  Convergence vs particle count", fontsize=11.5, fontweight="bold")
    axa.set_xlim(0, 152); axa.set_ylim(0.10, 0.97)
    axa.grid(True, ls=":", alpha=0.5); axa.legend(loc="lower right", fontsize=8.8)

    # (b) final V x U vs M, with compute cost
    Ms     = np.array(list(runs.keys()))
    finals = np.array([y[-1] for (_, y, _) in runs.values()])
    evals  = Ms * 152  # ~ M*(T+1) + M(OBL)
    axb.plot(Ms, finals, "-o", color="#2e6da4", lw=2.2, ms=8, zorder=4)
    for M, f in zip(Ms, finals):
        axb.annotate(f"{f:.3f}", (M, f), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.6, color="#2e6da4")
    # highlight sweet spot = best final V x U
    knee = Ms[int(np.argmax(finals))]
    axb.axvline(knee, color="#27ae60", ls=":", lw=1.8)
    axb.text(knee, 0.12, f"sweet spot\nM = {knee}", color="#1e8449",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    axb.axhline(BO_BASELINE, color="#7f8c8d", ls="--", lw=1.5)
    axb.set_xlabel("Particle count  $M$", fontsize=11)
    axb.set_ylabel("Final  V $\\times$ U  (at $T=150$)", fontsize=11)
    axb.set_title("(b)  Final quality vs particle count", fontsize=11.5, fontweight="bold")
    axb.set_xticks(Ms); axb.set_ylim(0.10, 0.97)
    axb.grid(True, ls=":", alpha=0.5)
    # secondary axis: compute cost (total circuit evaluations)
    axc = axb.twinx()
    axc.bar(Ms, evals, width=4.5, color="#f0b27a", alpha=0.35, zorder=1)
    axc.set_ylabel("total circuit evaluations (cost)", fontsize=9.5, color="#b9770e")
    axc.tick_params(axis="y", labelcolor="#b9770e")

    plt.tight_layout()
    p = os.path.join(OUT, "fig3_particle_sweep.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print("fig3 saved:", p, "| M values:", list(runs.keys()))

if __name__ == "__main__":
    make_fig2()
    make_fig3()
