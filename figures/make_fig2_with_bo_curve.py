#!/usr/bin/env python3
"""Fig 2, two panels:
  (a) ablation bar chart (unchanged, final V/U by optimizer config)
  (b) real BO growth curve reconstructed from Chen et al. 2025's own
      released optimization log (results_reference/chen2025_unconditional_9.log),
      NOT a number picked from the paper's text.

Paper's own metric definitions (Sec 2.3, Eq 4-5):
  Validity   = (# valid molecules) / (# total generated molecules)
  Uniqueness = (# unique molecules) / (# valid molecules)
Paper's Fig 3a benchmark methodology text: "We generated 5000 samples
using each method" for the cross-model comparison.
BUT the released log (this file) explicitly logs "# of samples: 10000"
in all 6 of its checkpoint segments -> the log's shot count does not
match the paper's stated benchmark methodology. This script surfaces
that discrepancy directly in the figure caption instead of silently
using an unlabeled number.
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG  = os.path.join(HERE, "..", "results_reference", "chen2025_unconditional_9.log")

def parse_bo_log(path):
    it_re   = re.compile(r"Iteration number: (\d+)")
    val_re  = re.compile(r"validity \(maximize\): ([\d.]+)")
    uniq_re = re.compile(r"uniqueness \(maximize\): ([\d.]+)")
    recs = []
    cur_it, cur_v = None, None
    with open(path, "r") as f:
        for line in f:
            m = it_re.search(line)
            if m:
                cur_it, cur_v = int(m.group(1)), None
                continue
            m = val_re.search(line)
            if m:
                cur_v = float(m.group(1)); continue
            m = uniq_re.search(line)
            if m and cur_v is not None:
                recs.append((cur_it, cur_v, float(m.group(1))))
                cur_v = None
    return recs

recs = parse_bo_log(LOG)
it  = np.array([r[0] for r in recs])
V   = np.array([r[1] for r in recs])
U   = np.array([r[2] for r in recs])
VU  = V * U
VU_best = np.maximum.accumulate(VU)
peak_idx = int(np.argmax(VU))
peak_it, peak_V, peak_U, peak_VU = it[peak_idx], V[peak_idx], U[peak_idx], VU[peak_idx]
n_evals = len(recs)

print(f"[BO log] n_evals={n_evals}  peak: iter={peak_it}  V={peak_V:.4f}  U={peak_U:.4f}  V*U={peak_VU:.4f}")

# ---------------------------------------------------------------- panel (a)
plt.rcParams.update({"font.size": 14.5, "font.family": "DejaVu Sans"})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(17.0, 6.0))

labels = ["BO\n(Chen 2025)", "Pure QPSO\n(pending)", "QPSO\n(+Sobol/OBL)", "QPSO\n(+AE-QTS)", "AE-QPSO\n(this work)"]
Vb = [peak_V*100, None, 94.0, 94.9, 95.9]
Ub = [peak_U*100, None, 96.2, 96.4, 97.0]
x = np.arange(len(labels)); w = 0.30
cV = "#3E7CB1"; cU = "#E08A1E"
for i, (v, u) in enumerate(zip(Vb, Ub)):
    if v is None:
        for off in (-w/2, w/2):
            axa.add_patch(mpatches.Rectangle((x[i]+off-w/2, 85.0), w, 11.5, fill=False,
                          edgecolor="#9aa4ad", lw=1.3, ls=(0, (4, 3)), hatch="///", zorder=3))
        axa.text(x[i], 90.7, "run\npending", ha="center", va="center", fontsize=12,
                 color="#8a8f94", style="italic", fontweight="bold")
        continue
    axa.bar(x[i]-w/2, v, w, color=cV, edgecolor="#22384d", lw=1.0, zorder=3)
    axa.bar(x[i]+w/2, u, w, color=cU, edgecolor="#7a4a10", lw=1.0, zorder=3)
    axa.text(x[i]-w/2, v+0.18, f"{v:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold", color="#1b2631")
    axa.text(x[i]+w/2, u+0.18, f"{u:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold", color="#1b2631")
axa.set_ylabel("Percentage (%)", fontsize=15); axa.set_ylim(85.0, 99.0)
axa.set_xticks(x); axa.set_xticklabels(labels, fontsize=13)
axa.set_title("(a)  Final validity / uniqueness by configuration\n(9 heavy atoms, $M{=}64$, $T{=}150$, 5000 shots)",
              fontsize=14.5, fontweight="bold", pad=10)
hV = mpatches.Patch(color=cV, label="Validity  $V$"); hU = mpatches.Patch(color=cU, label="Uniqueness  $U$")
axa.legend(handles=[hV, hU], loc="upper left", fontsize=12.5, framealpha=0.95)
axa.grid(axis="y", ls=":", alpha=0.55, zorder=0); axa.spines[["top", "right"]].set_visible(False)
axa.tick_params(axis="y", labelsize=12.5)

# ---------------------------------------------------------------- panel (b)
axb.plot(it, VU_best, color="#1e8449", lw=2.4, label="BO best-so-far  $V{\\times}U$")
axb.scatter([peak_it], [peak_VU], color="#c0392b", zorder=5, s=45, label=f"peak @ iter {peak_it}: {peak_VU:.4f}")
axb.axhline(peak_VU, color="#c0392b", ls=":", lw=1.3, alpha=0.7)
axb.set_xlabel("BO iteration (as logged)", fontsize=15)
axb.set_ylabel("Best  $V \\times U$  so far", fontsize=15)
axb.set_title("(b)  Chen et al. 2025 -- reconstructed BO growth curve\n"
              "(from released log, NOT digitized from a paper figure)",
              fontsize=14.5, fontweight="bold", pad=10)
axb.grid(True, ls=":", alpha=0.5); axb.spines[["top", "right"]].set_visible(False)
axb.legend(loc="lower right", fontsize=12)
axb.text(0.02, 0.03,
    f"log: \"# of samples: 10000\" (all 6 checkpoint segments)\n"
    f"total evaluations logged: {n_evals}  (6 checkpointed restarts,\n"
    f"iteration numbering continuous, no resets observed)\n"
    f"NOTE: paper text (Fig. 3a benchmark) states 5000 samples were\n"
    f"used for the cross-model comparison -- the released log itself\n"
    f"used 10000. This mismatch is in the source paper, not in our\n"
    f"re-analysis. Eq.5 uniqueness = unique / valid (not / total).",
    transform=axb.transAxes, fontsize=9.3, va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#c9a227", lw=1.0))

plt.tight_layout()
out_png = os.path.join(HERE, "fig2_VU_bars.png")
out_pdf = os.path.join(HERE, "fig2_VU_bars.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print("saved", out_png, out_pdf)
