import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import matplotlib.patches as mpatches
plt.rcParams.update({"font.size":15,"font.family":"DejaVu Sans"})
# Values below are read directly from each run's final log line, all M=64,
# T=150, 5000 shots, seed=0 (except BO baseline, see chen2025 reference log):
#   Pure QPSO   <- results_qpso_nosobol/unconditional_9_qpso_nosobol_M64T150.log
#                  (no Sobol, no OBL, no AE-QTS, no VU-decouple)
#   +Sobol      <- results_qpso_pure/unconditional_9_qpso_pure_M64T150.log
#                  (Sobol init only; OBL/AE-QTS/VU-decouple all off)
#   +AE-QTS     <- results_qpso_aeqts/unconditional_9_qpso_aeqts_M64T150.log
#                  (AE-QTS signed mbest only; no Sobol/OBL/VU-decouple)
#   AE-QPSO     <- results_v8/unconditional_9_ae_v8.log (all four on)
# Previous hardcoded values [94.9,96.4] mislabeled "+AE-QTS" were actually the
# Pure QPSO run's numbers; the real AE-QTS-only run (98.06/97.80) was never
# plotted. Fixed here.
labels=["BO\n(Chen 2025)","Pure QPSO","QPSO\n(+Sobol)","QPSO\n(+AE-QTS)","AE-QPSO\n(this work)"]
V=[95.5, 94.90, 94.04, 98.06, 95.94]
U=[92.5, 96.35, 96.19, 97.80, 97.04]
x=np.arange(len(labels)); w=0.30
cV="#3E7CB1"; cU="#E08A1E"
fig,ax=plt.subplots(figsize=(11.6,5.7))
for i,(v,u) in enumerate(zip(V,U)):
    if v is None:  # placeholder: dashed empty bars + note
        for off,c in [(-w/2,cV),(w/2,cU)]:
            ax.add_patch(mpatches.Rectangle((x[i]+off-w/2,85.0),w,11.5,fill=False,
                         edgecolor="#9aa4ad",lw=1.3,ls=(0,(4,3)),hatch="///",zorder=3))
        ax.text(x[i],90.7,"run\npending",ha="center",va="center",fontsize=13,color="#8a8f94",
                style="italic",fontweight="bold")
        continue
    b1=ax.bar(x[i]-w/2,v,w,color=cV,edgecolor="#22384d",lw=1.0,zorder=3)
    b2=ax.bar(x[i]+w/2,u,w,color=cU,edgecolor="#7a4a10",lw=1.0,zorder=3)
    ax.text(x[i]-w/2,v+0.18,f"{v:.1f}",ha="center",va="bottom",fontsize=14.5,fontweight="bold",color="#1b2631")
    ax.text(x[i]+w/2,u+0.18,f"{u:.1f}",ha="center",va="bottom",fontsize=14.5,fontweight="bold",color="#1b2631")
ax.set_ylabel("Percentage (%)",fontsize=16); ax.set_ylim(85.0,99.0)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=14.5)
ax.set_title("Ablation: validity and uniqueness by optimizer configuration\n(9 heavy atoms, $M{=}64$, $T{=}150$, 5000 shots)",
             fontsize=16,fontweight="bold",pad=12)
hV=mpatches.Patch(color=cV,label="Validity  $V$"); hU=mpatches.Patch(color=cU,label="Uniqueness  $U$")
ax.legend(handles=[hV,hU],loc="upper left",fontsize=14,framealpha=0.95)
ax.grid(axis="y",ls=":",alpha=0.55,zorder=0); ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis="y",labelsize=13.5)
plt.tight_layout()
plt.savefig("fig2_VU_bars.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.savefig("fig2_VU_bars.pdf",bbox_inches="tight",facecolor="white")
print("saved fig2")
