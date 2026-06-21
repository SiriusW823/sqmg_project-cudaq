#!/usr/bin/env python3
"""Figure 1 - AE-SOQPSO quantum-classical hybrid workflow (paper method flowchart)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_SETUP="#34495e"; C_OPT="#2e6da4"; C_OPT_BG="#eaf2fb"
C_QEVAL="#c0622a"; C_QEVAL_BG="#fbeee3"; C_OUT="#2e7d4f"; C_LOOP="#8e44ad"; TXT="#1b2631"

fig, ax = plt.subplots(figsize=(9.6, 13.8))
ax.set_xlim(0,100); ax.set_ylim(0,152); ax.axis("off")

def box(x,y,w,h,title,body,fc,ec,tc="white",fs=10.5,bfs=9.0,align="center"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2.2",
                 linewidth=1.6,edgecolor=ec,facecolor=fc,zorder=3))
    cx=x+w/2; ha="center" if align=="center" else "left"; bx=cx if align=="center" else x+3.5
    if title:
        ax.text(cx,y+h-3.6,title,ha="center",va="top",fontsize=fs,fontweight="bold",color=tc,zorder=4)
        if body:
            ax.text(bx,y+(h-5.0)/2+0.4,body,ha=ha,va="center",fontsize=bfs,color=tc,zorder=4,linespacing=1.5)
    elif body:
        ax.text(bx,y+h/2,body,ha=ha,va="center",fontsize=bfs,color=tc,zorder=4,linespacing=1.5)

def arrow(x1,y1,x2,y2,color=TXT,lw=2.0,style="-|>",ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=18,
                 lw=lw,color=color,zorder=2,linestyle=ls))

def alabel(x,y,t,color=TXT,fs=8.2):
    ax.text(x,y,t,ha="center",va="center",fontsize=fs,color=color,fontstyle="italic",zorder=5,
            bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.9))

ax.text(50,151.0,"Quantum-Classical Hybrid Workflow: AE-SOQPSO for 9-Heavy-Atom Molecular Generation",
        ha="center",va="top",fontsize=12.4,fontweight="bold",color=TXT)

box(11,134.0,78,13.2,"Problem Setup",
    "Unconditional generation of drug-like molecules (9 heavy atoms: C, N, O)\n"
    "Generator: 20-qubit chemistry-inspired dynamic quantum circuit, D = 134 trainable weights\n"
    "Objective:  maximize  fitness = V x U   (validity x uniqueness)",
    C_SETUP,C_SETUP,fs=10.6,bfs=8.7)
arrow(50,134.0,50,131.2)

ax.add_patch(FancyBboxPatch((6,59.0),88,71.5,boxstyle="round,pad=0.6,rounding_size=2.2",
             linewidth=1.7,edgecolor=C_OPT,facecolor=C_OPT_BG,zorder=1))
ax.text(8.5,128.9,"Classical Optimizer  -  AE-SOQPSO  (M = 64 particles, T = 150 iterations)",
        ha="left",va="top",fontsize=10.6,fontweight="bold",color=C_OPT,zorder=4)

box(11,116.0,37,10.0,"1  Sobol Initialization",
    "Owen-scrambled Sobol sequence\nM=64 particles in [0,1]^134",
    "white",C_OPT,tc=TXT,fs=9.6,bfs=8.2)
box(52,116.0,37,10.0,"2  OBL Phase 0",
    "opposition x' = 1 - x\nkeep max-fitness of (x, x')",
    "white",C_OPT,tc=TXT,fs=9.6,bfs=8.2)
arrow(48,121.0,52,121.0)
arrow(29.5,116.0,29.5,112.2); arrow(70.5,116.0,70.5,112.2)
ax.text(50,111.0,"iterate  t = 1 ... T",ha="center",va="center",fontsize=8.8,
        color=C_LOOP,fontstyle="italic",fontweight="bold")

box(11,97.0,80,12.6,"3  Attractor & mbest update",
    "-  standard QPSO mean-best:  mbest = mean_i(pbest_i)\n"
    "-  AE-QTS U-shaped harmonic correction:  + r*Sum_k (best_k - worst_k)/k   [Amplitude-Ensemble]\n"
    "-  V-U decoupled pull toward best-V-ever / best-U-ever positions",
    "white",C_OPT,tc=TXT,fs=9.8,bfs=8.3,align="left")
arrow(50,97.0,50,94.0)
box(11,82.0,80,11.0,"4  QPSO position update + diversity",
    "-  x_i = p_i +/- a*|mbest - x_i|*ln(1/u),   a cosine-annealed 1.2 -> 0.3\n"
    "-  Cauchy heavy-tail mutation (prob 0.15)   -  paired exploration step",
    "white",C_OPT,tc=TXT,fs=9.8,bfs=8.3,align="left")
arrow(50,82.0,50,79.0)
box(11,68.5,80,9.5,"5  Stagnation control",
    "stagnation_limit = 12  ->  reinit 25% worst particles   -   mode-collapse recycling",
    "white",C_OPT,tc=TXT,fs=9.8,bfs=8.3,align="left")

arrow(50,68.5,50,54.5,lw=2.4)
alabel(50,57.0,"candidate weight vectors  { w_i }")

ax.add_patch(FancyBboxPatch((6,13.5),88,41.0,boxstyle="round,pad=0.6,rounding_size=2.2",
             linewidth=1.7,edgecolor=C_QEVAL,facecolor=C_QEVAL_BG,zorder=1))
ax.text(8.5,52.9,"Quantum + GPU Evaluation  -  parallel subprocess pool (8 x V100, CUDA-Q 0.7.1)",
        ha="left",va="top",fontsize=10.6,fontweight="bold",color=C_QEVAL,zorder=4)

box(11,42.0,80,7.6,"6  Batch evaluate",
    "split M particles into ceil(M/8) rounds  -  one subprocess per GPU (isolated CUDA context)",
    "white",C_QEVAL,tc=TXT,fs=9.8,bfs=8.3,align="left")
arrow(50,42.0,50,38.6)
for i,gx in enumerate([14,40.5,67]):
    lab = f"GPU {i}" if i<2 else "GPU 7"
    box(gx,30.6,22,7.4,lab,"worker_eval.py","white",C_QEVAL,tc=TXT,fs=8.8,bfs=7.8)
ax.text(50,27.2,"... 8 workers in parallel ...",ha="center",va="center",fontsize=8.0,
        color=C_QEVAL,fontstyle="italic")
for gx in [25,51.5,78]:
    arrow(gx,30.6,50,25.6,lw=1.3)
box(11,16.8,80,9.4,"7  Per-worker quantum sampling -> scoring",
    "cudaq.sample(_qmg_n9, 5000 shots):  20-qubit dynamic circuit, 90 mid-circuit measurements\n"
    "bitstrings -> SMILES (RDKit) -> validity V, uniqueness U -> fitness = V x U",
    "white",C_QEVAL,tc=TXT,fs=9.6,bfs=8.2,align="left")

arrow(91,21.5,95.6,21.5,color=C_LOOP,lw=2.2)
arrow(95.6,21.5,95.6,104.0,color=C_LOOP,lw=2.2,style="-")
arrow(95.6,104.0,91,104.0,color=C_LOOP,lw=2.2)
ax.text(97.0,63,"fitness -> update pbest / gbest",rotation=90,ha="center",va="center",
        fontsize=8.6,color=C_LOOP,fontstyle="italic",fontweight="bold")

arrow(50,13.5,50,9.8,lw=2.4)
box(20,0.6,60,8.8,"Output",
    "g-best weights  ->  best molecules     (V8:  V x U = 0.9310,  V = 0.959,  U = 0.970)",
    C_OUT,C_OUT,fs=10.4,bfs=8.7)

plt.tight_layout(pad=0.4)
plt.savefig("fig1_method_flowchart.png",dpi=300,bbox_inches="tight",facecolor="white")
print("saved")
