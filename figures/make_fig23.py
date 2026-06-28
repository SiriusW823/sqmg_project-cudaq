#!/usr/bin/env python3
"""Fig 2 (V and U compared separately, 4 curves) + Fig 3 (convergence + circuit-resource table).
Run on DGX111. Reads result CSVs; outputs PNGs into figures/."""
import os, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

ROOT = os.path.expanduser("~/sqmg_project-cudaq")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
BO_V, BO_U = 0.955, 0.925   # Chen et al. 2025 (paper data)

def incumbent(csv):
    """Per-iteration V and U of the best-fitness-so-far solution."""
    df = pd.read_csv(csv)
    bestf=-1; bv=bu=np.nan; rows=[]
    for _,r in df.iterrows():
        if r["fitness"]>bestf:
            bestf=r["fitness"]; bv=r["validity"]; bu=r["uniqueness"]
        rows.append((r["qpso_iter"],bv,bu))
    g=pd.DataFrame(rows,columns=["it","V","U"]).groupby("it").last()
    return g.index.values, g["V"].values, g["U"].values, int(df["qpso_iter"].max())

SERIES = [   # label, dir, csv-glob, color, style
 ("AE-QPSO (this work)","results_v8","unconditional_9_ae_v8.csv","#2e6da4","-"),
 ("QPSO (no AE-QTS)","results_qpso_pure","unconditional_9_qpso_pure_M64T150.csv","#c0392b","-"),
 ("QPSO, no Sobol/OBL","results_qpso_nosobol","unconditional_9_qpso_nosobol_M64T150.csv","#7f8c8d","--"),
]

def make_fig2():
    fig,(axV,axU)=plt.subplots(1,2,figsize=(12.4,5.0))
    miny=1.0
    for label,d,fn,col,ls in SERIES:
        path=os.path.join(ROOT,d,fn)
        if not os.path.exists(path):
            axV.plot([],[],color=col,ls=ls,label=label+"  [run pending]"); axU.plot([],[],color=col,ls=ls,label=label+"  [pending]"); continue
        x,V,U,done=incumbent(path)
        tag="" if done>=150 else f" [iter {done}/150]"
        axV.plot(x,V,color=col,ls=ls,lw=2.2,label=f"{label}: {V[-1]:.3f}{tag}")
        axU.plot(x,U,color=col,ls=ls,lw=2.2,label=f"{label}: {U[-1]:.3f}{tag}")
        miny=min(miny,np.nanmin(V[2:]),np.nanmin(U[2:]))
    axV.axhline(BO_V,color="#1e8449",ls=":",lw=1.8); axV.text(150,BO_V+0.003,f"BO baseline V={BO_V}",ha="right",va="bottom",fontsize=8.5,color="#1e8449",style="italic")
    axU.axhline(BO_U,color="#1e8449",ls=":",lw=1.8); axU.text(150,BO_U+0.003,f"BO baseline U={BO_U}",ha="right",va="bottom",fontsize=8.5,color="#1e8449",style="italic")
    lo=max(0.40, np.floor(miny*20)/20-0.02)
    for ax,t,yl in [(axV,"(a)  Validity  V",("Validity  V")),(axU,"(b)  Uniqueness  U",("Uniqueness  U"))]:
        ax.set_xlabel("QPSO iteration  t",fontsize=11); ax.set_ylabel(yl,fontsize=11)
        ax.set_title(t,fontsize=12,fontweight="bold"); ax.set_xlim(0,152); ax.set_ylim(lo,1.005)
        ax.grid(True,ls=":",alpha=0.5); ax.legend(loc="lower right",fontsize=9)
    fig.suptitle("Optimizer comparison: validity and uniqueness  (9 heavy atoms, M=64, T=150, 5000 shots)",fontsize=12.5,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    p=os.path.join(OUT,"fig2_VU_comparison.png"); plt.savefig(p,dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
    print("fig2 saved",p)

def conv(csv):
    df=pd.read_csv(csv); g=df.groupby("qpso_iter")["gbest_fitness"].max(); return g.index.values,g.values,int(df["qpso_iter"].max())

def make_fig3():
    runs={}
    v8=os.path.join(ROOT,"results_v8","unconditional_9_ae_v8.csv")
    if os.path.exists(v8): runs[64]=conv(v8)
    for d in sorted(glob.glob(os.path.join(ROOT,"results_sweep_M*"))):
        try: M=int(os.path.basename(d).split("results_sweep_M")[1])
        except: continue
        cs=glob.glob(os.path.join(d,"*.csv"))
        if cs: runs[M]=conv(cs[0])
    runs=dict(sorted(runs.items()))
    fig,(axa,axb)=plt.subplots(1,2,figsize=(12.6,5.1))
    cmap=plt.cm.viridis(np.linspace(0.05,0.9,len(runs)))
    for c,(M,(x,y,done)) in zip(cmap,runs.items()):
        axa.plot(x,y,lw=2.0,color=c,label=f"M={M}"+("" if done>=150 else f" [{done}/150]"))
    axa.axhline(0.8834,color="#7f8c8d",ls="--",lw=1.5); axa.text(150,0.8834+0.005,"BO baseline",ha="right",va="bottom",fontsize=8.2,color="#5d6d7e",style="italic")
    axa.set_xlabel("QPSO iteration  t",fontsize=11); axa.set_ylabel("Best  V x U  so far",fontsize=11)
    axa.set_title("(a)  Convergence vs particle count",fontsize=12,fontweight="bold")
    axa.set_xlim(0,152); axa.set_ylim(0.10,0.97); axa.grid(True,ls=":",alpha=0.5); axa.legend(loc="lower right",fontsize=8.8)
    # (b) circuit-resource table
    axb.axis("off")
    axb.set_title("(b)  Circuit-resource scaling vs heavy atoms",fontsize=12,fontweight="bold")
    Ns=list(range(2,10))
    rows=[[str(N), str(N*(N+1)), str(2*N+2), str(8+(N-2)*(N+3)*3//2)] for N in Ns]
    col=["heavy\natoms","qubits:\nstatic circuit","qubits:\ndynamic circuit","params:\ndynamic circuit"]
    tb=axb.table(cellText=rows,colLabels=col,cellLoc="center",loc="center",
                 colColours=["#2e6da4"]*4)
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1,1.55)
    for (r,c0),cell in tb.get_celld().items():
        if r==0:
            cell.set_text_props(color="white",fontweight="bold")
        if r-1>=0 and Ns[r-1]==9:
            cell.set_facecolor("#eaf1fb"); cell.set_text_props(fontweight="bold")
    axb.text(0.5,0.04,"static = N(N+1),  dynamic = 2N+2,  params = 8+(N-2)(N+3)*3/2;  N=9 is this work",
             transform=axb.transAxes,ha="center",fontsize=8.2,style="italic",color="#555")
    plt.tight_layout()
    p=os.path.join(OUT,"fig3_sweep_table.png"); plt.savefig(p,dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
    print("fig3 saved",p,"| M:",list(runs.keys()))

if __name__=="__main__":
    make_fig2(); make_fig3()
