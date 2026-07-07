import re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":16,"font.family":"DejaVu Sans"})
REF="/sessions/friendly-stoic-wright/mnt/uploads/chemistry_constraint_qiskit_4HBA_3HBD_0.log"
t=open(REF,encoding="utf-8",errors="replace").read()
rvu=np.array([float(x) for x in re.findall(r"product_validity_uniqueness \(maximize\):\s*([0-9.]+)",t)])
rba=np.array([float(x) for x in re.findall(r"HBA \(close to 4\):\s*([0-9.]+)",t)])
rbd=np.array([float(x) for x in re.findall(r"HBD \(close to 3\):\s*([0-9.]+)",t)])
rx=np.arange(len(rvu))
d=pd.read_csv("my_s5000_hbahbd.csv")
mx=d["iter_label"].values; mvu=d["product_validity_uniqueness"].values
mba=d["HBA"].values; mbd=d["HBD"].values
def cmax(a): return np.maximum.accumulate(a)
def roll(a,w=9): return pd.Series(a).rolling(w,center=True,min_periods=1).mean().values
C_ME="#1F5FA6"; C_REF="#E08A1E"; C_TGT="#1E7A45"
fig,ax=plt.subplots(1,3,figsize=(17,5.0))
xmax=max(rx.max(),mx.max())
# (a) V x U best-so-far
ax[0].plot(mx,cmax(mvu),color=C_ME,lw=3.0,label="This work (AE-QPSO)")
ax[0].plot(rx,cmax(rvu),color=C_REF,lw=3.0,label="Reference (Chen et al.)")
ax[0].set_ylabel("Best  $V\\times U$  so far",fontsize=18); ax[0].set_title("(a)  Validity $\\times$ Uniqueness",fontsize=18,fontweight="bold")
ax[0].legend(loc="lower right",fontsize=14)
# (b) HBA
ax[1].plot(mx,roll(mba),color=C_ME,lw=3.0); ax[1].plot(rx,roll(rba),color=C_REF,lw=3.0)
ax[1].axhline(4.0,color=C_TGT,ls="--",lw=2.0)
ax[1].text(0.30,4.72,"target = 4",transform=ax[1].get_yaxis_transform(),ha="left",va="center",color=C_TGT,fontsize=14,style="italic")
ax[1].set_ylabel("mean HBA",fontsize=18); ax[1].set_title("(b)  H-bond acceptors (HBA)",fontsize=18,fontweight="bold"); ax[1].set_ylim(1.5,5.2)
# (c) HBD
ax[2].plot(mx,roll(mbd),color=C_ME,lw=3.0); ax[2].plot(rx,roll(rbd),color=C_REF,lw=3.0)
ax[2].axhline(3.0,color=C_TGT,ls="--",lw=2.0)
ax[2].text(0.30,3.62,"target = 3",transform=ax[2].get_yaxis_transform(),ha="left",va="center",color=C_TGT,fontsize=14,style="italic")
ax[2].set_ylabel("mean HBD",fontsize=18); ax[2].set_title("(c)  H-bond donors (HBD)",fontsize=18,fontweight="bold"); ax[2].set_ylim(0.8,4.2)
for a in ax:
    a.set_xlabel("Iteration  $t$",fontsize=18); a.grid(True,ls=":",alpha=0.55); a.tick_params(labelsize=14); a.set_xlim(0,xmax)
    a.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig4_hbahbd_compare.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.savefig("fig4_hbahbd_compare.pdf",bbox_inches="tight",facecolor="white")
print("ok")
