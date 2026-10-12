#!/usr/bin/env python3
"""Trace the BO baseline numbers back to the upstream QMG repository.

Reads the published BO trajectory and molecule sample of Chen et al. 2025
(https://github.com/PEESEgroup/QMG, commit 0f514ad) and prints the figures this
project has quoted as "the BO baseline":

  * 0.902  (V 94.2 / U 95.7)  -- the v1 preprint's BO bar
  * 0.8834 (V 95.5 / U 92.5)  -- the constant hard-coded in qpso_optimizer_ae.py

Usage:
    git clone https://github.com/PEESEgroup/QMG.git /path/to/qmg_upstream
    python benchmark/trace_upstream_bo.py /path/to/qmg_upstream

Needs only pandas (pickle holds a plain {smiles: count} dict).
"""
import os
import pickle
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/qmg_upstream")
d = os.path.join(root, "results_chemistry_constraint_bo")

# ── 1. The BO trajectory for the unconditional 9-heavy-atom task ──────────────
df = pd.read_csv(os.path.join(d, "unconditional_9_4.csv")).copy()
df["vu"] = df["validity"] * df["uniqueness"]
n = len(df)
gm = df["generation_method"].value_counts().to_dict()
shots_ok = (df["validity"] * 10000).round(6).mod(1).eq(0).all()
print(f"unconditional_9_4.csv: {n} evaluations  {gm}")
print(f"  validity is a multiple of 1/10,000 on every row: {shots_ok}  "
      f"(=> 10,000 shots per evaluation, the constrained_bo.py default)")
best = df.loc[df["vu"].idxmax()]
print(f"  max V×U during BO: trial {int(best.trial_index)}  "
      f"V={best.validity:.4f}  U={best.uniqueness:.4f}  V×U={best.vu:.4f}")
# A paper reports one-decimal percentages; multiplying those back gives the constant.
rv, ru = (float(f"{x * 100:.1f}") / 100 for x in (best.validity, best.uniqueness))
print(f"  as percentages {rv * 100:.1f}% × {ru * 100:.1f}% = {rv * ru:.4f}"
      f"   <- the hard-coded 0.8834 (exact product {best.vu:.4f})")
bsf = df["vu"].cummax()
print("  best-so-far:", ", ".join(f"{k}:{bsf.iloc[k - 1]:.4f}"
                                  for k in (205, 403, 501, 603) if k <= n))

# ── 2. The published molecule sample behind Chen et al. Fig. 3a ───────────────
p = os.path.join(d, "sample_unconditional_9_4_501.pkl")
smiles = pickle.load(open(p, "rb"))
total = sum(smiles.values())
valid = total - smiles.get(None, 0)
V = valid / total
U = (len(smiles) - (1 if None in smiles else 0)) / valid
row = df.loc[df.trial_index == 501].iloc[0]
print(f"\nsample_unconditional_9_4_501.pkl: {total:,} molecules from trial 501's parameters")
print(f"  V={V:.4f}  U={U:.4f}  V×U={V * U:.4f}   <- the preprint's 'BO 94.2 / 95.7 / 90.2'")
print(f"  (same parameters during BO, 10,000 shots: V={row.validity:.4f} "
      f"U={row.uniqueness:.4f} V×U={row.vu:.4f})")
