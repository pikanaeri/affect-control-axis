#!/usr/bin/env python
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[1]
BATT=str(_REPO / "results" / "behavior_battery_gemma-3-12b-it.json")
OUT =str(_REPO / "figures")
BLUE="#0072B2"; ORANGE="#E69F00"; GRAY="#8C8C8C"; GREEN="#009E73"; INK="#1a1a1a"; MUTED="#666666"; LGRAY="#D9D9D9"
plt.rcParams.update({"savefig.dpi":300,"figure.facecolor":"white","font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":13,"axes.titleweight":"bold","axes.edgecolor":"#B0B0B0","axes.linewidth":0.9,
    "axes.spines.top":False,"axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.grid":True,"grid.color":"#ECECEC","axes.axisbelow":True,"legend.frameon":False})

d=json.load(open(BATT))
NM={"interpretation_bias":"Interpretation bias","risk_estimation":"Risk estimation",
    "prosocial_helping":"Prosocial helping","moral_harshness":"Moral harshness",
    "confidence":"Confidence","sentiment_outlook":"Sentiment / outlook"}
rows=sorted(d["behaviors"], key=lambda r:r["effect"])   # ascending so largest on top
labels=[NM[r["name"]] for r in rows]
steer =[r["effect"] for r in rows]
prime =[r["prime_effect"] for r in rows]
rand  =[r["rand_effect"] for r in rows]
y=np.arange(len(rows))

fig,(axA,axB)=plt.subplots(1,2,figsize=(10.6,4.9),gridspec_kw={"width_ratios":[1.25,1]})
# --- Panel A: causal (steering) + random control ---
axA.axvline(0,color="#999999",lw=1)
axA.barh(y, steer, height=0.52, color=BLUE, zorder=2, label="steering effect")
axA.scatter(rand, y, marker="x", s=65, color=GRAY, linewidth=2.2, zorder=3, label="random control")
for yi,v in zip(y,steer):
    axA.text(v+0.7, yi, "%.0f"%v, va="center", ha="left", fontsize=9.5, fontweight="bold", color=INK)
axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=11)
axA.set_xlim(min(-13,min(rand)-3), max(steer)+7)
axA.set_xlabel("shift, first-token logit")
axA.set_title("Causal: steering the affect axis (+a vs \u2212a)", fontsize=12)
axA.legend(loc="lower right", fontsize=9)
# --- Panel B: input manipulation (text-affect prime) ---
axB.axvline(0,color="#999999",lw=1)
axB.barh(y, prime, height=0.52, color=ORANGE, zorder=2)
for yi,v in zip(y,prime):
    axB.text(v+1.0, yi, "%.0f"%v, va="center", ha="left", fontsize=9.5, fontweight="bold", color=INK)
axB.set_yticks(y); axB.set_yticklabels([])
axB.set_xlim(0, max(prime)+9)
axB.set_xlabel("shift, first-token logit")
axB.set_title("Input: text-affect prime (distress vs positive)", fontsize=12)
fig.suptitle("Affect shifts a battery of judgments \u2014 all 6 move in the predicted direction",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.text(0.5,0.005,
    "Gemma-3-12B, text-valence axis. Every behavior moves in the human-predicted direction under BOTH a causal "
    "steer (left, vs a near-zero random control) and a text-affect prime (right). Risk estimation is the one whose "
    "steering effect is not clearly above its random control \u2014 its prime effect (+31) carries it.",
    ha="center", va="bottom", fontsize=8.0, color=MUTED, wrap=True)
fig.subplots_adjust(bottom=0.22, wspace=0.08, top=0.88)
for ext in ("png","pdf"): fig.savefig(os.path.join(OUT,f"fig10_affect_battery.{ext}"), bbox_inches="tight")
print("wrote fig10_affect_battery")

# quick bootstrap CI for reporting
eff=np.array(steer); rng=np.random.default_rng(0)
bs=[float(np.mean(rng.choice(eff,len(eff),replace=True))) for _ in range(5000)]
print("mean steering effect %.2f  95%% CI [%.2f, %.2f] | sign %d/%d positive | mean|rand| %.2f"%(
    eff.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5),
    int((eff>0).sum()), len(eff), np.mean(np.abs(rand))))
