#!/usr/bin/env python
"""Figure for the generation-tone experiment.
Reads results/generation_tone_*.json (or a path arg) -> figures/fig_generation_tone.{png,pdf}.
Three scorers faceted (own x-scale each) so the 'same effect across measures' reads at a glance."""
import os, json, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 1:
    PATH = sys.argv[1]
else:
    hits = sorted(glob.glob(os.path.join(REPO, "results", "generation_tone_*.json")))
    PATH = hits[0] if hits else os.path.join(REPO, "results", "generation_tone_gemma-3-12b-it.json")
OUT = os.path.join(REPO, "figures"); os.makedirs(OUT, exist_ok=True)

BLUE="#0072B2"; GREEN="#009E73"; ORANGE="#E69F00"; INK="#1a1a1a"; MUTED="#666666"
plt.rcParams.update({"savefig.dpi":300,"figure.facecolor":"white","font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":12,"axes.titleweight":"bold","axes.edgecolor":"#B0B0B0","axes.linewidth":0.9,
    "axes.spines.top":False,"axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.grid":True,"grid.color":"#ECECEC","axes.axisbelow":True})

d = json.load(open(PATH))
EFF, AG, POW = d["effects"], d.get("agreement",{}), d.get("power",{})
contrasts = [
 ("image_positive_vs_distress",   "Image  positive − distress"),
 ("mediation_restore_vs_distress","Mediation  restore − distress"),
 ("steer_pos_vs_neg",             "Steer  −a(pos) − +a(neg)"),
 ("random_vs_baseline",           "Random control"),
]
scorers = [("VADER",BLUE),("RoBERTa",GREEN),("axis",ORANGE)]
y = np.arange(len(contrasts))[::-1]

fig, axs = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
for ax,(sc,col) in zip(axs, scorers):
    ax.axvline(0, color="#999999", lw=1)
    for yi,(key,_) in zip(y, contrasts):
        e = EFF.get(key,{}).get(sc)
        if not e or e.get("diff") is None or e["diff"]!=e["diff"]: continue
        lo = e["diff"]-e["ci_lo"]; hi = e["ci_hi"]-e["diff"]
        filled = bool(e.get("sig"))
        ax.errorbar(e["diff"], yi, xerr=[[lo],[hi]], fmt="o", color=col, ms=8, capsize=3, lw=2,
                    mec="white", mew=1.2, mfc=(col if filled else "white"))
    ax.set_title(sc)
    ax.set_xlabel("valence effect")
    ax.margins(x=0.25)
axs[0].set_yticks(y); axs[0].set_yticklabels([c[1] for c in contrasts], fontsize=10.5)
fig.suptitle("Visual affect shifts open-generation tone — same direction across three scorers",
             fontsize=13.5, fontweight="bold", y=1.0)
foot = ("Filled = 95%% CI excludes 0.   Scorer agreement (pooled r): VADER~RoBERTa %.2f · VADER~axis %.2f · RoBERTa~axis %.2f.   "
        "Power (VADER image contrast): Cohen's d = %.2f, ~%d generations/group for 80%%.") % (
        AG.get("vader_roberta",float('nan')), AG.get("vader_axis",float('nan')), AG.get("roberta_axis",float('nan')),
        POW.get("cohens_d",float('nan')), POW.get("n_per_group_for_80pct",0))
fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8.2, color=MUTED, wrap=True)
fig.subplots_adjust(bottom=0.20, wspace=0.12, top=0.87)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT, f"fig_generation_tone.{ext}"), bbox_inches="tight")
print("wrote fig_generation_tone from", os.path.basename(PATH))
