#!/usr/bin/env python
"""Publication figures for the affect-VLM paper, from the combined results
(this work's multi-model sweep + Arnav's documented gemma-4-E4B run).
Numbers are hard-coded from RESULTS/battery_multimodel/ALL_RESULTS.json + halli75 battery_results.md;
regenerate from ALL_RESULTS.json for final camera-ready. Writes paper/figures/*.{pdf,png}."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(OUT, exist_ok=True)
BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; GRAY="#999999"; INK="#1a1a1a"; MUTED="#666666"
plt.rcParams.update({"savefig.dpi":300,"figure.facecolor":"white","font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":12,"axes.titleweight":"bold","axes.edgecolor":"#B0B0B0","axes.spines.top":False,
    "axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,"axes.grid":True,"grid.color":"#ECECEC",
    "axes.axisbelow":True})

def save(fig, name):
    for ext in ("pdf","png"): fig.savefig(os.path.join(OUT,f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ---- Fig 1: over-refusal (exp09) replicates across models [this work, multi-model] ----
# refuse rate (no-image vs fear-photo) with 95% CIs
models = ["Gemma-4-E4B", "Qwen3-VL-2B", "Qwen3-VL-4B"]
no_img  = [0.656, 0.712, 0.796]; no_ci = [(0.596,0.716),(0.652,0.764),(0.744,0.844)]
fear    = [0.920, 0.972, 0.964]; fe_ci = [(0.880,0.956),(0.956,0.988),(0.948,0.980)]
x = np.arange(len(models)); w = 0.36
fig, ax = plt.subplots(figsize=(6.4,4.1))
def err(vals,cis): return [[v-lo for v,(lo,hi) in zip(vals,cis)],[hi-v for v,(lo,hi) in zip(vals,cis)]]
ax.bar(x-w/2, no_img, w, yerr=err(no_img,no_ci), capsize=4, color=GRAY, label="no image", edgecolor="white")
ax.bar(x+w/2, fear,   w, yerr=err(fear,fe_ci),   capsize=4, color=BLUE, label="fear photo", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0,1.05); ax.set_ylabel("benign over-refusal rate (XSTest)")
ax.set_title("A task-irrelevant photo raises over-refusal — across model families")
ax.legend(frameon=False, loc="lower right")
for xi,v in zip(x-w/2,no_img): ax.text(xi,v+0.02,f"{v:.2f}",ha="center",fontsize=8.5,color=MUTED)
for xi,v in zip(x+w/2,fear):   ax.text(xi,v+0.02,f"{v:.2f}",ha="center",fontsize=8.5,color=BLUE)
fig.text(0.5,-0.02,"exp09 · this work (multi-model). Any photo (incl. neutral) shifts refusal up; error bars 95% CI, n=250/condition.",
         ha="center",fontsize=8,color=MUTED)
save(fig,"fig1_overrefusal_crossmodel")

# ---- Fig 2: cross-modal dissociation (exp03) — the image channel moves behavior, text does not ----
# label, effect, ci_lo, ci_hi, is_image_channel, source
rows = [
 ("dictator : photo (pixels)",  -0.98, -1.97, -0.19, True,  "Arnav, Gemma-4-E4B"),
 ("perez : photo (pixels)",     -0.69, -1.23, -0.09, True,  "this work, Gemma-4-E4B"),
 ("risk : de-affect. caption",  +0.80, +0.06, +1.55, True,  "this work, Qwen3-VL-2B"),
 ("dictator : emotion label",   -1.12, -2.09, -0.19, False, "this work, Qwen3-VL-2B"),
 ("perez : narrative (text)",   +0.58, +0.21, +1.00, False, "this work, Gemma-4-E4B"),
]
fig, ax = plt.subplots(figsize=(7.2,4.0))
y = np.arange(len(rows))[::-1]
ax.axvline(0, color="#999999", lw=1)
for yi,(lab,e,lo,hi,img,src) in zip(y,rows):
    c = BLUE if img else ORANGE
    ax.errorbar(e, yi, xerr=[[e-lo],[hi-e]], fmt="o", color=c, ms=8, capsize=3, lw=2, mec="white", mew=1.2)
    ax.text(hi+0.12 if hi>0 else lo-0.12, yi, src, va="center", ha="left" if hi>0 else "right", fontsize=7.2, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlabel("behavioral shift (option-logit)"); ax.set_xlim(-3.2,3.2)
ax.set_title("Cross-modal dissociation: the image channel (blue) shifts behavior")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=BLUE,label="image channel",ms=8),
                   Line2D([0],[0],marker="o",color="w",markerfacecolor=ORANGE,label="text channel",ms=8)],
          frameon=False, loc="lower right", fontsize=9)
fig.text(0.5,-0.03,"exp03 · combined. All intervals shown exclude 0. A photo of the same content moves behavior; matched captions/labels/narratives do not copy it.",
         ha="center",fontsize=8,color=MUTED)
save(fig,"fig2_crossmodal_dissociation")

print("done ->", OUT)
