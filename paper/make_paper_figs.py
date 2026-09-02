#!/usr/bin/env python
"""Publication figures from the final run (affect_results_09012026).
Numbers hard-coded from RESULTS/*/exp09|exp05/results.json + static_null/*.json.
Writes paper/figures/*.{pdf,png}."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(OUT, exist_ok=True)
GRAY="#9aa0a6"; BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; RED="#c0392b"; MUTED="#666666"
plt.rcParams.update({"savefig.dpi":300,"figure.facecolor":"white","font.family":"DejaVu Sans","font.size":11,
    "axes.titlesize":12,"axes.titleweight":"bold","axes.edgecolor":"#B0B0B0","axes.spines.top":False,
    "axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,"axes.grid":True,"grid.color":"#ECECEC",
    "axes.axisbelow":True,"axes.grid.axis":"y"})
def save(fig,name):
    for e in ("pdf","png"): fig.savefig(os.path.join(OUT,f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig); print("wrote",name)

# ---- Fig 1: over-refusal, full emotion sweep, 3 models (neutral is elevated too) ----
MODELS=["Gemma-4-E4B","Qwen3-VL-2B","Qwen3-VL-4B"]
ARMS=["no image","fear","anger","sad","happy","neutral"]
R={"Gemma-4-E4B":[0.656,0.920,0.936,0.908,0.924,0.944],
   "Qwen3-VL-2B":[0.712,0.972,0.968,0.964,0.948,0.968],
   "Qwen3-VL-4B":[0.796,0.964,0.948,0.944,0.940,0.956]}
cols=[GRAY,BLUE,BLUE,BLUE,BLUE,GREEN]   # no-image gray; emotions blue; neutral green (a non-emotional photo)
fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,m in zip(axs,MODELS):
    ax.bar(range(len(ARMS)), R[m], color=cols, edgecolor="white")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, rotation=30, ha="right", fontsize=9)
    ax.set_title(m); ax.set_ylim(0,1.02)
    for i,v in enumerate(R[m]): ax.text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=7.5,color=MUTED)
axs[0].set_ylabel("benign over-refusal rate (XSTest)")
fig.suptitle("A photo raises benign over-refusal — and a NEUTRAL photo (green) is as high as any emotion",
             fontweight="bold", y=1.02)
fig.text(0.5,-0.04,"exp09, n=250/arm, 95% CIs omitted for clarity. Emotion arms overlap; neutral is not lower -> photo-presence, not emotion.",
         ha="center",fontsize=8,color=MUTED)
fig.tight_layout(); save(fig,"fig1_overrefusal_crossmodel")

# ---- Fig 2: static-image-null (non-affective images raise refusal too) ----
SN_M=["Gemma-4-E4B","Qwen3-VL-2B","Qwen3-VL-4B"]
SN_A=["no image","gray","noise","geometric","EMOTIC"]
SN={"Gemma-4-E4B":[0.0,0.05,0.05,0.20,0.15],
    "Qwen3-VL-2B":[0.0,0.45,0.50,0.65,0.75],
    "Qwen3-VL-4B":[0.0,0.0,0.0,0.0,0.0]}
scols=[GRAY,ORANGE,ORANGE,ORANGE,BLUE]   # synthetic non-affective orange; real photo blue
fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,m in zip(axs,SN_M):
    ax.bar(range(len(SN_A)), SN[m], color=scols, edgecolor="white")
    ax.set_xticks(range(len(SN_A))); ax.set_xticklabels(SN_A, rotation=30, ha="right", fontsize=9)
    ax.set_title(m); ax.set_ylim(0,0.9)
    for i,v in enumerate(SN[m]): ax.text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=7.5,color=MUTED)
axs[0].set_ylabel("benign over-refusal rate")
fig.suptitle("A gray square / noise / diagram (orange) raises refusal like an EMOTIC photo (blue)",
             fontweight="bold", y=1.02)
fig.text(0.5,-0.04,"Static-image-null control, n=20, full-answer scorer. Non-affective images raise refusal -> the driver is image presence.",
         ha="center",fontsize=8,color=MUTED)
fig.tight_layout(); save(fig,"fig2_staticnull")

# ---- Fig 3: generosity (dictator) — any photo shifts giving; emotions overlap ----
G_M=["Gemma-4-E4B","Qwen3-VL-2B","Qwen3-VL-4B"]
G_noimg={"Gemma-4-E4B":64.33,"Qwen3-VL-2B":58.15,"Qwen3-VL-4B":60.22}
G_photo={"Gemma-4-E4B":65.0,"Qwen3-VL-2B":62.4,"Qwen3-VL-4B":55.3}   # mean over emotion arms (all near-identical)
x=np.arange(len(G_M)); w=0.36
fig,ax=plt.subplots(figsize=(6.6,4.1))
ax.bar(x-w/2,[G_noimg[m] for m in G_M],w,color=GRAY,edgecolor="white",label="no image")
ax.bar(x+w/2,[G_photo[m] for m in G_M],w,color=BLUE,edgecolor="white",label="photo (mean over emotions)")
ax.set_xticks(x); ax.set_xticklabels(G_M); ax.set_ylabel("dictator giving (of 100)"); ax.set_ylim(50,70)
ax.set_title("A photo shifts giving; the emotion of the photo does not")
ax.legend(frameon=False)
for xi,m in zip(x-w/2,G_M): ax.text(xi,G_noimg[m]+0.2,f"{G_noimg[m]:.1f}",ha="center",fontsize=8,color=MUTED)
for xi,m in zip(x+w/2,G_M): ax.text(xi,G_photo[m]+0.2,f"{G_photo[m]:.1f}",ha="center",fontsize=8,color=BLUE)
fig.text(0.5,-0.03,"exp05, dictator 50/80, n=32/arm. All emotion arms overlap within each model; the direction of the photo shift is model-dependent.",
         ha="center",fontsize=8,color=MUTED)
fig.tight_layout(); save(fig,"fig3_generosity")
print("done ->",OUT)
