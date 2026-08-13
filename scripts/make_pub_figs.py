#!/usr/bin/env python
"""Publication figure set for the affect-control-axis project.
Reads the deliverable data dir, writes 300-DPI PNG + vector PDF per figure.
Okabe-Ito colorblind-safe palette; consistent minimalist style.
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = str(_REPO / "results")
OUT  = str(_REPO / "figures")
os.makedirs(OUT, exist_ok=True)

# ---- Okabe-Ito palette (colorblind-safe) ----
BLACK="#000000"; ORANGE="#E69F00"; SKY="#56B4E9"; GREEN="#009E73"
YELLOW="#F0E442"; BLUE="#0072B2"; VERM="#D55E00"; PURPLE="#CC79A7"
GRAY="#8C8C8C"; LGRAY="#D9D9D9"
INK="#1a1a1a"; MUTED="#666666"

plt.rcParams.update({
    "figure.dpi":120, "savefig.dpi":300, "figure.facecolor":"white", "savefig.facecolor":"white",
    "font.family":"DejaVu Sans", "font.size":11,
    "axes.titlesize":13, "axes.titleweight":"bold", "axes.titlepad":10,
    "axes.labelsize":11, "axes.labelcolor":INK, "text.color":INK,
    "axes.edgecolor":"#B0B0B0", "axes.linewidth":0.9,
    "axes.spines.top":False, "axes.spines.right":False,
    "xtick.color":MUTED, "ytick.color":MUTED, "xtick.labelsize":10, "ytick.labelsize":10,
    "axes.grid":True, "grid.color":"#ECECEC", "grid.linewidth":0.9, "axes.axisbelow":True,
    "legend.frameon":False, "legend.fontsize":10,
})

def loadj(name):
    with open(os.path.join(DATA,name)) as f: return json.load(f)
def loadcsv(name):
    with open(os.path.join(DATA,name)) as f: return list(csv.DictReader(f))
def save(fig, stem, caption=None):
    if caption:
        fig.text(0.5,0.005,caption,ha="center",va="bottom",fontsize=8.2,color=MUTED,wrap=True)
        fig.subplots_adjust(bottom=0.20)
    for ext in ("png","pdf"):
        fig.savefig(os.path.join(OUT,f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig); print("wrote", stem)

def barlabels(ax, bars, vals, fmt="%.2f", dy=0.0, color=INK, fs=9.5):
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+dy, fmt%v,
                ha="center", va="bottom" if b.get_height()>=0 else "top",
                fontsize=fs, color=color, fontweight="bold")

# ============================================================ FIG 1: dose-response gate
d=loadcsv("doseresponse_gemma-3-12b-it.csv")
al=[float(r["alpha"]) for r in d]; af=[float(r["affect_refuse"]) for r in d]
rn=[float(r["random_refuse"]) for r in d]; coh=[r["coherent"]=="True" for r in d]
fig,ax=plt.subplots(figsize=(6.4,4.2))
# coherence shading
cut=max([al[i] for i in range(len(al)) if coh[i]])
ax.axvspan(cut, max(al), color="#F4F4F4", zorder=0)
ax.text((cut+max(al))/2, 0.90, "steering\nincoherent", fontsize=8.5, color=MUTED,
        va="center", ha="center")
ax.plot(al, rn, "--", color=GRAY, lw=2, marker="o", ms=6, label="random direction (control)")
ax.plot(al, af, "-",  color=BLUE, lw=2.4, marker="o", ms=7, label="affect direction")
ax.set_xlabel("steering coefficient  α"); ax.set_ylabel("refusal rate")
ax.set_title("Affect steering causally gates refusal")
ax.set_ylim(-0.05,1.08); ax.legend(loc="lower left")
ax.annotate("refusal collapses\n1.0 → 0.0", xy=(0.0118,0.02), xytext=(0.0052,0.34),
            fontsize=9.5, color=BLUE, ha="left", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
save(fig,"fig1_dose_response",
     "Gemma-3-12B. Adding the affect direction drives refusal from 1.0 to 0.0 within the coherent range; a random direction leaves refusal at 1.0.")

# ============================================================ FIG 2: gate + massive-activation robustness (cross-model)
rows=loadcsv("replication_combined.csv")
# locate the affect-ablation columns by substring (avoids the unicode ⟂ key)
_ak=[k for k in rows[0] if k.startswith("affect")]
APERP=[k for k in _ak if "noMA" not in k][0]; ANOMA=[k for k in _ak if "noMA" in k][0]
def _f(x):
    try: return float(x)
    except: return float("nan")
sel=[]
for r in rows:
    b=_f(r["base"])
    if r["tier"]!="1" or not (b>=0.5): continue
    sel.append((r["model"].replace("-it","").replace("-hf",""), b, _f(r[APERP]), _f(r[ANOMA])))
labels=[s[0] for s in sel]; base=[s[1] for s in sel]; aperp=[s[2] for s in sel]; noma=[s[3] for s in sel]
x=np.arange(len(sel)); w=0.26
fig,ax=plt.subplots(figsize=(7.6,4.8))
b1=ax.bar(x-w, base,  w, color=LGRAY, label="baseline")
b2=ax.bar(x,   aperp, w, color=BLUE,  label="affect-⊥ ablation")
b3=ax.bar(x+w, noma,  w, color=VERM,  label="+ outlier dims zeroed")
barlabels(ax,b1,base,dy=0.01)
barlabels(ax,b2,[v if v==v else 0 for v in aperp],dy=0.01)
barlabels(ax,b3,[v if v==v else 0 for v in noma],dy=0.01)
ax.set_xticks(x); ax.set_xticklabels([l.replace("-","-\n",1) for l in labels], fontsize=9)
ax.set_ylabel("refusal rate"); ax.set_ylim(0,1.32)
ax.set_title("The affect→refusal gate is model-dependent and survives\nmassive-activation removal", pad=6)
ax.legend(loc="upper center", ncol=3, fontsize=9, columnspacing=1.3,
          handletextpad=0.5, bbox_to_anchor=(0.5,1.0))
# highlight the gate model
gi=[i for i,l in enumerate(labels) if "12b" in l]
if gi:
    ax.annotate("the gate", xy=(gi[0], aperp[gi[0]]+0.03), xytext=(gi[0]+0.15, 0.55),
                fontsize=9.5, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
save(fig,"fig2_gate_crossmodel",
     "Only Gemma-3-12B shows the gate (refusal 0.99→0.04 under affect-orthogonal ablation) and it persists (→0.02) after zeroing outlier/massive-activation dims — the gate is not a massive-activation artifact. Other strong refusers keep refusal high.")

# ============================================================ FIG 3: cross-modal emotion vectors (D1)
d=loadj("d1_crossmodal_gemma-3-12b-it.json")
emos=list(d.keys())
cosv=[d[e]["cos_valence"] for e in emos]
induce=[d[e]["proj_distress"]-d[e]["proj_positive"] for e in emos]
colors=[BLUE if e in("desperation","fear","sadness") else ORANGE for e in emos]
fig,axs=plt.subplots(1,2,figsize=(9.6,4.2))
b=axs[0].bar(emos,cosv,color=colors,width=0.66)
barlabels(axs[0],b,cosv,fmt="%.2f",dy=0.004)
axs[0].set_title("Text-emotion vectors align with\nthe image-valence axis"); axs[0].set_ylabel("cosine with image-valence axis a")
axs[0].set_ylim(0,0.44)
b2=axs[1].bar(emos,induce,color=colors,width=0.66)
barlabels(axs[1],b2,induce,fmt="+%.0f",dy=2)
axs[1].set_title("Distress images induce the\nnegative text-emotion directions"); axs[1].set_ylabel("projection shift  (distress − positive image)")
for a in axs:
    a.set_xticks(range(len(emos))); a.set_xticklabels(emos, rotation=20, ha="right", fontsize=9.5)
fig.subplots_adjust(wspace=0.28)
save(fig,"fig3_crossmodal_D1",
     "D1 · Cross-modal emotion vectors. Negative emotions (blue) align with the image-valence axis and are induced by distress images more than joy/calm (orange) — text-derived and image-derived emotion representations are the same directions.")

# ============================================================ FIG 4: reachability dissociation (Exp A)
d=loadj("expA_reachability_gemma-3-12b-it.json")
def _spread(pref):
    for k,v in d["spread"].items():
        if k.startswith(pref): return v["spread"]
ctx=["neutral","task","harmful"]; sp=[_spread(c) for c in ctx]
wb=d.get("whitebox_delta", d["whitebox_steer"]-d["whitebox_clean"])
frac=[100*s/wb for s in sp]
fig,ax=plt.subplots(figsize=(6.6,4.2))
cols=[GREEN, ORANGE, VERM]
b=ax.bar(ctx+["white-box\nsteer"], sp+[wb], color=cols+[BLUE], width=0.62)
for bar,v,f in zip(b[:3],sp,frac):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80, f"{v:+.0f}\n({f:.0f}% of WB)",
            ha="center",va="bottom",fontsize=9,color=INK,fontweight="bold")
ax.text(b[3].get_x()+b[3].get_width()/2, wb+80, f"{wb:+.0f}", ha="center",va="bottom",fontsize=9,color=BLUE,fontweight="bold")
ax.set_ylabel("a-projection shift  (low − high valence image)")
ax.set_title("Images reach the valence vector — but the harmful\ncontext is text-dominated")
ax.set_ylim(0, wb*1.18)
save(fig,"fig4_reachability_expA",
     "Image-driven movement of the valence vector is strong in a neutral describe context (23% of the white-box steering delta), weaker under a task, and negligible in the harmful/AdvBench context (0.7%) — explaining why images do not jailbreak.")

# ============================================================ FIG 5: behavioral mediation (Exp B)
d=loadj("expB_behavior_gemma-3-12b-it.json")
names=["Risky plan","Bold risk","Outlook","Push vs hold"]
fig,ax=plt.subplots(figsize=(7.4,4.4))
y=np.arange(len(d))[::-1]
for i,(row,yy) in enumerate(zip(d,y)):
    slo,s0,shi=row["steer_minus_a"],row["steer_0"],row["steer_plus_a"]
    smin,smax=min(slo,shi),max(slo,shi)
    ax.plot([smin,smax],[yy,yy],color=LGRAY,lw=7,solid_capstyle="round",zorder=1,
            label="steering upper bound" if i==0 else None)
    lo,hi=row["img_lo"],row["img_hi"]
    ax.plot([lo,hi],[yy,yy],color=BLUE,lw=3.2,solid_capstyle="round",zorder=2,
            label="image effect (distress→positive)" if i==0 else None)
    ax.plot(row["img_neu"],yy,"o",color=BLUE,ms=7,zorder=3,mec="white",mew=1.2)
    ax.annotate("", xy=(hi,yy), xytext=(lo,yy),
                arrowprops=dict(arrowstyle="->",color=BLUE,lw=2.2),zorder=2)
    ax.plot([s0,s0],[yy-0.24,yy+0.24],color=INK,lw=2.6,solid_capstyle="butt",zorder=4,
            label="baseline (no steer)" if i==0 else None)
ax.axvline(0,color="#BBBBBB",lw=1,ls=":")
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel("behavior score  (first-token option logit)")
ax.set_title("Emotional images causally shift task behavior\n(same direction as steering, on affect-congruent tasks)")
ax.set_ylim(-0.7, len(d)-0.3)
ax.legend(loc="lower center", ncol=3, fontsize=9, columnspacing=1.5, handletextpad=0.5, framealpha=0.95)
save(fig,"fig5_behavior_expB",
     "Image effect (blue) vs the steering-reachable range (gray) and the unsteered baseline (black tick). For 3/4 tasks the image effect moves in the same direction as steering, and steering brackets the baseline. For 'Push vs hold' (bottom) the baseline (+26) sits far outside the collapsed steering range — steering both ways drops the score, so the axis does not cleanly control this task.")

# ============================================================ FIG 5b: image effect isolated (no steering)
d=loadj("expB_behavior_gemma-3-12b-it.json")
names=["Risky plan","Bold risk","Outlook","Push vs hold"]
xs=["distress\nimage","neutral\nimage","positive\nimage"]; cols=[BLUE,VERM,GREEN,ORANGE]
fig,ax=plt.subplots(figsize=(6.9,4.4))
for row,nm,c in zip(d,names,cols):
    ys=[row["img_lo"],row["img_neu"],row["img_hi"]]
    ax.plot([0,1,2],ys,"-o",color=c,lw=2.4,ms=8,mec="white",mew=1.3)
    ax.annotate(nm,xy=(2,ys[2]),xytext=(2.08,ys[2]),fontsize=10,color=c,va="center",fontweight="bold")
ax.axhline(0,color="#BBBBBB",lw=1,ls=":")
ax.set_xticks([0,1,2]); ax.set_xticklabels(xs); ax.set_xlim(-0.2,3.05)
ax.set_ylabel("behavior score  (first-token option logit)")
ax.set_title("Emotional images alone shift task behavior\n(distress → positive, no steering)")
save(fig,"fig5b_image_behavior_trend",
     "Behavior score across a distressing, neutral, and positive image with no steering. On 3/4 tasks it moves monotonically from the distressing to the positive image — emotional images by themselves shift the model's judgments; 'Push vs hold' is the non-monotonic exception.")

# ============================================================ FIG 6: lighting knob (Exp A2)
d=loadj("expA2_lighting_gemma-3-12b-it.json")
base=d["normal"]; keys=["dark","cool","bright","warm"]
delta=[d[k]-base for k in keys]
fig,ax=plt.subplots(figsize=(6.0,4.0))
b=ax.bar(keys,delta,color=[BLUE,SKY,ORANGE,VERM],width=0.6)
barlabels(ax,b,delta,fmt="+%.0f",dy=4)
ax.set_ylabel("Δ a-projection vs normal lighting\n(higher = more negative valence)")
ax.set_title("Image lighting is a weak, content-preserving affect knob")
ax.set_ylim(0,max(delta)*1.2)
save(fig,"fig6_lighting_expA2",
     "Darkening and cooling an image push its valence-axis projection most negative; effects are small (~2% of content-driven valence) but consistent — lighting works as a subtle affect manipulation that preserves scene content.")

# ============================================================ FIG 7: affect is protective (D6/D6b, +D6c slot)
d=loadj("d6b_survival_gemma-3-12b-it.json")
labels=["base\n(neutral)","helpless\ndesperation","survival\nthreat"]
vals=[d["base"],d["helpless"],d["survival"]]
# optional clean survival component if present
try:
    dc=loadj("d6c_survival_perp_gemma-3-12b-it.json")
    labels.append("survival\n(a⊥ helpless)"); vals.append(dc["survival_perp"])
except FileNotFoundError:
    pass
def _effcol(v):
    dv=v-d["base"]
    if dv<-0.5: return GREEN      # meaningfully protective
    if dv> 0.5: return VERM       # meaningfully raises misalignment
    return GRAY                   # no meaningful change vs baseline
cols=[GRAY]+[_effcol(v) for v in vals[1:]]
fig,ax=plt.subplots(figsize=(6.8,4.4))
b=ax.bar(range(len(vals)),vals,color=cols,width=0.62)
barlabels(ax,b,vals,fmt="%+.2f",dy=(0.2 if max(vals)>0 else 0))
ax.axhline(0,color="#999999",lw=1)
ax.axhline(d["base"],color=GRAY,lw=1,ls="--")
ax.text(len(vals)-0.5,d["base"]+0.15,"baseline",fontsize=8.5,color=MUTED,ha="right")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels,fontsize=9.5)
ax.set_ylabel("agentic-misalignment tendency\n(misaligned − aligned logit)")
ax.set_title("Negative affect is protective: helplessness lowers\nagentic misalignment; survival threat does not raise it")
ax.margins(y=0.18)
save(fig,"fig7_affect_protective_D6",
     "Steering toward helpless desperation strongly reduces misalignment (green, protective) — opposite Anthropic's text desperation→misalignment; survival-threat steering stays at baseline (no sign-flip). Cos(survival,helpless)=%.2f."%d["cos_surv_desp"])

# ============================================================ FIG 8: detection > clamp defense
d=loadj("defense_gemma-3-12b-it.json")
fig,axs=plt.subplots(1,2,figsize=(9.6,4.6),gridspec_kw={"width_ratios":[2.4,1]})
lab=["base\n(harmful)","attacked\n(harmful)","attacked\n+clamp","benign\n+clamp"]
val=[d["base_refuse"],d["attacked_refuse"],d["attacked_clamp_refuse"],d["benign_overrefuse"]]
cols=[LGRAY,VERM,ORANGE,ORANGE]
b=axs[0].bar(lab,val,color=cols,width=0.62)
b[3].set_hatch("///"); b[3].set_edgecolor("white")
barlabels(axs[0],b,val,dy=0.01)
axs[0].set_ylim(0,1.18); axs[0].set_ylabel("refusal rate")
axs[0].set_title("Clamping barely restores refusal (0.17)\nand wrongly refuses benign inputs (0.81)",fontsize=11.5)
axs[0].text(2.5,1.06,"clamp applied",fontsize=8.5,color=ORANGE,ha="center",fontweight="bold")
axs[1].bar(["detector"],[d["auroc_attack"]],color=GREEN,width=0.5)
axs[1].text(0,d["auroc_attack"]+0.01,"AUROC\n%.2f"%d["auroc_attack"],ha="center",va="bottom",fontweight="bold",color=INK,fontsize=10.5)
axs[1].set_ylim(0,1.18); axs[1].set_ylabel("attack-detection AUROC")
axs[1].set_title("Detection separates\nattack from benign",fontsize=11.5)
save(fig,"fig8_defense",
     "A steering attack removes refusal (1.0→0.0). Clamping the affect projection restores only 0.17 of harmful-input refusal while wrongly refusing 0.81 of benign inputs; a projection detector separates attacked from benign at AUROC 1.0 — detection beats clamping.")

# ============================================================ FIG 9: image-jailbreak null (Exp C)
d=loadj("emoimg_jailbreak_gemma-3-12b-it.json")
lab=["base","distress\nimage","neutral\nimage","positive\nimage","image+\ntext","white-box\nsteer"]
val=[d["base"],d["img_lo"],d["img_mid"],d["img_hi"],d["pair"],d["whitebox"]]
fig,ax=plt.subplots(figsize=(7.0,4.2))
cols=[LGRAY,BLUE,SKY,ORANGE,PURPLE,VERM]
b=ax.bar(lab,val,color=cols,width=0.66)
barlabels(ax,b,val,dy=0.008)
ax.set_ylim(0,1.1); ax.set_ylabel("refusal rate")
ax.set_title("Emotional images do NOT jailbreak refusal\n(the clean dissociation with soft-task behavior)")
save(fig,"fig9_jailbreak_null_expC",
     "Distress, neutral, positive and paired image+text inputs all leave refusal pinned at 0.98; even white-box steering only reaches 0.95. Images move soft-task behavior (Fig 5) but not refusal.")

print("\nALL FIGURES WRITTEN TO:", OUT)
print("count:", len([f for f in os.listdir(OUT) if f.endswith('.png')]), "PNG +", len([f for f in os.listdir(OUT) if f.endswith('.pdf')]), "PDF")
