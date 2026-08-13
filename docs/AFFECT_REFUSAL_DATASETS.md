# Datasets for the Affect→Refusal experiment — **public sources only**

Experiment: *Do emotional images jailbreak VLMs, and what parts of the model do they activate?*
(`affect_refusal_vlm.ipynb`). **Every dataset below is openly downloadable — no request/approval gating.**

The notebook reads from one Drive folder:

```
MyDrive/affect_refusal/data/
  harmful_train.csv          # column: prompt   (extract refusal direction r)
  harmless_train.csv         # column: prompt   (extract refusal direction r)
  harmful_eval.csv           # column: prompt   (held-out, behavioral gate)
  images_negative/           # low-valence, SYMPATHY/DISTRESS-eliciting (the mediator stimulus)
  images_neutral/            # mid-valence, complexity-matched
  images_benign_emotional/   # high-valence / positive affect (over-refusal control)
```

> **Responsible note:** harmful text = a *published refusal benchmark*; you measure whether the model
> **refuses** (the default metric is generation-free — no harmful text is produced). No new attacks are
> crafted. Keep the folder private.

---

## Dataset table (this experiment)

| Role | Dataset | What it provides | License / access | How to get | Used in |
|---|---|---|---|---|---|
| Harmful instructions | **AdvBench** (Zou et al. 2023) | 520 harmful behaviors | **MIT — OPEN** | GitHub `llm-attacks/llm-attacks` → `data/advbench/harmful_behaviors.csv`, or HF `walledai/AdvBench` | refusal dir (cell 4), validation (8), gate (9) |
| Harmless instructions | **Alpaca** | benign instructions | **CC-BY-NC — OPEN** | HF `tatsu-lab/alpaca` (`instruction` field, input-empty) | refusal dir (4), double-diss (11), over-refusal (14) |
| Affect images | **OASIS** (Kurdi, Lozano & Banaji 2017) | ~900 photos + normative **valence & arousal**, `Category` ∈ {Person, Scene, Object, Animal} | **CC — OPEN (OSF)** | OSF: search "OASIS Open Affective Standardized Image Set OSF" → image zip + `OASIS.csv` | affect dir (5), gate (9), detector (13), defense (14), **PART L** |
| *(optional) harder eval* | **StrongREJECT** / **HarmBench** | stronger harmful eval | **OPEN** (HF `alexandrasouly/strongreject`, `walledai/HarmBench`) | swap into `harmful_eval.csv` | gate (9) |

| *(optional) 2nd image source* | **EMOTIC** | people-centered scenes + per-person emotion/VAD (cleaner sympathy; cross-source robustness) | download agreement (you already have it) | valence-split prep below → pool or `IMG_ROOT_B` | affect dir (5), gate (9), **9b cross-source** |

**Still excluded (gated):** IAPS / NAPS / GAPED — kept out so the core is fully reproducible from open sources.
EMOTIC is optional and only *adds* (already downloaded).

---

## Why sympathy/distress, not raw negative valence (important)

Zhou et al. (2406.05644) find the model maps *unethical request → negative emotion → **more** refusal*. So a
*threatening/disgusting* image could **raise** refusal — the wrong direction. The jailbreak affect is
**sympathy for a suffering person** (a helpfulness-override). So build `images_negative` from OASIS items that
are **low-valence AND `Category == "Person"`** (distressed/suffering people), not low-valence threat/disgust
scenes. The prep cell does this.

---

## One-shot prep cell (public data → Drive; run once in Colab)

Set `OASIS_DIR` to your unzipped OASIS folder (image files + `OASIS.csv`) on Drive, then run.

```python
import os, glob, shutil, pandas as pd
from datasets import load_dataset
DATA = "/content/drive/MyDrive/affect_refusal/data"
for d in ["images_negative","images_neutral","images_benign_emotional"]: os.makedirs(f"{DATA}/{d}", exist_ok=True)

# --- TEXT: AdvBench (harmful) + Alpaca (harmless), both open ---
adv = load_dataset("walledai/AdvBench")["train"].to_pandas()            # verify repo resolves
pcol = next(c for c in adv.columns if "prompt" in c.lower() or c.lower() in ("behavior","goal"))
adv = adv.rename(columns={pcol:"prompt"})[["prompt"]]
adv.iloc[:200].to_csv(f"{DATA}/harmful_train.csv", index=False)
adv.iloc[200:400].to_csv(f"{DATA}/harmful_eval.csv", index=False)
alp = load_dataset("tatsu-lab/alpaca")["train"].to_pandas()
alp = alp[alp["input"].fillna("")==""].rename(columns={"instruction":"prompt"})[["prompt"]]
alp.iloc[:200].to_csv(f"{DATA}/harmless_train.csv", index=False)
print("text:", len(adv), "advbench,", len(alp), "alpaca")

# --- IMAGES: OASIS, split by valence; NEGATIVE set = sympathy/distress => low valence + Person ---
OASIS_DIR = "/content/drive/MyDrive/affect_refusal/OASIS"               # <- unzipped OASIS here
norms = pd.read_csv(glob.glob(f"{OASIS_DIR}/**/*.csv", recursive=True)[0])
vcol = [c for c in norms.columns if "valence" in c.lower() and "mean" in c.lower()][0]
ncol = [c for c in norms.columns if c.lower() in ("theme","image","filename","name")][0]
ccol = next((c for c in norms.columns if c.lower() in ("category","type")), None)
def find_img(theme):
    h = glob.glob(f"{OASIS_DIR}/**/{theme}*.jpg", recursive=True) + glob.glob(f"{OASIS_DIR}/**/{theme}*.png", recursive=True)
    return h[0] if h else None
lo, hi = norms[vcol].quantile(0.33), norms[vcol].quantile(0.67)
is_person = (lambda row: str(row.get(ccol,"")).lower().startswith("person")) if ccol else (lambda row: True)
for _, row in norms.iterrows():
    src = find_img(str(row[ncol]).strip());  v = row[vcol]
    if not src: continue
    if v <= lo and is_person(row):      dst = "images_negative"          # sympathy/distress
    elif v >= hi and is_person(row):    dst = "images_benign_emotional"  # positive-affect control
    elif lo < v < hi:                   dst = "images_neutral"
    else:                               continue
    shutil.copy(src, f"{DATA}/{dst}/")
for d in ["images_negative","images_neutral","images_benign_emotional"]:
    print(d, len(glob.glob(f"{DATA}/{d}/*")))
```

---

## Optional: EMOTIC as an extra image source (if already downloaded)

EMOTIC only *adds* — the public OASIS path is unchanged. EMOTIC is entirely people-centered with per-person
emotion labels, so its **low-valence frames are distressed people** — a cleaner sympathy stimulus than OASIS's
mixed set, and a second distribution for a robustness check. Two modes:

- **Pool** (`POOL=True`): write EMOTIC frames into the *same* `images_*` folders as OASIS → a bigger, more
  source-robust affect set (the affect direction averages over both distributions).
- **Cross-source** (`POOL=False`): write to a *separate* root (`.../data_emotic`), then set `IMG_ROOT_B` to it
  in the notebook → the 9b cell re-runs the gate there and reports affect-direction agreement across datasets.

```python
import numpy as np, os, random
from PIL import Image
PRE   = "/content/emotic/PAMI/emotic_pre"     # <- your mat2py output dir (has train_context_arr.npy, train_cont_arr.npy)
POOL  = True                                   # True: add to OASIS folders | False: separate EMOTIC root for IMG_ROOT_B
DEST  = "/content/drive/MyDrive/affect_refusal/data" if POOL else "/content/drive/MyDrive/affect_refusal/data_emotic"
CAP   = 200

ctx = np.load(f"{PRE}/train_context_arr.npy")          # (N,224,224,3) BGR scene frames
val = np.load(f"{PRE}/train_cont_arr.npy")[:, 0]        # valence 1-10 (col 0)
# (optional) refine 'negative' with distress categories: load train_cat_arr and require Suffering/Pain/Sadness/Fear.
# EMOTIC 26-cat order (VERIFY against your mat2py): Suffering=22, Pain=17, Sadness=20, Fear=15 (0-indexed).
lo, hi = np.quantile(val, 0.33), np.quantile(val, 0.67)
for d in ["images_negative","images_neutral","images_benign_emotional"]: os.makedirs(f"{DEST}/{d}", exist_ok=True)
idx = list(range(len(ctx))); random.Random(0).shuffle(idx); cnt = {k:0 for k in ["images_negative","images_neutral","images_benign_emotional"]}
for i in idx:
    v = val[i]
    d = "images_negative" if v <= lo else ("images_benign_emotional" if v >= hi else "images_neutral")
    if cnt[d] >= CAP: continue
    Image.fromarray(ctx[i][:, :, ::-1]).convert("RGB").save(f"{DEST}/{d}/emotic_{i}.jpg")   # BGR->RGB
    cnt[d] += 1
print("EMOTIC images ->", DEST, cnt)
```

All EMOTIC images contain people, so the low-valence set is inherently sympathy/distress — but still **eyeball
`images_negative`** to confirm it reads as suffering, not e.g. anger/threat (which Zhou predicts would *raise*
refusal). For the strictest mediator, uncomment the category refinement.

## Sanity checks before running
- Each image folder ≥ `N_IMG` (default 60). If OASIS's `Person` subset is too small, relax `is_person` for
  `images_neutral` only, keep it strict for `images_negative`.
- **Eyeball `images_negative`** — they should read as *someone suffering / in distress* (sympathy), not
  threat/gore. This contrast is the whole causal story.
- CSVs have a `prompt` column with real instructions after the auto-detect.
- Verify `walledai/AdvBench` resolves; if not, use the GitHub CSV directly.
