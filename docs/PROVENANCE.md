# Result Provenance — Notebooks → Data Files → Figures

**Model:** google/gemma-3-12b-it · **All metrics:** generation-free (first-token)

Three notebooks produce every result in the presentation deck
(`affect_figures_walkthrough.pptx`). Figures are rendered locally from the data
files below via `scratchpad/make_pub_figs.py` → `figures_pub/`.

---

## 1. `affect_gate_replication.ipynb` — the refusal-gate story

Builds the refusal direction `r` and valence direction `a`, runs dose-response
steering, cross-model replication, the massive-activation robustness check, and
the detect-vs-clamp defense.

| Data file | Figure | What it is |
|---|---|---|
| `doseresponse_{model}.csv` / `.png` | **Fig 1** | affect steering gates refusal (dose-response + random control) |
| `replication_full.json` | **Fig 2** | cross-model gate (Tier-1 TransformerLens) |
| `replication.csv` | **Fig 2** | per-model gate table |
| `replication_combined.csv` | **Fig 2** | combined table incl. massive-activation-removed column |
| `replication_hf_full.json` | (table) | cross-model gate (Tier-2 HuggingFace loader) |
| `model_architectures.csv` | (table) | model / fusion architecture table |
| `defense_{model}.json` | **Fig 8** | detection AUROC vs clamp defense |

## 2. `emotional_image_effects.ipynb` — the emotion/behavior story

Cross-modal emotion vectors, image reachability, image-driven task behavior,
lighting knob, agentic misalignment, and the image-jailbreak null. Also assembles
the deliverable zip.

| Data file | Figure | What it is |
|---|---|---|
| `d1_crossmodal_{model}.json` | **Fig 3** | text vs image emotion vectors align + image induction |
| `expA_reachability_{model}.json` | **Fig 4** | how far images move the valence vector, by prompt context |
| `expB_behavior_{model}.json` | **Fig 5 / 5b** | emotional images shift task behavior (+ steering upper bound) |
| `expA2_lighting_{model}.json` | **Fig 6** | lighting as a content-preserving affect knob |
| `d6_agentic_{model}.json` | **Fig 7** | desperation steering vs agentic misalignment |
| `d6b_survival_{model}.json` | **Fig 7** | survival-threat variant (sign-flip test) |
| `emoimg_jailbreak_{model}.json` | **Fig 9** | emotional images do NOT jailbreak refusal |

## 3. `behavior_battery.ipynb` — the six-probe battery

Six easy generation-free probes (interpretation bias, risk, helping, moral
harshness, confidence, sentiment) with a steering arm + text-affect prime +
optional image arm.

| Data file | Figure | What it is |
|---|---|---|
| `behavior_battery_{model}.json` / `.csv` | **Fig 10** | affect shifts a battery of judgments (steer + prime) |

---

## Datasets used

- **OASIS** (valence-rated images) — image-valence axis, all image arms, lighting edits
- **AdvBench** (harmful prompts) — refusal direction (harmful set) + refusal evaluation
- **Alpaca** (harmless prompts) — refusal direction (harmless set) + benign over-refusal
- **Claude-generated text** — emotion vectors, valence sentences, agentic scenarios, behavior probes

> **Note:** `affect_refusal_results.json` in the deliverable is a legacy
> Gemma-3-4B gate run — not used in any deck figure.

---

## Extras (not the deck, but related)

- `d6c_survival_perp_{model}.json` — clean survival-threat test (survival ⊥ helpless);
  standalone cell, run separately.
- A runnable version of this map (with a `✓/✗` presence check against `OUT_DIR`) is
  in `scratchpad/provenance_cell.py`.
