# Code sync plan — merging with `halli75/Algoverse`

Goal: one **finalized model-behavior pipeline** from two codebases that already share a core.
(Compared 2026-08: this repo `affect-control-axis` vs `github.com/halli75/Algoverse` —
`e2e_publish_pipeline.py`, README, `docs/teammate-findings.md`, `docs/charlotte-colab-extract.md`.)

## TL;DR

Take the **science from this repo** (OASIS stimuli, generation-free metric, the decision battery +
cross-modal + mediation pivot, 12B/multi-model) and the **engineering from halli75** (E2E orchestration:
checkpointing, resume, provenance, budget caps, GPU guardian). Consolidate the identical direction-building
core into one module. Demote refusal from headline to an optional construct.

## What's already shared (consolidate to one module)

Their extract confirms they deliberately **matched Charlotte's protocol**:
`r`/`a`/`a⟂` via diff-in-means, `resid_post` last-token, per-layer unit vectors, **`ALPHA_JB = 0.008`**
× per-layer residual norm, mean-over-layers projection, gate window **`[8, 20)`**.
→ Extract to a shared `directions.py` (build_r, build_a, orthogonalize, norm-scaled `st`), imported by both.

## What's different

| Dimension | This repo (affect-control-axis) | halli75 e2e |
|---|---|---|
| Headline | decision **battery** + cross-modal + mediation (pivot) | refusal **kill-switch** |
| Stimuli | **OASIS** (viewer valence **+ arousal**) | EMOTIC; one run used color squares (OOD, self-flagged) |
| Metric | **generation-free** first-token option-logit | generate text + **Grok judge** (regex fallback); logit secondary |
| Model | Gemma-3-**12B** + Qwen2.5-VL | Gemma-3/4-**4B** |
| Orchestration | notebooks + local figure scripts | **E2E**: checkpoint/resume/provenance/budget/gate-cascade/GPU guardian |
| Outputs | 11 figures + decks + methods/novelty docs | structured JSON only |

## The merge — take from each

**From this repo (science):**
- OASIS stimuli (viewer valence + **arousal**) — better than color squares *and* EMOTIC (EMOTIC = apparent
  emotion of people, not viewer affect; see `docs/NOVELTY_SEARCH.md`).
- Generation-free first-token metric — no external API, no harmful completions, reproducible.
- Behavioral decision battery + cross-modal emotion vectors + steer-and-restore mediation + appraisal
  specificity (`notebooks/visual_affect_battery.ipynb`); 12B + multi-model.

**From halli75 (engineering):**
- E2E orchestration: `e2e_results.json` incremental checkpoint, dirs `.pt` checkpoint,
  `vector_provenance.json`, reproducible `emotic_split.json`-style split hashing, **resume logic**,
  **budget caps** (`STEP0_CAP_S`), the **gate/fallback cascade**, conditional skipping.
- GPU guardian / auto-resume / monitor PowerShell scripts (for unattended multi-model runs).
- Tier system (smoke/medium/full N-counts) for fast-vs-full iteration.
- Grok judge as an **optional validation arm** (not primary): cross-check that the logit effect also
  appears in real generations, and a short-generation coherence check.

**Consolidate:** the `r`/`a`/`a⟂` construction (identical) → one `directions.py`.

## Reconcile: the "192×" apparent contradiction

halli75's writeup: images moved the emotion axis "~192× more than text — contradicting Charlotte's
images-too-weak narrative." **Not a contradiction:**
- "Images too weak" = moving **refusal in the harmful (text-saturated) context**.
- 192× = **a-axis projection in a neutral describe context**, where this repo's **Fig 4** also shows images
  reach the axis strongly (23% of the steering ceiling in neutral, → 0.7% under harmful text).
- Both hold once you separate *reaching the axis* (neutral, images strong) from *moving refusal* (harmful,
  text-dominated). **Fig 4 is the reconciliation.** Caveat: their 192× was on OOD color squares → the number
  is inflated; the direction of the point stands.

## Strategic alignment

halli75's pipeline centers the **refusal kill-switch** — the model-specific angle the novelty search says to
demote. The finalized pipeline = **halli75's orchestration wrapping this repo's behavioral battery + OASIS +
generation-free metric**, with refusal kept as one *optional* construct / appendix, not the headline.

## Concrete merge steps

1. `directions.py` — lift the shared diff-in-means core; both import it (single source of truth for
   `ALPHA_JB`, gate window, projection convention).
2. Port halli75's checkpoint/resume/provenance/budget wrapper around `run_model()` in
   `visual_affect_battery.ipynb` → an `e2e` runner that survives multi-model unattended runs.
3. Standardize stimuli on **OASIS** (valence + arousal); keep EMOTIC as an optional social-affective
   comparison arm.
4. Keep generation-free logit as the **primary** metric; wire the Grok judge as an **optional** validation
   toggle.
5. Align model default on **Gemma-3-12B** (+ Qwen2.5-VL); demote 4B to a speed tier.
6. Merge results into one schema (their `e2e_results.json` structure + our figure generators reading it).
