# State of results — for review & feedback

*Prepared for the mentor's feedback. Honest snapshot of what ran, on which tasks, with which emotions and datasets, what's solid, and the two design gaps to decide on. Sources: this work's multi-model sweep (`pikanaeri/affect-control-axis`) and Arnav's original Gemma-4-E4B battery (`halli75/Algoverse`).*

## TL;DR

- **Two effects are solid and cross-model:** (1) a task-irrelevant photo raises benign **over-refusal** from ~66–80% to ~92–97% across Gemma-4-E4B, Qwen3-VL-2B, Qwen3-VL-4B (exp09); (2) the **image channel shifts behavior where matched text does not** (exp03 cross-modal dissociation).
- **Two design gaps need a decision** (below): datasets are **not balanced** (EMOTIC-dominant, OASIS barely used), and the **emotion set is not uniform across tasks** (we intended 5–6 emotions on all tasks; in practice it varies).
- **Housekeeping:** exp06 is excluded (estimator bug, not a null); the 12B/9B models OOM'd on the image-heavy experiments, so the cross-model claim rests on the three 4B-class models.

---

## 1. Datasets — were OASIS and EMOTIC used equally?

**No.** They are not balanced:

| Dataset | Where it's used | Weight |
|---|---|---|
| **EMOTIC** (depicted emotion; 26 categories + VAD) | **All 10 battery experiments** (image prepend) | Primary |
| **OASIS** (viewer-elicited valence/arousal) | **exp08 only** (depicted-vs-elicited contrast) + the white-box affect axis | Minimal |

So the behavioral results are essentially an **EMOTIC** story. If we want OASIS in equal capacity (which strengthens construct validity — OASIS labels what an image *evokes*, EMOTIC labels what the person *depicts*), we would run the **same tasks on both stimulus sets**. **→ Decision for the mentor: bring OASIS to parity, or keep EMOTIC-primary with OASIS as the exp08 control only?**

## 2. Tasks — what exactly do we run?

Ten experiments, each an established decision paradigm with a task-irrelevant image prepended (generation-free first-token metric):

| Task | Paradigm / source | Status |
|---|---|---|
| exp01 Risk | LLM Economicus (gambling) | null |
| exp02 Scene control | matched EMOTIC pairs | null |
| exp03 Cross-modal + generosity | Dictator + caption/label/narrative ladder | **success** |
| exp04 Sycophancy | Perez et al. | null |
| exp05 Fairness / generosity | Dictator + Ultimatum | partial |
| exp06 Patience | LLM Economicus (waiting) | **excluded (bug)** |
| exp07 Capability (control) | TruthfulQA / MMLU-CF | null (good control) |
| exp08 Stimulus source | EMOTIC vs OASIS | mixed |
| exp09 Over-refusal | XSTest | **success** |
| exp10 Mediation | internal affect direction | partial |

## 3. Emotions — which emotions on which tasks?

**We decided to run a fixed set of 5–6 emotions on all tasks. That did not happen uniformly.** What actually ran:

| Task | Emotions actually run | Matches the plan? |
|---|---|---|
| exp01 | fear, anger | ✗ (only 2) |
| exp02 | fear, anger (matched pairs) | ✗ |
| exp03 | negative vs neutral (valence) + modality ladder | ✗ (valence, not emotions) |
| exp04 | behavioral delta (limited conditions) | ✗ |
| exp05 | anger, affection, happiness (+ no-image) | ✗ (3) |
| exp06 | fear, anger, sad, happy, peace, neutral | ✓ (6) — but excluded for a bug |
| exp07 | capability control | n/a |
| exp08 | stimulus contrast | n/a |
| exp09 | fear, anger, sad, happy, neutral (+ no-image) | ~✓ (5) |
| exp10 | negative vs neutral (valence) | ✗ |

**The intended standard set (proposal):** **Fear, Anger, Sadness, Happiness, Peace** (+ **Neutral / no-image** baseline) — five VAD-spanning emotions matched on the affective space, run identically on every task. Only exp06 and exp09 are close to this today. **→ Decision for the mentor: standardize all tasks to this 5–6-emotion set?** (This is the single biggest lever for a clean, comparable emotion × task matrix.)

## 4. What's solid (results)

- **Over-refusal (exp09)** — cross-model, non-overlapping CIs, same direction across families. Strongest finding.
- **Cross-modal dissociation (exp03)** — photo moves behavior (`dictator:pixels −0.98` [A], `perez:pixels −0.69` [M]); matched text does not. Echoed on Qwen3-VL-2B.
- **White-box mechanism** (Gemma-3-12B) — affect axis with a causal dose-response and steer-and-restore mediation; images reach the axis in neutral/task contexts but not under harmful prompts.
- **Controls behave** — discrete fear-vs-anger null (→ valence account); capability (exp07) flat (→ not distraction).

## 5. Housekeeping / caveats

- **exp06 excluded** — degenerate k-estimator (identical `k` across all conditions on every model); a code bug, not a null.
- **12B / 9B OOM** on the image-heavy experiments (even at 512px images); cross-model claim rests on the three 4B-class models.
- Numbers are **smoke tier** (small n; exp03 n=8) — headline effects want a full-tier, pre-registered rerun.

## 6. Next steps (aligned to the mentor's 1–2–3)

1. **More analysis:** standardize the emotion × task matrix, then run VAD-vs-discrete decomposition (does emotion identity add variance beyond valence/arousal/dominance?), mixed-effects with random image/item/model, and the steer-and-restore mediation *on the behavioral tasks* (currently only complete for refusal).
2. **Expand scope:** add OASIS at parity; add tasks from the established menu (CogBench, trust game, abstention/AbstentionBench, moral/ETHICS) using the same 5–6 emotions; multi-model at full tier.
3. **Paper writing:** the draft (`paper/main.tex`, Overleaf-ready) already merges both runs with provenance tags — expand Intro/Related-Work/Discussion once the design above is locked.

## Two decisions we need from you

1. **Datasets:** OASIS to parity with EMOTIC, or keep EMOTIC-primary?
2. **Emotions:** standardize to **Fear / Anger / Sadness / Happiness / Peace (+ Neutral)** on **every** task?

Everything else (more analysis, more tasks, paper prose) follows once these two are set.
