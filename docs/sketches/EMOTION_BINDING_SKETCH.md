# Multi-Character Emotion Binding in VLMs — Experiment Sketch

*Novelty verified by targeted prior-art search: **OPEN**. The building blocks are mature but disjoint —
binding-ID mechanistics exist only for **synthetic objects**, VLM emotion interpretability treats emotion as a
**global** property, and multi-character emotion/ToM is **output-level benchmarking**. Nobody has done
character-indexed emotion probing + attribution-swap steering. Main risk is incremental positioning, not being
scooped. Closest three papers in §7.*

---

## 0. The one-sentence claim

> A vision-language model binds **emotion to specific people**: in a multi-person scene it carries a
> **character-indexed** emotion representation. We test whether the **right emotion is bound to the right
> character**, and causally **swap the attribution** (make the model attribute A's emotion to B).

**Paradigm inherited:** the **binding-ID** framework (Feng & Steinhardt 2310.17191, text; Saravanan et al.
2505.22200, VLM on synthetic shapes) — with **characters = entities** and **emotion = the bound attribute**,
in **real multi-person scenes**.

---

## 1. Why this is the most *feasible* of the novel ideas

**EMOTIC already labels emotion per bounding-boxed person.** A multi-person EMOTIC image gives ground-truth
*per-character* emotion (26 categories + VAD) with the person's bbox — exactly the clean, per-entity
supervision the binding test needs, and **you already have it loaded in the pipeline.** No new dataset, no
coarse-label problem. MoMentS (multi-character video, character-indexed questions) becomes the naturalistic
generalization testbed on top.

---

## 2. The probe target & the binding tests

Take a 2-person image where **A is happy, B is angry** (EMOTIC gives this directly).

1. **Character-indexed decodability.** Can you decode A's emotion from *A's* representation and B's from *B's*?
   Per-character emotion probe + shuffled-label selectivity.
2. **Binding specificity (the core test).** Build the **cross-character confusion matrix**: A's emotion should
   be decodable from A's representation but **not** from B's. Off-diagonal leakage = mis-binding. This is what
   separates "an emotion is present" from "*whose* emotion is bound where."
3. **Factorizability / causal swap** (à la Saravanan 2505.22200). Mean-intervention: swap the activations tied
   to A and B — does the *attributed* emotion swap with them? Clean causal evidence of a binding structure.
4. **Attribution-swap steering.** Intervene to make the model *report* A's emotion for B. Behavioral flip =
   the binding is causally editable.
5. **Illusory emotional conjunctions** (mechanistic version of Campbell et al. 2411.00238). Does mis-binding
   rise when characters are spatially close / visually similar / emotionally opposite? Turns a known behavioral
   binding failure into a representational account.

---

## 3. Character-indexed readout (the one new harness piece)

You must associate an internal representation with a *specific* person. Two options, use both:
- **Prompt-indexed (primary, reuses your task readout):** ask per character — "What is the person **on the
  left / in the red shirt / at [bbox]** feeling?" — and read the answer-prep (last) token. Matched-readout
  discipline from the emotion work carries over directly (same question form per character).
- **Token-indexed (secondary, mechanistic):** pool the **image patch tokens overlapping each person's bbox**
  (EMOTIC gives boxes) and probe those. Lets you test whether binding lives in the visual tokens vs the
  answer position.

Both sites (`resid_post` + `attn_out`) as before; binding-ID work localizes to specific attention heads, so
`attn_out` is especially relevant here.

---

## 4. Analyses (reuse the pipeline; new bits are per-character indexing + swap intervention)

1. **Behavioral gate.** Can the model report *per-person* emotion in multi-person EMOTIC scenes (vs a
   single-person control)? If it can't, binding is uninterpretable — gate first.
2. **Character-indexed decodability** + selectivity (§2.1).
3. **Binding-specificity confusion matrix** (§2.2) — the headline figure.
4. **Causal swap via mean-intervention** (§2.3) + **attribution-swap steering** (§2.4), with bootstrap CIs.
5. **Illusory-conjunction analysis** (§2.5): mis-binding vs proximity / similarity / emotional contrast.
6. **Controls:** single-person images (binding trivially correct — positive control); a non-emotion attribute
   (e.g., clothing color) as a "binding works for objects here too" cross-check against Saravanan.

---

## 5. Contributions

1. **First character-indexed emotion probe** — emotion as a *bound, per-person* attribute, vs the global
   whole-image emotion of Zhang et al. 2605.21980.
2. **Extends the binding-ID mechanism from synthetic shapes to emotion + real multi-person scenes** (Saravanan
   2505.22200 → naturalistic affect).
3. **Causal attribution-swap steering** — reassign *whose* emotion; not done for emotion in any VLM.
4. **Mechanistic account of emotional mis-binding** (illusory emotional conjunctions).
5. **Safety framing:** mis-binding emotion to the wrong person is a concrete failure mode for deployed
   affective/assistive systems (blaming the wrong person's anger, etc.).

---

## 6. Models & timeline

- **Models:** the existing sweep (bridge-supported Qwen3.5 / LLaVA-OneVision / InternVL / Gemma), starting
  with whichever passes the per-person behavioral gate.
- **4–6 weeks:**
  - Wk 0–1: multi-person EMOTIC subset + per-character readout; behavioral gate.
  - Wk 1–2: character-indexed probes + binding-specificity confusion matrix.
  - Wk 2–4: causal swap (mean-intervention) + attribution-swap steering, bootstrap CIs.
  - Wk 4–6: illusory-conjunction analysis; MoMentS generalization; write-up.

---

## 7. Related work & the exact wedge (verified search)

1. **Saravanan, Tapaswi, Gandhi — "Investigating Mechanisms for In-Context Vision Language Binding" (arXiv
   2505.22200, CVPR-W 2025)** ✓ — *your method template.* Binding-ID in VLMs via factorizability + mean
   interventions — but **only synthetic shapes with color/item attributes.** Wedge: emotion as the attribute,
   real people as entities.
2. **Zhang et al. — "Interpreting and Enhancing Emotional Circuits in Large VLMs" (arXiv 2605.21980, ICML
   2026)** ✓ — *closest on emotion+VLM+steering.* Treats emotion as a **global, whole-image** property; does
   **not** attribute emotion to specific individuals. Wedge: per-character binding.
3. **Feng & Steinhardt — "How do Language Models Bind Entities in Context?" (arXiv 2310.17191, 2024)** — the
   foundational binding-ID framing we inherit (text, abstract entities). Wedge: multimodal + emotion + people.

Secondary: **Campbell et al. — "Understanding the Limits of VLMs Through the Lens of the Binding Problem"
(arXiv 2411.00238, NeurIPS 2024)** — VLM binding failures, but **behavioral only** (motivates the illusory-
conjunction analysis); **MoMentS (2507.04415)** and **EMOTIC** — multi-person emotion at the **output** level
only.

**Positioning line:** *"Binding-ID showed VLMs tag objects with attributes; emotional-circuit work showed VLMs
represent emotion globally. We show VLMs tag **people** with **emotions** — and that this binding can be
probed, and causally swapped."*

---

## 8. Kill-criteria (be honest early)

- Model fails the **per-person** behavioral gate (can't separate two people's emotions) → binding
  uninterpretable; drop to controlled 2-person compositions or switch model.
- **No binding specificity** (A's emotion equally decodable from B) → either the readout doesn't isolate
  characters (fix the indexing) or the model genuinely doesn't bind emotion per-person — the latter is itself
  a publishable negative result (VLMs represent scene-level affect, not per-person).
- **Swap intervention has no behavioral effect** → the probed direction isn't the causal binding variable;
  report as correlational and investigate the attention-head route (attn_out).
