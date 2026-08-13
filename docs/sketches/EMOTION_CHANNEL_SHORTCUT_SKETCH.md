# Modality-Channel Shortcut in VLM Emotion Inference — Experiment Sketch

*Novelty verified by targeted prior-art search: the **mechanistic core is OPEN**. The behavioral half (does a
VLM lean on vision or dialogue for emotion?) is partly done by **VIBE** (accuracy only); the internal half —
does ablating a channel **collapse an internal appraisal-probe direction** — is unclaimed. Cite/differentiate
in §7.*

**Scope correction from the search (important):** *MoMentS (2507.04415) is a **Theory-of-Mind** benchmark, not
an emotion dataset.* The genuine multimodal-video **emotion** datasets with a dialogue channel are **MC-EIU**
(VIBE's data), **MELD**, **IEMOCAP**, **CMU-MOSEI**. The channel experiment needs a **dialogue channel**, so
it lives on these — **not** EMOTIC (image-only, no dialogue).

---

## 0. The one-sentence claim

> When a VLM reads emotion from a video with dialogue, does it use the **visual scene** (face/body) or
> **shortcut through the dialogue text**? We **causally ablate each channel** and measure the drop in **(a)**
> emotion-prediction accuracy **and (b)** an **internal linear-probe appraisal/emotion direction** — the
> modality-level analog of the known within-visual "face shortcut."

The novel core is **(b)**: nobody has shown that removing the visual channel *collapses an internal emotion
representation*. Accuracy-only channel ablation (VIBE) can't distinguish "vision unused" from "vision used but
redundant."

---

## 1. Construct & method

- **Internal target:** your existing **appraisal/emotion linear probe** on the LM residual stream (per layer,
  `resid_post` + `attn_out`), trained as in the emotion pipeline.
- **Channel ablations (three input conditions, à la VIBE):** Full (video+dialogue) · Vision-only (drop
  subtitles) · Text-only (drop frames). For each, measure both emotion accuracy **and** appraisal-probe
  decodability/selectivity.
- **The shortcut signal:** if the appraisal direction stays decodable under **Text-only** but **collapses
  under Vision-only**, the internal emotion representation is *built from dialogue* — a modality-level
  shortcut. (And vice-versa.)
- **Causal token-level version (stronger than input ablation):** patch/knock out **visual tokens** vs
  **dialogue tokens** and mediation-analyze how much of the channel→emotion-output effect passes **through**
  the appraisal direction vs routes around it. Directly extends Tak et al.'s mediation to the cross-channel
  setting.

---

## 2. Data

| Dataset | Modality | Emotion labels | Role |
|---|---|---|---|
| **MC-EIU** (VIBE's) | multi-party TV video + dialogue | per-speaker emotion, ~9,125 clips | **primary** — matches the predecessor you beat |
| **MELD** | TV video + dialogue | 7-way utterance emotion | second testbed, classic MER modality-bias anchor |
| **IEMOCAP / CMU-MOSEI** | dyadic video + text/audio | categorical + dimensional | robustness / dimensional check |
| crowd-enVENT (text) | text | appraisal dims | optional: train the appraisal direction text-side for a cross-channel transfer variant |

---

## 3. Analyses

1. **Behavioral channel ablation** (replicate + extend VIBE): accuracy under Full / Vision-only / Text-only.
2. **Internal probe under ablation** (the novel core): appraisal-direction decodability + selectivity in each
   condition; **collapse = shortcut**.
3. **Causal channel mediation:** visual-token vs dialogue-token patching → direct vs through-appraisal effect,
   bootstrap CIs.
4. **Shortcut verdict per model/emotion:** which emotions are vision-grounded vs dialogue-shortcut (subtle
   facial-cue emotions likely most shortcut).
5. **Controls:** shuffled-label selectivity; a vision-only-decidable attribute (e.g., scene/setting) as a
   positive control that the visual channel *is* readable internally.

---

## 4. Contributions

1. **First mechanistic modality-shortcut result for emotion** — internal appraisal-probe collapse under
   channel ablation, not just accuracy (the wedge vs VIBE).
2. **Extends the "face shortcut" (within-visual) to a "dialogue shortcut" (across-modality)** — and from
   correlational attention maps to **causal** probing/patching (wedge vs *Anatomy of a Feeling*).
3. **Ports Tak et al.'s appraisal-mediation machinery to the multimodal cross-channel setting** (wedge vs its
   text-only origin).
4. **Safety framing:** a VLM that appears to "read the room" but is really paraphrasing the subtitles is
   silently blind to visual affect — a deployment risk in assistive/affective systems.

---

## 5. Related work & wedge (verified search)

1. **VIBE: "Can a VLM Read the Room?" (arXiv 2506.11162, 2025)** ✓ — *behavioral predecessor to beat.*
   Visual-only / text-only / both **accuracy** on MC-EIU; finds text ≈/> vision, models misread facial cues.
   Wedge: **no internal probe, no causal mediation, no shortcut framing.**
2. **Tak et al. — "Mechanistic Interpretability of Emotion Inference in LLMs" (arXiv 2502.05489, 2025)** ✓ —
   *method you extend.* Appraisal probes + mediation + steering, **text-only.** Wedge: multimodal, channel-
   conditioned.
3. **"Anatomy of a Feeling" (arXiv 2509.19595, EMNLP'25 Findings)** ✓ — *the shortcut line.* Face-vs-body-vs-
   context **within vision**, via **attention maps (correlational).** Wedge: across-modality + causal + probes.

Adjacent: unimodal-bias-in-MLLMs causal work (2403.18346, VQA not emotion); Modality-Importance Score for
video QA (2408.12763, not emotion/mechanistic); classic MER text-dominance (MELD/IEMOCAP ablations).

**Positioning line:** *"VIBE showed a VLM's emotion answers barely improve with vision; we show why — the
model's internal emotion representation is built from the dialogue channel, and survives removing the picture."*

---

## 6. Feasibility & kill-criteria

- **Feasibility:** reuses your probe/mediation code; the new bits are the channel-ablation input conditions
  and per-channel token patching. Needs a **dialogue-bearing emotion video dataset** (MC-EIU/MELD) — the one
  setup change from EMOTIC.
- **Kill-criteria:** if the appraisal probe collapses equally under *both* ablations → representation needs
  both channels (no shortcut — still a clean result). If it survives both → redundant coding (report as
  robustness, not shortcut). Either outcome is publishable.
- **Integration option:** this is a natural **video capstone for the emotion-appraisal paper** — the same
  appraisal direction, now shown to be dialogue-shortcut on naturalistic video — rather than a standalone.
