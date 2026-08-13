# Cross-Modal Transfer of Belief / Theory-of-Mind Directions in VLMs — Experiment Sketch

*Provisional (pending the targeted novelty search — dataset availability and the exact closest-competitor
differentiation may adjust this). The design is a **label-swap** on the emotion-appraisal pipeline you already
have: `appraisal → belief-state`, `crowd-enVENT → text ToM`, `EMOTIC → visual ToM`. Probes, matched-readout
protocol, subspace battery, and steering are reused verbatim.*

---

## 0. The one-sentence claim

> A vision-language model's **belief-tracking direction**, learned from **text** false-belief stories,
> transfers to **decode and steer** the belief it infers from an **image/video** — i.e. theory-of-mind
> geometry is (or isn't) modality-shared.

Precedent to extend, not compete with: **Zhu et al. 2024, "Language Models Represent Beliefs of Self and
Others" (arXiv 2402.18496, ICML 2024)** — separable, decodable, steerable belief directions, **text-only**.
We ask whether that geometry survives when the belief must be read from a scene.

**Novelty status (verified by targeted prior-art search): OPEN.** The exact composition — train a belief
direction on **text** activations, transfer it to **decode + steer** belief inferred from **image/video**,
and quantify the **cross-modal subspace geometry** — is unclaimed. Every ingredient exists separately; nobody
has assembled this intersection. See §8 for the three papers we must cite and beat.

---

## 1. The construct & the probe target (what replaces "pleasantness")

Classic false-belief (Sally-Anne) structure gives three probe targets, in increasing richness:

1. **False-belief indicator** (binary): does the agent hold a *false* belief (didn't witness the move) vs a
   true belief? — the simplest gate.
2. **Belief content / location** (categorical): *where does the agent think the object is* (A vs B)? — the
   steerable direction; "make the model believe the agent thinks A."
3. **Self vs other** (binary): the model's **own** knowledge (reality) vs the **agent's** belief — the
   self/other split from Zhu et al., extended to vision.

**The validating dissociation (the crux):** on false-belief items, *belief ≠ reality*. A genuine ToM
direction must track the **agent's (false) belief**, not the ground-truth location. So we train two probes —
a **belief** probe and a **reality** probe — and show on false-belief items they **diverge**, with behavior
following belief. A probe that only tracks reality is a location detector, not a mental-state representation.
This is the strongest evidence the direction is really ToM.

---

## 2. Data

| Side | Role | Dataset | Why | Labels used |
|---|---|---|---|---|
| **Text** (train the direction) | supervision | **ToMi** (procedural Sally-Anne), **BigToM** (causal template), OpenToM | templated → clean ground-truth belief location + true/false-belief flag per item | belief location, false-belief flag, self/other |
| **Image/Video** (test transfer) — *primary* | held-out modality | **GridToM** (arXiv 2506.14224, ICML 2025) | **Best fit.** 2D gridworld, 1,296 samples, multi-perspective (omniscient/protagonist/participant) as **video + text**; explicit **initial / first-order / second-order belief labels per perspective** — the authors already used it for per-item belief probing. Clean labels + visual channel in one. | belief location, order-1/2 belief, self/other |
| **Image/Video** — *secondary testbed* | generalization | **MMToM-QA** (ACL 2024, belief-inference split: 300 belief Qs ×3 subtypes, 134 labeled videos) | second, more naturalistic visual testbed; belief-inference items give usable labels | belief location |
| **Image/Video** — *tertiary* | stress test | **MoMentS** (arXiv 2507.04415, Findings EMNLP 2025) | 2,300+ MCQs, 7 mental-state categories; labels coarser (per-question) — use as hard generalization check, not primary probe supervision | belief (coarse) |

**Big feasibility win from the search:** GridToM *already* provides clean per-item belief labels with a
visual channel, so the "curate our own visual Sally-Anne set" labor item from the first draft is **largely
eliminated** — GridToM is our probe testbed. (Note the irony: GridToM is also our closest competitor — see
§8 — so using *their* dataset to make the cross-modal-transfer + geometry point they didn't is a clean,
honest, and strong position.) Text side stays ToMi/BigToM (clean labels there are what matter for *training*
the direction). Both GridToM and MMToM-QA are video-capable but also have static/text-paired forms, so we can
start with frames before committing to full video.

---

## 3. Extraction — identical harness, carry the readout lesson

- **Backbone:** VLM's LM residual stream (`resid_post`) **and** `attn_out`, per layer, via TransformerLens
  bridge (or the HF `output_hidden_states` backend for unsupported models).
- **Readout MUST match across modalities** (the Direction-A/B confound from the emotion work). Present
  scenario + the **same belief question** ("Where does {agent} think the {object} is?") and read the
  answer-prep (last) token on **both** modalities → `READOUT="task"` analog. Never text-without-question vs
  image-with-question.
- **Position/site sweep:** same `SITE ∈ {resid_post, attn_out}` extraction so we can locate where ToM lives
  (Tak et al. found emotion in mid-layer MHSA; ToM may localize differently — an interesting finding either
  way).

---

## 4. Analyses (mirror the emotion pipeline, same code)

1. **Behavioral gate (calibration).** Can the model *do* text ToM (ToMi accuracy) and *visual* ToM (MMToM-QA
   number)? If it can't infer belief behaviorally, the probe is uninterpretable — gate first.
2. **Within-modality decodability (gate).** Per-layer probe for belief location/false-belief on text, and
   separately on image; AUROC/R² + **shuffled-label selectivity**. Both must pass before transfer means
   anything.
3. **Belief-vs-reality dissociation (validation).** On false-belief items, show the belief probe tracks the
   agent's belief and *not* reality (and vice-versa for a reality probe). This certifies the direction is ToM.
4. **Cross-modal transfer (headline).** Train on text belief activations → apply to image belief activations;
   bidirectional; Spearman/AUROC vs a **permutation null**. High = shared ToM axis; ~0 = modality-bound.
5. **Subspace geometry.** Principal angles / Procrustes / RSA between text-belief and image-belief subspaces;
   frame with the **Γ (shared bimodal) / Ω (modality-specific)** decomposition — is belief in Γ?
6. **Self/other split in vision.** Angle between "own-knowledge" and "agent-belief" directions; does Zhu
   et al.'s text-only separation hold under image input?
7. **Causal steering (payoff).** Inject the **text-derived** belief-location direction during **image**
   inference → does the model's ToM answer flip toward the steered location? Dose-response curve. This is
   cross-modal **belief steering** — a capability the field hasn't shown.

**Controls:** split-half positive control; a non-ToM **semantic** direction (e.g., object identity) as a
"this transfers" positive control for the instrument; reality-probe as the dissociation control.

---

## 5. Contributions

1. **First cross-modal *transfer* test of a mental-state (belief) direction in a VLM** — train on **text**,
   read out under **image/video**. This is the exact axis GridToM (our closest competitor) does *not* touch:
   it trains probes on **joint** multimodal activations, never text→image transfer.
2. **First cross-modal *geometry* of belief** — principal angles / Procrustes / Γ-vs-Ω between text- and
   image-derived belief subspaces. GridToM's geometry viz is within a single modality; this has no precedent
   for the belief construct.
3. **Cross-modal belief *steering*** — inject the text-derived direction under visual input; method + causal
   evidence (GridToM steers on joint text+video, not isolated visual inference).
4. **Self/other belief split extended to the visual modality** (extends Zhu et al. 2402.18496, text-only).
5. **Belief-vs-reality dissociation** as a validation protocol for ToM probes in VLMs.
6. **Safety framing:** self/other separation is the substrate of deception/sycophancy — "does the model keep
   *its* knowledge separate from the character's when reasoning from an image?" (motivated by "Split Beliefs,"
   arXiv 2603.18373).

---

## 6. Models & timeline

- **Models:** the existing sweep — a bridge-supported native-multimodal (Qwen3.5), a projector-fusion
  (LLaVA-OneVision / InternVL), a scale point (Gemma). Start with whichever passes the ToM behavioral gate.
- **4–6 weeks:**
  - Wk 0–1: harness + datasets; text + visual ToM **behavioral gate**.
  - Wk 1–2: text belief probes (ToMi/BigToM); **belief-vs-reality dissociation**; self/other split.
  - Wk 2–4: **cross-modal transfer** + subspace geometry + permutation nulls (both sites).
  - Wk 4–6: **causal belief steering** under image input; write-up.

---

## 7. What would make it fail / kill-criteria (be honest early)

- Model fails the **visual ToM behavioral gate** → no interpretable probe (switch model or simplify to
  curated visual Sally-Anne).
- No visual dataset exposes clean per-item **belief labels** → must curate a small set (the one real labor
  item; budget for it, like the emotion appraisal-annotation item).
- **Belief probe = reality probe** (no dissociation) → the "ToM direction" is a location detector; report as a
  negative result about VLM ToM representations (still publishable, but reframes the paper).

---

## 8. Related work & the exact wedge (verified prior-art search)

**Verdict: the specific composition is OPEN.** Ingredients exist separately; the intersection is unclaimed.
Three papers to cite and differentiate:

1. **GridToM — "From Black Boxes to Transparent Minds" (arXiv 2506.14224, ICML 2025)** — *closest competitor.*
   Trains logistic probes on **attention-head** activations of multimodal backbones (LLaVA-Next-Video,
   Qwen2-VL), shows belief/perspective are linearly decodable, and steers via additive head shifts to *boost*
   ToM. **Wedge:** probes trained on **joint multimodal** activations (no text→image **transfer**); geometry
   viz is **within a single modality** (no cross-modal principal angles); steering on joint text+video, not
   isolated visual inference. We use *their* dataset to make the transfer + geometry point they didn't.
2. **"Textual Steering Vectors Can Improve Visual Understanding in MLLMs" (arXiv 2505.14071)** — *closest on
   mechanism.* Derives directions from the **text-only backbone** (SAE / mean-shift / linear probe) and
   applies them to the MLLM to improve visual tasks. **Wedge:** construct is **spatial/counting**, not
   belief/ToM; no belief-subspace geometry. This is literally our transfer mechanism applied to a social
   construct instead of a perceptual one.
3. **Zhu et al. — "Language Models Represent Beliefs of Self and Others" (arXiv 2402.18496, ICML 2024)** —
   text-only origin of belief directions + self/other split; we extend it into vision.

Secondary: **"Task Vectors are Cross-Modal" (2410.22330, ICML 2025)** — modality-invariant *semantic* task
vectors (not social); **"Split Beliefs / To See or To Please" (2603.18373)** — VLMs hold different beliefs
from image vs text input (behavioral) — motivates *why* text↔image belief directions might diverge, i.e. why
the transfer test is non-trivial.

**One-line positioning:** *"GridToM showed belief is linearly present inside a VLM; we show whether it is the
**same** belief geometry the model already learned from text — by transferring the text direction into vision,
steering with it, and measuring the subspace angle."*
