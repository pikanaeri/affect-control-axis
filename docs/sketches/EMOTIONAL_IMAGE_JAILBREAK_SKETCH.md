# Emotional-Image Jailbreaks & the Affect→Refusal Mechanism in VLMs — Experiment Sketch

*Novelty verified by deep prior-art search (2026-07): **OPEN in full form**, but the space is fast-moving with
two close competitors (EmoAgent 2508.03986 behavioral; Valence-Arousal Subspace 2604.03147 mechanistic-but-
text). The two unclaimed contributions are (1) the **image's affective semantics** as the jailbreak vector and
(2) the **causal chain image→affect representation→refusal-direction gating + defense**. Lead with those two;
do NOT pitch it as "emotional images jailbreak VLMs" (behavioral, scoopable). See §8.*

**Responsible-research framing (non-negotiable, and what makes this publishable):** lead with **detection +
defense**; use **existing red-team/refusal benchmarks** as the harmful-request source — do **not** craft or
release new attacks; report attack success only as evidence the mechanism is real; include a
responsible-disclosure statement. The contribution is a *mechanism + defense*, not an exploit.

---

## 0. The one-sentence claim

> In a vision-language model, the **affective semantics of an image** (a distressing/sympathetic/fear scene —
> not adversarial pixels, not text-in-image) shift an internal **affect/appraisal representation** that
> **causally suppresses the refusal direction** (Arditi et al. 2406.11717), lowering guardrails on an
> accompanying request — and **probing that representation detects it while clamping it defends.**

Two novel pieces: **(1)** image-borne affect as the vector; **(2)** the causal mediation `emotional image →
affect rep → refusal-direction gating`, with detection + ablation defense.

---

## 1. Constructs & directions (what we probe/steer)

- **Refusal direction `r`** — extracted the standard way (Arditi): mean activation difference between
  harmful and harmless *text* instructions, per layer; validated by ablation (removing `r` breaks refusal)
  and addition (adding `r` induces refusal). This is the **target** the affect is hypothesized to gate.
- **Affect/appraisal direction `a`** — your existing appraisal/valence direction (crowd-enVENT text +
  EMOTIC), extended to a "distress/negative-affect" axis. This is the hypothesized **mediator**.
- (Optional) **Affective-ToM / user-distress direction** — the model's estimate of the *requester's* distress
  (for the empathy-override variant, §6).

---

## 2. Stimuli (reuse existing benchmarks; do NOT craft attacks)

- **Harmful requests:** an **existing** refusal/red-team benchmark (e.g. AdvBench-style / a standard refusal
  eval) — the *text* request is held fixed and standard.
- **Affective images (the manipulated variable):** emotionally evocative but otherwise benign photos —
  distress, grief, sympathy, fear — drawn from **existing affective datasets** (EMOTIC negative-valence
  subset; IAPS/NAPS/GAPED psychometric evoked-emotion sets, which give calibrated valence/arousal).
- **Neutral-image control:** valence-neutral images matched for complexity/objects.
- **Key: the harmful text is identical across conditions; only the paired image's affect changes.** That
  isolates *image affect* as the causal variable and cleanly separates you from content/typographic jailbreaks.

Factorial: {harmful, harmless request} × {negative-affect, neutral image} × affect intensity (from the
psychometric valence rating). Split by image source to avoid leakage.

---

## 3. Analyses (behavioral → correlational → causal → defense)

1. **Behavioral gate (does the effect exist?).** Refusal rate on the harmful benchmark under
   negative-affect vs neutral images. A drop = the phenomenon exists. (This alone replicates nothing prior —
   no one has done *image-affect* → refusal.) Report intensity dose-response using the valence ratings.
2. **Representation link (correlational).** Probe the affect direction `a` and the refusal direction `r`
   per layer; does negative-affect image input raise `a` and lower the projection onto `r`? Track the
   **`a`↔`r` relationship** across layers.
3. **Causal mediation (the core result).** Does the affect representation *mediate* the image→refusal-drop
   effect? Two complementary tests:
   - **Interchange / patching:** patch the affect subspace from a negative-affect run into a neutral run —
     does refusal drop *as if* the image were emotional? Inverse: patch neutral affect into an emotional run —
     is refusal *restored*? Report a **mediation ratio** (fraction of the total image-induced refusal change
     restored by the affect intervention), bootstrap CIs.
   - **Double dissociation:** steering `a` (distress↑) should lower refusal on harmful items **without**
     changing behavior on neutral/harmless items; steering an unrelated direction should not move refusal.
   Use interchange-intervention accuracy, **not** probe accuracy, as the headline (per the rigor bar).
4. **Refusal-direction gating (mechanism specificity).** Show the affect effect operates *through* `r`: when
   `r` is ablated, does the affect manipulation lose its effect on compliance? (If yes → affect gates the
   Arditi direction specifically, the precise novel claim vs 2604.03147's lexical-mediation story.)
5. **Detection.** A linear probe on `a` (or the `a`-onto-`r` projection) as an **early-warning detector** that
   an input is pushing the model toward affect-driven compliance; ROC vs the neutral control.
6. **Defense (the payoff).** **Clamp/ablate `a`** (RepE / circuit-breaker style) at inference and show refusal
   is **restored** under emotional images — while **emotion *recognition* is preserved** (the model can still
   correctly say what emotion the image shows). Compare against: no defense; generic refusal-direction
   steering (2602.07013-style); JRS-style representation-shift removal (2603.17372). The win = refusal
   restored *specifically on affect-driven cases*, no over-refusal on benign emotional images, no capability loss.

---

## 4. Metrics

- Refusal / attack-success rate by condition + intensity (behavioral).
- Mediation ratio + interchange-intervention accuracy (causal, headline).
- Detector ROC-AUC.
- Defense: refusal restoration on affect-driven cases; **over-refusal rate on benign emotional images**
  (false-positive safety cost); general-capability retention; emotion-recognition retention.
- Controls: neutral-image, harmless-request, random-direction, shuffled-label; activation-norm / cosine-drift
  / entropy vs steering strength (off-distribution check).

---

## 5. Contributions

1. **First image-affect jailbreak vector** — emotional *semantics* of a benign image, not adversarial pixels /
   typography / harmful content (wedge vs EmoAgent 2508.03986 [text personas] and JRS 2603.17372 [harmful
   content]).
2. **First causal `affect → refusal-direction` mediation**, multimodal and stimulus-induced (wedge vs
   2604.03147 [text-only, not attack-framed, no Arditi direction] and Zhou 2406.05644 [correlational]).
3. **Affect-specific detector + clamp defense** that restores refusal without harming emotion recognition
   (wedge vs generic refusal steering / JRS-Rem).
4. **Recognition-vs-adoption safety principle:** a robust VLM should *recognize* image emotion without letting
   it *override* the safety policy; the jailbreak collapses that separation, and orthogonalizing them defends.

---

## 6. Variants / extensions

- **Empathy-override (affective-ToM):** make the mediator the model's estimate of the *requester's* distress
  (image of a distressed person + "help me…") and test whether *user-distress modeling* gates refusal.
  Behavioral surface already shown (Lost in Delusion 2606.00975 up to 4.5×; Adaptive Capitulation 2607.19629)
  — you add the mechanism.
- **Cross-modal transfer of the mechanism:** is it the *same* affect direction whether distress arrives via
  image or text? (Ties to the team's cross-modal appraisal work.)
- **Persistence/accumulation:** does the affect state persist across turns and compound safety drift?

---

## 7. Models & timeline

- **Models:** bridge-supported VLM (start Gemma-3-4B / a Qwen3.5-VL / LLaVA-OneVision); replicate on a second
  architecture. Refusal-direction extraction is cheap and standard.
- **~4–6 weeks:**
  - Wk 0–1: refusal-direction + affect-direction extraction; behavioral gate (affect vs neutral image).
  - Wk 1–3: representation link + **causal mediation** (patching, double dissociation, mediation ratio).
  - Wk 3–4: refusal-direction gating specificity; detector.
  - Wk 4–6: **clamp defense** + baselines + over-refusal/capability/recognition retention; write-up + disclosure.

---

## 8. Related work & wedge (deep search, 2026-07)

| Paper | What it is | Your wedge |
|---|---|---|
| **"The Emotional Baby Is Truly Deadly"** (Xun et al.) 2508.03986 ✓ | closest **behavioral** — affective prompts hijack MLRM reasoning; per an earlier deep read the affect vector is emotional **text** personas, images are the harmful queries | affect is **in the image**; you add the **mechanism** (verify its modality directly — see note) |
| **Valence-Arousal Subspace in LLMs** 2604.03147 ✓ | closest **mechanistic** — VA subspace causally controls refusal/sycophancy | **text-only**, not attack-framed, no Arditi direction, no image induction |
| **How Alignment & Jailbreak Work** (Zhou) 2406.05644 ✓ | emotion-in-mid-layers → refusal, **correlational** | causal steering + **external affective induction** |
| **JRS** 2603.17372 ✓ | image→rep-shift→refusal-failure + clamp defense | their shift = harmful **content**, not **affect** |
| **Arditi et al.** 2406.11717 ✓ | refusal = a single direction | the direction you show **affect gates** |

Secondary / motivation: Persona Vectors 2507.21509 (trait↔refusal geometry); Lost in Delusion 2606.00975 /
Adaptive Capitulation 2607.19629 (distress→compliance, behavioral); refusal-direction steering in VLMs
2602.07013; **"EmoAgent: Assessing and Safeguarding Human-AI Interaction for Mental Health Safety" (Qiu et al.)
2504.09689** — *motivation, NOT a competitor*: emotionally engaging AI harms vulnerable users (>34.4%
deterioration) and its EmoGuard monitors the *user's* mental state externally — cite as the behavioral,
external-guardrail counterpart to our *internal* representation-level detector/clamp.

> **NAME-COLLISION WARNING:** two different papers call their system "EmoAgent" — 2508.03986 (Xun et al., the
> jailbreak competitor above) and 2504.09689 (Qiu et al., the mental-health motivation cite). **Never cite the
> bare name "EmoAgent"; always use the title + arXiv id.**

**Sourcing caveat:** the ✓ five were read in full; verify the modality of 2508.03986 (text vs image affect)
and the snippet-only secondary papers before citing.

**Positioning line:** *"2508.03986 showed emotional text can talk a VLM into harm, and JRS showed images move
the refusal representation — we show the **emotion in a picture** does it, localize the affect representation
that **gates** refusal, and **clamp it to defend**."*

---

## 9. Kill-criteria (be honest early)

- **No behavioral drop** (affect images don't lower refusal) → report as a robustness/negative result (still
  useful: "VLM refusal is invariant to image affect").
- **Effect exists but not affect-mediated** (patching `a` doesn't move refusal) → the drop routes around the
  affect rep; investigate the direct visual→refusal path instead.
- **Defense causes over-refusal on benign emotional images** → the affect direction is entangled with benign
  emotion; report the entanglement and pursue a more surgical (subspace) clamp.
