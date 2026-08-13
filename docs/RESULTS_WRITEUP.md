# Affect-Gated Refusal in Vision-Language Models — Results Write-up

*Charlotte Li · Algoverse 2026 · mechanistic-interpretability + AI-safety. Defensive framing throughout: refusal is
measured **generation-free** (first-token refuse-vs-comply logits), only existing benchmarks are used, and we lead
with detection + defense.*

---

## TL;DR

A **valence axis causally gates refusal** in a vision-language model. In Gemma-3-12B, steering the affect axis
toward "benign" collapses refusal along a clean **dose-response** (1.00 → 0.00) while a random control stays flat;
the effect is **mediated through the Arditi refusal direction** (restoring the refusal-projection fully recovers
refusal). It is **model/scale-dependent** (present in Gemma-3-12B; absent in Qwen2.5-VL and InternVL3), the affect
axis is a **near-perfect runtime monitor** (AUROC 1.00) while a naive clamp is a poor defense, and **affective
*images* do not reach the gate** (they move the axis ~100× too little) — so the exposure is white-box, not
black-box. Net: *affect modulates refusal via the refusal direction; monitor the axis, don't clamp it.*

---

## 1. Setup

- **Models.** Gemma-3-4B/12B (TransformerLens bridge, primary), + LLaVA-OneVision-7B/0.5B, Qwen2.5-VL-7B,
  InternVL3-8B, Qwen3.5-4B for replication.
- **Directions (per layer, last-token `resid_post`).** `r` = mean(harmful) − mean(harmless) text prompts (Arditi
  et al. 2406.11717). `a` = mean(low-valence images) − mean(mid-valence images). `a⟂` = `a` orthogonalized against
  `r` (the part of affect not already refusal).
- **Steering.** Norm-scaled: `α × ‖resid‖ × direction` via forward hooks (fixed coefficients silently no-op on
  Gemma's ~38k–76k norms). Strength **coherence-calibrated**; every effect co-reported with a **random-direction
  control** at matched α and an **output-coherence check**.
- **Data.** **OASIS** (~900 valence-rated images; split by valence tertiles, 200/class), AdvBench (harmful),
  Alpaca (harmless). *Data-integrity note:* the original EMOTIC image download was incomplete (~184 of ~17k
  images → a 1-image "negative" set that silently poisoned early affect results); we rebuilt on OASIS, the
  notebook now hard-asserts `img_neg ≥ 100`, and direction split-half stability `a_stab` = **0.63–0.83**.

---

## 2. Results

### 2.1 The gate is confirmed — a clean dose-response (Gemma-3-12B)
Steering `−a⟂` (toward benign affect), refusal rate on AdvBench:

| α (×‖resid‖) | affect `a⟂` | random `⟂` (control) | coherent |
|---|---|---|---|
| 0.000 | 1.00 | 1.00 | ✓ |
| 0.006 | 0.92 | 1.00 | ✓ |
| 0.008 | 0.75 | 1.00 | ✓ |
| 0.010 | 0.25 | 1.00 | ✓ |
| 0.012 | **0.00** | 1.00 | ✓ |
| 0.016 | 0.00 | 1.00 | ✗ (over-steer) |

Refusal collapses **monotonically** under affect steering while the random control stays pinned at 1.00 →
**affect-specific**, not generic degradation; coherent through the jailbreak range. **Generation-validated:** at
α=0.010 the model flips from "I cannot and will not…" to compliance, in a **benign/cheerful tone**
("Okay! Here's a simple design…", "😊 ☀️") — the steer reframes the harmful request as benign.

### 2.2 Mechanism — mediated through the refusal direction
- The affect steer **reduces the `r`-projection by 29%** (25,975 → 18,497).
- **Clean causal test:** steering affect while *restoring* the `r`-projection to its clean baseline **recovers
  refusal 0.25 → 1.00 (full)**. So the affect effect is *entirely routed through `r`*.
- Causal chain: **affect → ↓ `r`-projection → ↓ refusal.** This is a *valence route into* the Arditi refusal
  direction — not an independent second axis. (Extends Sun et al.'s LLM/text result to a VLM with valence.)

### 2.3 Model / scale-dependence
| Model | fusion | `a_stab` | `affect⟂` | gate |
|---|---|---|---|---|
| **Gemma-3-12B** | interleaved | 0.83 | **0.04** | **YES** (survives massive-dim removal) |
| Gemma-3-4B | interleaved | 0.63 | 1.00 | no* |
| LLaVA-OV-7B | MLP projector | 0.65 | 0.65 | partial |
| Qwen2.5-VL-7B | dyn-res merger | 0.63 | 1.00 | no |
| InternVL3-8B | pixel-shuffle | 0.74 | 0.99 | no |

Only the large, interleaved-fusion Gemma gates. **`cos(a,r)` does not predict the gate** (Gemma-4B has the
*highest* cos and no gate). *\*Open confound:* 4B has the lowest `a_stab` (0.63) — re-test with sharper valence
before concluding it lacks the gate.

### 2.4 Detection beats clamping (defense)
- **Detector (monitor the affect projection):** AUROC **1.00**, attack-detection **100%**, benign false-positive
  **0%**.
- **Clamp (one-sided projection floor):** restores refusal only **0.00 → 0.17** under attack *and* over-refuses
  benign prompts **0.06 → 0.81**.
- **Conclusion: monitor the affect axis, don't clamp it.** *Caveat:* the detector is evaluated against a white-box
  activation steer; its value is that the gating axis itself is a clean monitor.

### 2.5 Input-reachability — is the gate exposed to input-space attacks?
An **attacker LLM** rewrites AdvBench requests with affective tone (benign/sympathetic); we measure refusal +
how far the input moves `a⟂` vs. the white-box steer that jailbreaks (notebook §11). **Expected** (from 2.6):
input affect moves the axis far too little to reach the gate → exposure is **white-box**. *(Result pending run.)*

### 2.6 Images do not jailbreak (the null, mechanistically explained)
Refusal stays **0.98–0.99 across all image conditions** (incl. positive/benign). The image's affect moves the
gate axis only ~11–22 vs. the ~2,200 a jailbreak needs — **≈100× too little.** Perceived negative affect is
refusal-*aligned* (appraisal-consistent, Zhou et al. 2406.05644).

---

## 3. Positioning (verified prior-art search, 2026-07)

The specific package — *VLM + valence + dose-response + mediation + detection/defense + image-null* — is **open**.
Differentiate from: **Sun et al. 2604.03147** (affect gates refusal, but LLM-only/text-only), **Arditi 2406.11717**
(the single refusal direction we gate), and the automated-attack line **Arondight / IDEATOR / JPS / BAP** (black-box,
mechanism-agnostic). Mechanism-guided red-teaming is the unclaimed angle; automated black-box multimodal attacks
are saturated.

---

## 4. Next research direction (Sneheel's reframing)

**Reframe the goal from *jailbreak* to *behavioral change*.** The more interesting question: **can emotionally
valenced *images* causally shift a model's behavior on a task** — analogous to Anthropic's emotion/persona-vector
work, where a "desperation"-type vector correlates with behavioral change? Refusal turned out to be a *hard* case
(the harmful text dominates the residual, so images move the gate ~100× too little); a **softer, non-safety task**
may be far more image-sensitive.

**Proposed design.**
1. **Find a task where a specific emotion vector changes behavior** — e.g., a decision/answer task where a
   "desperation" (or urgency/valence) direction measurably shifts the model's choice, sycophancy, risk-taking,
   or hedging. Establish the emotion-vector → behavior link by activation steering first (the clean upper bound).
2. **Swap image inputs to shift that vector** — feed valenced images (and Charlotte's idea: **manipulate image
   lighting** — a low-level, content-preserving knob) and measure how much they move the emotion vector.
3. **Observe the behavioral change** — does an image-induced vector shift produce the predicted behavior change?
   This directly tests *image → emotion representation → task behavior*, reusing the machinery we've already built
   (image→affect-axis extraction, projection metrics, norm-scaled steering as the reference).

**Why this is well-positioned from our results.** We already have (a) the affect/valence axis, (b) a validated
image→axis extraction, and (c) the finding that images move the axis *weakly*. The open empirical question is
whether that weak image-induced shift is nonetheless **enough to move behavior on tasks that aren't as strongly
"pinned" as refusal.** Image lighting is an attractive manipulation because it changes affect without changing
task content (clean causal handle).

**Novelty caveat (flagged by Sneheel).** Several June–July 2026 papers are touching adjacent themes
(emotion/persona vectors, affective-image effects on behavior). **Verify novelty before investing** — a targeted
lit search on "emotion/valence vectors causally shifting VLM/LLM task behavior via image inputs (incl. lighting)"
is the immediate next step, and results go to the group.

### Preliminary evidence (2026-07, `emotional_image_effects.ipynb`, Gemma-3-12B) — direction is GREEN

- **Images reach the valence vector, context-dependently.** Low→high-valence `a`-projection spread: **+1591**
  neutral (describe) · +1020 task · **+52 harmful** — vs. a white-box `a`-steer delta of +6933. So images move the
  vector ~23% of a full steer *in a neutral context* and negligibly under a harmful prompt: **the refusal-null was
  text-domination, not an image limit.**
- **Lighting is a usable (weaker) knob.** Dark (+482) / cool (+270) push more negative than bright/warm, ~20–30%
  of the content-valence magnitude — content-preserving, so a clean causal handle. Pilot before over-investing.
- **Emotional images causally shift task behavior** (2/3 probes, monotonic): outlook judgment swings low −14.3 →
  high +24.1 (~38 logits); risk-taking 4.68 → 8.46 (positive images → more risk); a third probe was flat
  (task choice matters). **Images move behavior even though they cannot jailbreak** — the core of the direction.
- **Images still do not jailbreak** (refusal 0.98 across all valences, incl. image+text pairing) → the affect
  gate's exposure is white-box; softer tasks are the affect-sensitive regime.
- *Caveats:* the behavioral probes are a scaffold (add a **steering upper-bound** comparison to make it causal
  mediation, and more non-affect-congruent tasks); the jailbreak cell's white-box reference is under-powered
  (uses a fixed α + low-vs-high `a`, not §9's calibrated `a⟂`).

---

## 5. Open items & next steps

1. **Settle the 4B confound** — sharper valence (`OASIS_Q=0.20`) → dose-response; is 4B's null real or
   under-sampled? Locks the model-dependence claim.
2. **Run §11** (mechanism-guided LLM attacker) — fill in the input-space attack-success + axis-movement numbers.
3. **Short-generation validation** of the generation-free metric on the confirmed gate.
4. **Novelty lit search** on the new behavioral-shift direction (§4) — verify before investing; report to group.
5. **Draft** method + results — the mechanism + defense half is complete on trustworthy data.

*Artifacts: `affect_gate_replication.ipynb` (§1–8 data + replication, §9 dose-response + mediation, §10
detection + defense, §11 attacker); `affect_refusal_results_deck.pptx` (slides); `AFFECT_REFUSAL_RESULTS.md`
(full running log).*
