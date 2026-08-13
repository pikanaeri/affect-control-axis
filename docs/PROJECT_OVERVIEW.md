# Affect-Gated Refusal in Vision-Language Models — Project Overview

*Mechanistic-interpretability + AI-safety study. Target: NeurIPS / ICLR. Status: in progress (Gemma-3-4B
confirmed; multi-VLM replication underway).*

---

## 1. Premise

**Question.** Does the *affective content* of a multimodal input change whether a vision-language model (VLM)
refuses a harmful request — and if so, is there an internal representation that mediates it, that we can detect
and defend?

The intuition is that safety-tuned models may treat *how* an input *feels* (distressing, sympathetic, benign)
as evidence about *whether* to comply — an "affect gate" sitting upstream of the refusal decision. If such a
gate exists, it is both a **vulnerability** (an attacker who moves the model's affect representation could lower
guardrails) and a **defense surface** (the same representation can be monitored).

**Defensive framing (non-negotiable).** We measure refusal, never harm. The refusal signal is *generation-free*
(first-token refusal-vs-compliance logits) — no harmful text is produced. We use only existing benchmarks
(AdvBench harmful, Alpaca harmless); no attacks are crafted or released. Attack success appears only as
mechanism evidence, and we lead with detection + defense.

---

## 2. Background — the two mechanisms we build on

- **Refusal direction `r` (Arditi et al., [2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024).**
  Refusal in chat LLMs is mediated by a *single* residual-stream direction (difference-in-means of harmful vs
  harmless prompts): ablating it removes refusal, adding it induces refusal. We confirm `r` replicates in a VLM.
- **Affect / valence representations.** Emotion is linearly represented in LM activations; in text LLMs a
  valence-arousal subspace can bidirectionally control refusal (Sun et al.,
  [2604.03147](https://arxiv.org/abs/2604.03147)). Whether an *affect* direction gates refusal **in a VLM**,
  orthogonally to `r`, and reachable through the *image* modality, is the open question this project targets.

---

## 3. Approach & methods

- **Directions.** `r` = mean(harmful) − mean(harmless) text prompts, per layer, last token. `a` (affect) =
  mean(negative EMOTIC images) − mean(neutral images). `a⟂` = `a` orthogonalized against `r`.
- **Norm-scaled steering.** Gemma-3 residual norms are huge (median ≈ 38,000), so interventions are scaled as
  `α × ‖resid‖`; fixed coefficients silently no-op (this caused every early null).
- **Coherence calibration + controls.** Steering strength is calibrated to keep output *fluent* (α ≈ −0.008 on
  Gemma); we always co-report an output-coherence check and a **random-direction control** at matched α to
  separate a concept-specific effect from generic degradation.
- **Generation-free refusal metric.** First-token log-prob mass on refusal openers minus compliance openers.
- **Data.** EMOTIC (distress/sympathy vs neutral people images), AdvBench (harmful), Alpaca (harmless), plus
  benign-emotional images for the positive-affect condition.

> **Data update (2026-07):** the affect axis is now built from **OASIS** valence (200/class), after the EMOTIC
> image download proved incomplete (~184 of ~17k images → a poisoned 1-image negative set). Direction stability
> `a_stab` = 0.63–0.83. All confirmed results below are on this valid axis; older EMOTIC/4B numbers are provisional.

---

## 3.5 Confirmed results (Gemma-3-12B, OASIS axis)

1. **Gate confirmed — clean dose-response.** Steering `−a⟂` collapses refusal **1.00 → 0.00** monotonically
   (α 0.006→0.012) while the **random ⟂ control stays 1.00** (affect-specific), coherent throughout;
   generations flip refuse→comply in a benign/cheerful tone.
2. **Mechanism = mediated through `r`.** Restoring the `r`-projection to baseline while steering affect **recovers
   refusal 0.25 → 1.00** → the effect is entirely routed through `r` (−29% r-projection). A valence route into the
   Arditi direction, not an independent axis.
3. **Model/scale-dependent.** Gate in Gemma-3-12B; absent in Qwen2.5-VL / InternVL3 (good `a_stab`); partial in
   LLaVA-7B; 4B open confound (lowest `a_stab`). `cos(a,r)` does not predict it.
4. **Detection beats clamping.** The affect-projection is a perfect monitor (AUROC 1.00, 100% detect, 0% FP);
   a clamp restores only 0.17 and over-refuses benign (0.06→0.81). **Defend by monitoring, not clamping.**
5. **Input-reachability (§11, mechanism-guided LLM attacker).** An attacker LLM reframes AdvBench prompts with
   affective tone; we measure input-space attack success + axis movement vs. the white-box steer. Expected
   (from the image-null): input affect moves the axis ~100× too little → exposure is **white-box**.

---

## 4. Key findings so far *(EMOTIC-era detail — see §3.5 for the confirmed OASIS results)*

1. **`r` is real and replicates in a VLM.** Adding `0.25×‖resid‖×r` to harmless prompts flips refusal
   **0.12 → 1.00**.
2. **Distressing images do NOT jailbreak.** Refusal stays **0.98–0.99** across all image conditions, including
   positive/benign. The image's affect moves the gate axis only ~11–22 vs the ~2,200 the jailbreak steer needs —
   **≈100× too little.** Perceived negative affect is refusal-*aligned* (appraisal-consistent with Zhou et al.
   [2406.05644](https://arxiv.org/abs/2406.05644)).
3. **But an affect axis coherently gates refusal.** Steering `−a⟂` (toward benign affect) at α ≈ −0.008 drives
   refusal **1.00 → 0.00 while the model stays fluent**; a **random ⟂ direction at the same α leaves refusal at
   1.00** (affect-specific, not degradation). Sign: positive affect jailbreaks; negative affect raises refusal.
4. **Mid-layer, and mediated by `r`.** The gate is strongest in mid layers ([8,20) → 0.00; late [24,34) → 1.00).
   Under the `a⟂` jailbreak the `r`-projection drops ~27%, and ablating `r` makes the jailbreak vanish — `a⟂`
   gates refusal *through* `r` (orthogonality ≠ causal independence, Wollschläger et al.
   [2502.17420](https://arxiv.org/abs/2502.17420)).
5. **Detect & defend.** The affect-projection separates clean vs attacked prompts at **AUROC 1.000**; a one-sided
   clamp keeps harmless intact (over-refusal 0.12 → 0.07) but only partially restores refusal (0.00 → 0.12) —
   **conclusion: monitor the affect axis, don't clamp it.**
6. **Confound resolved — not a massive-activation artifact.** Gemma is the only gate-positive model *and* the
   only massive-activation model. The geometric `cos(a,r)` overlap is ~93% carried by 6 outlier dims (dim 443 in
   33/34 layers) — but the **causal jailbreak survives zeroing those dims** (refusal stays 0.00). Geometry and
   causation dissociate: the causal gate lives in the *distributed* part of `a⟂`.
7. **Model-dependent (replication).** `r` validates in every VLM tested, but the affect gate appears in Gemma-3
   and is absent in LLaVA-OneVision-7B. This is the current frontier: **is the gate a fusion-architecture
   property?** (see §5).

---

## 5. Model coverage

We test across VLM families to ask whether the gate tracks image–text **fusion architecture**. **Tier 1** =
bootable by the TransformerLens bridge; **Tier 2** = needs the model-agnostic Hugging Face `output_hidden_states`
extractor (built into `affect_gate_replication.ipynb`).

| Model | Family | Params | Vision encoder | Image–text fusion | LLM backbone | TL bridge | Tier |
|---|---|---|---|---|---|:--:|:--:|
| Gemma-3-4B-it | Gemma-3 | 4B | SigLIP-400M | Interleaved / full-attention | Gemma-3 | ✅ | 1 |
| LLaVA-OneVision-7B | LLaVA-OV | 7B | SigLIP-SO400M | MLP projector (late fusion) | Qwen2-7B | ✅ | 1 |
| LLaVA-OneVision-0.5B | LLaVA-OV | 0.5B | SigLIP-SO400M | MLP projector (late fusion) | Qwen2-0.5B | ✅ | 1 |
| Qwen3.5-4B | Qwen3.5 | 4B | Qwen native ViT | Native multimodal (dynamic-res) | Qwen3.5 | ✅ | 1 |
| Qwen2.5-VL-7B-Instruct | Qwen2.5-VL | 7B | Qwen2.5 ViT | Dynamic-res + MLP merger | Qwen2.5-7B | ❌ | 2 |
| InternVL2.5-8B | InternVL2.5 | 8B | InternViT-300M | Pixel-shuffle + MLP projector | InternLM2.5-7B | ❌ | 2 |
| Phi-3.5-vision-instruct | Phi-3.5-V | 4.2B | CLIP ViT-L/14 | LLaVA-style MLP projector | Phi-3.5-mini | ❌ | 2 |
| Idefics2-8B | Idefics2 | 8B | SigLIP-SO400M | Perceiver resampler (Q-former-like) | Mistral-7B | ❌ | 2 |
| MiniCPM-V-2.6 | MiniCPM-V | 8B | SigLIP-400M | Perceiver resampler | Qwen2-7B | ❌ | 2 |

✅ = TransformerLens-bridge supported (Tier 1)  ❌ = needs HF backend (Tier 2). The set spans interleaved/full-
attention, MLP-projector late fusion, dynamic-resolution merging, pixel-shuffle, and perceiver/resampler
(Q-former-like) fusion — so `gate present?` can be correlated against fusion type.

---

## 6. Novelty & related work (verified prior-art search, 2026-07)

*A 105-agent deep literature search with adversarial verification (24/25 claims confirmed). Full log:
`tasks/w8lkqbgcu.output`.*

**Overall verdict.** The generic idea "an LLM auto-generates multimodal inputs to break a VLM" is **PARTIALLY
CLAIMED** — but the specific fusion this project occupies — *a mechanism-guided, affect-specific, VLM affect→
refusal gate with detection + defense* — is **OPEN**.

### 6.1 Automated multimodal VLM attacks — SATURATED (differentiate, don't compete)
All black-box / gradient / output-driven and **mechanism-agnostic** (none uses an internal direction):
- **Arondight** ([2407.15050](https://arxiv.org/abs/2407.15050), ACM MM'24) — red-team VLM makes images +
  red-team LLM makes text; 84.5% ASR on GPT-4.
- **IDEATOR** ([2411.00827](https://arxiv.org/abs/2411.00827), ICCV'25) — a VLM writes jailbreak text + diffusion
  makes the image; 94% ASR on MiniGPT-4.
- **JPS** ([2508.05087](https://arxiv.org/abs/2508.05087), ACM MM'25) — co-optimizes adversarial image + text.
- **BAP** ([2406.04031](https://arxiv.org/abs/2406.04031), ECCV'24) — jointly optimizes image+text for fusion VLMs.
- **RTD/RedDiffuser** ([2503.06223](https://arxiv.org/abs/2503.06223)) — LLM-guided diffusion, image-only.

### 6.2 Emotional / affective jailbreaks — ESTABLISHED (text), so lead with mechanism
- **FreakOut-LLM** ([2604.04992](https://arxiv.org/abs/2604.04992)) — stress priming raises ASR +65% relative;
  **text-only, no activation analysis.**
- **Lost in Delusion** ([2606.00975](https://arxiv.org/abs/2606.00975)) — distress suppresses safety up to 4.5×;
  LLM mental-health, not mechanistic.
- **EmoAgent / "Emotional Baby"** ([2508.03986](https://arxiv.org/abs/2508.03986)) — closest multimodal-affect
  result ("harmful even when the visual risk is correctly perceived"). **But its manipulation is emotional
  *text*, not composed adversarial images** (the "autonomous image-attacker" reading was refuted 0-3) — so
  affect-**image** composition remains unclaimed.

### 6.3 Mechanistic competitors — the must-beats
- **Sun et al., Valence-Arousal Subspace** ([2604.03147](https://arxiv.org/abs/2604.03147)) — a 2D VA subspace
  bidirectionally gates refusal (arousal↓ → refusal 20%→86%; arousal↑ → 87%→5%). **LLM-only, text-only,
  activation-steering.** Our VLM + orthogonal-to-`r` + input-reachable extension is the wedge.
- **Subspace Rerouting** ([2503.06269](https://arxiv.org/abs/2503.06269)) — closest *mechanism-guided* attack
  (finds "acceptance subspaces," reroutes via GCG); LLM text, not VLM/affect.
- **Arditi** ([2406.11717](https://arxiv.org/abs/2406.11717)) — the single refusal direction we gate.
  **Wollschläger** ([2502.17420](https://arxiv.org/abs/2502.17420)) — orthogonality ≠ causal independence.

### 6.4 Why is the gate model-dependent? — UNDER-COVERED (our opportunity)
No verified work explains VLM-family dependence via fusion architecture. Build the story on:
- **Chen & Rando** ([2410.03489](https://arxiv.org/abs/2410.03489)) — the **projector** causes cross-model
  non-transfer of image jailbreaks.
- ([2510.01494](https://arxiv.org/abs/2510.01494)) — representation-space attacks don't transfer unless latent
  geometries align; input-space attacks do.
- ([2410.22330](https://arxiv.org/abs/2410.22330)) — VLMs form a shared, modality-agnostic *task vector*.

### 6.5 Unclaimed contributions (the paper's spine)
1. An affect direction **orthogonal to and gating `r` in a VLM** (Sun did LLM-only). ← strongest differentiation
2. **Mechanism-guided** (not black-box) — the attack/probe is guided by the internal affect→refusal gate.
3. **Detector + clamp defense for the affect gate** — *no competitor in the surviving evidence.*
4. The **image-null as a precise mechanistic feature**, plus **fusion-architecture dependence** of the gate.

**Positioning recommendation.** Do **not** headline "an LLM that generates multimodal attacks" (crowded —
Arondight/IDEATOR/JPS/BAP). Lead with **mechanism + defense + fusion-dependence**; treat mechanism-guided attack
generation as a *demonstration that the gate is input-reachable*, not the thesis.

---

## 7. Open questions & next steps

1. **Settle the 4B confound (highest priority).** Re-run with sharper valence (`OASIS_Q=0.20`) → §9 dose-response
   on Gemma-3-4B. If `a_stab` matches 12B's and the gate is still absent, the model/scale-dependence is real;
   if the gate appears, 4B was under-sampled. Decides the model-dependence claim.
2. **Run §11 (mechanism-guided LLM attacker).** Input-space attack success + affect-axis movement vs. the
   white-box steer — confirms whether the exposure is white-box (expected) or input-reachable.
3. **Short-generation validation** of the generation-free refusal proxy on the confirmed gate.
4. **Extend the defense / coverage.** Calibrate the monitor operating point; add a scorable resampler VLM
   (MiniCPM needs a bespoke loader) to close the fusion table.
5. **Draft** method + results toward submission — the mechanism + defense half is complete on trustworthy data.

**Notebook map:** `affect_gate_replication.ipynb` §1–§8 = data + multi-model replication; **§9** = dose-response +
mediation; **§10** = detection + defense; **§11** = mechanism-guided LLM attacker. Deck:
`affect_refusal_results_deck.pptx`. Full results log: `AFFECT_REFUSAL_RESULTS.md`.

---

## Appendix — verified sources

Automated attacks: 2407.15050, 2411.00827, 2508.05087, 2406.04031, 2503.06223, 2404.03027 (JailBreakV-28K
benchmark). Emotional: 2604.04992, 2606.00975, 2508.03986. Mechanism: 2406.11717, 2604.03147, 2502.17420,
2503.06269. Fusion/transfer: 2410.03489, 2510.01494, 2410.22330, 2508.01741. *(arXiv IDs; full verification log
in `tasks/w8lkqbgcu.output`.)*
