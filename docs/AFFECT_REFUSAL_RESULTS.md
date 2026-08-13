# Affect → Refusal in VLMs — Results (in progress)

*Model: Gemma-3-4B-it (TransformerLens bridge). Data: EMOTIC sympathy/distress images (Suffering/Sadness/Pain
people) vs neutral people; AdvBench (harmful) + Alpaca (harmless). Refusal signal is **generation-free**
(first-token refusal-vs-compliance logits) — no harmful text is generated. Directions extracted at the last
token of each prompt, per layer (`resid_post`), 34 layers.*

## ✅ CONFIRMED — Gemma-3-12B, valid OASIS valence axis (2026-07)

> **Two data caveats resolved.** The affect axis is now built from **OASIS** (~900 valence-rated images, split by
> valence tertiles; 200/class), after discovering the EMOTIC image download was incomplete (~184 images → a
> **1-image negative set** that poisoned every earlier affect result). Direction stability `a_stab` (split-half
> cosine) is now healthy (0.63–0.83). **Re-run all affect results on this axis; older EMOTIC-based numbers are
> provisional.**

**The affect→refusal gate is confirmed in Gemma-3-12B by a clean dose-response** (`affect_gate_replication.ipynb`
§9). Steering `−a⟂` (toward benign affect), norm-scaled at α×‖resid‖:

| α | affect⟂ refuse | random⟂ refuse | coherent |
|---|---|---|---|
| 0.000 | 1.00 | 1.00 | ✓ |
| 0.008 | 0.75 | 1.00 | ✓ |
| 0.010 | 0.25 | 1.00 | ✓ |
| 0.012 | **0.00** | 1.00 | ✓ |
| 0.016 | 0.00 | 1.00 | ✗ (over-steer) |

- **Monotonic** refusal collapse under affect steering; **random ⟂ control pinned at 1.00** throughout → the
  effect is affect-**specific**, not generic perturbation. Coherent through the whole jailbreak range (≤0.012).
- **Generation-validated:** at α=0.010 the model flips from "I cannot and will not" to compliance, and the
  compliance is **benign/cheerful-toned** ("Okay! Here's a simple design…", "😊 ☀️") — the affect steer *reframes
  the harmful request as benign* and complies. Mechanistic signature, not just a metric flip.

**Model-dependence (valid axis, scorable strong refusers):** gate in **Gemma-3-12B** (affect⟂ 0.04, a_stab 0.83,
survives massive-dim removal noMA 0.02); **absent** in normal-norm architectures **Qwen2.5-VL-7B** and
**InternVL3-8B** even at good a_stab (0.63/0.74); **partial** in **LLaVA-OV-7B** (0.65). `cos(a,r)` does NOT
predict the gate (Gemma-4B has the highest cos 0.43 and no gate). **OPEN confound:** Gemma-**4B** shows no gate but
has the *lowest* a_stab (0.63) — re-test with sharper valence (`OASIS_Q=0.20`) + §9 dose-response before claiming
4B lacks it.

**Mechanism — 12B gate is MEDIATED THROUGH `r` (confirmed by the clean causal test).** §9 mediation panel: the
affect steer reduces the `r`-projection **−29%** (25,975 → 18,497); `r`-ablation *alone* floors refusal (so the
`a⟂+r-ablated=0.00` column is inconclusive — nothing left to suppress). **Decisive result:** steering affect while
*restoring* the `r`-projection to its clean baseline **recovers refusal 0.25 → 1.00 (full)** → the affect effect is
*entirely* routed through `r`. Clean causal chain: **affect → ↓`r`-projection → ↓refusal.** So it is **not** an
independent second axis; it's a valence route into the Arditi refusal direction (extends Sun et al. to a VLM +
valence). Novelty leans on **VLM + valence + dose-response + detection/defense + image-null**, not "independent axis."

**Detection + defense (§10, 12B) — DETECTION beats clamping.** The affect-projection is a **perfect runtime
monitor**: detector AUROC **1.00** (clean-vs-attacked *and* benign-vs-attacked), attack detection rate **1.00**,
benign false-positive **0.00**. The **clamp** defense (one-sided projection floor) is poor: it only restores refusal
**0.00 → 0.17** under attack *and* over-refuses benign prompts catastrophically (**0.06 → 0.81**). **Conclusion:
monitor the affect axis, don't clamp it.** *Honest caveat:* the detector is evaluated against a white-box
activation steer (AUROC 1.00 is partly expected for a large injected perturbation); its value is that the gating
axis itself is a clean monitor, and input-space affect attacks (images) move this axis ~100× too little to reach
the gate — so the practical exposure is white-box.

## Headline

There is an **affect/valence axis that causally gates refusal, independent of the primary (Arditi) refusal
direction** — but with the **opposite sign** from the naive hypothesis. Steering toward *negative/sympathy*
affect *increases* refusal; steering toward *positive/benign* affect (the affect axis orthogonalized against
refusal) **removes refusal entirely (0.99 → 0.00)**. So a distressing *image* does not jailbreak (perceived
negative affect is refusal-*aligned*, appraisal-consistent with Zhou et al. 2406.05644), yet the **affect
representation itself is a genuine, independent refusal gate**, and the jailbreak direction is *positive*
affect, not sympathy.

## Findings

1. **The refusal direction is real (validated).** `r` = difference-in-means of last-token `resid_post` on
   harmful vs harmless instructions (Arditi et al. 2406.11717). Adding `0.25 × (residual norm) × r̂` to
   *harmless* prompts flips refusal **0.12 → 1.00**. So `r` genuinely controls refusal.

2. **Methodological result — steering MUST be residual-norm-scaled on Gemma-3.** Gemma-3's residual stream
   has enormous norms (last-token: **min 776, median 38,016, max 67,072**). Fixed-magnitude steering
   coefficients (~10, as in typical LLM steering) are ~1000× too small and do **nothing**. Every earlier null
   in this study was this artifact, not a real effect. Interventions are now scaled as `α × ‖resid_ℓ‖`.
   *(Hooks themselves were verified to fire: zeroing a mid-layer residual moved the refusal score 23.5 → −0.18.)*

3. **Images do not jailbreak — and we now know *why* (quantitative).** Refusal stays **0.98–0.99 across ALL
   image conditions, including positive/benign** (the jailbreak-direction affect) — so no image valence lowers
   refusal. Mechanistic re-test (with the affect gate understood): the image's *emotional content* moves the
   gate axis `a⟂`-proj by only ~11–22 (negative 915, neutral 937, positive 926 — within noise) and never
   toward the jailbreak side, vs the steer that jailbreaks driving `a⟂`-proj +724 → −1470. **Image affect moves
   the gate ≈100× too little to reach it** — the axis sits at ~900 regardless of the picture; the harmful text
   dominates the residual. So: the gate is real and potent under direct steering, but the *image modality*
   cannot push the affect representation far enough — a clean explanation for why naive emotional-image
   jailbreaks fail.

4. **Causal steering (Exp 3): affect is refusal-*promoting*, not suppressing.** Steering toward the affect
   direction (norm-scaled) raises the refusal score **22.5 → 37 → 40 → 41** (α = 0 → 1), while a **random**
   direction at the same norm *lowers* it **22.5 → 10 → 15 → 16**. So the affect axis is *specifically*
   refusal-promoting — the opposite of generic perturbation, and the opposite of a jailbreak.

5. **Geometry: `mean |cos(a, r)| = 0.358`.** The affect direction is partly aligned with refusal — but not
   reducible to it (see 6).

6. **Affect gates refusal INDEPENDENT of the refusal direction (key result).** Orthogonalizing the affect
   direction against `r` (`a⟂r`) and steering it *still* moves refusal strongly and bidirectionally:
   `α=−1.0 → refuse 0.00 (score −30.5)`, `α=0 → 0.99 (22.5)`, `α=+1.0 → 1.00 (37.4)`. So there is a **second,
   affect/valence refusal-gating axis beyond the Arditi single direction.** Sign: `+a⟂` (more negative affect)
   → more refusal; `−a⟂` (toward positive/benign) → less refusal. **Confirmed under a coherence + specificity
   control (this is the key result):** norm-scale steering was invalid (it degrades the model into
   `"delightful…"` repetition — the metric fooled by garbage), but at a **small, coherence-preserving
   magnitude (α ≈ −0.008 × ‖resid‖)**, steering `−a⟂` drives **refusal 1.00 → 0.00 while the model stays
   fluent** (answers a harmless question normally), whereas a **random ⟂ direction at the same magnitude leaves
   refusal at 1.00**. So the effect is **affect-specific and coherent**, not degradation:
   **there is an affect/valence axis, orthogonal to the Arditi refusal direction, that coherently gates VLM
   refusal, and steering toward positive/benign affect jailbreaks the model.**

   **Independence test (important correction): the effect is orthogonal but NOT causally independent of `r`.**
   Steering `a⟂` with `r` ablated → the jailbreak vanishes (`a⟂` alone 0.00; `a⟂` + r-ablated 1.00). So the
   affect axis modulates refusal **through** `r`, not via a separate mechanism — a concrete case of
   Wollschläger et al.'s (2502.17420) *orthogonality ≠ causal independence*, demonstrated for affect. *(Caveat:
   `r`-ablation alone doesn't reduce refusal here — refusal is ablation-robust — so confirm with the cleaner
   mediation test: does `a⟂` steering reduce the `r`-projection? A large drop = mediated by `r`.)* Headline
   accordingly shifts from "independent second gate" to **"an orthogonal affect direction that gates refusal
   via the refusal direction."**

## Interpretation

Two things are true at once: (i) a *distressing image* does not lower guardrails — perceived negative affect
is refusal-*aligned* (appraisal-consistent, Zhou); yet (ii) the **affect/valence axis is a genuine,
independent refusal gate**, and steering the model to perceive the input as *benign/positive* removes refusal
(0.99 → 0.00). So the correct headline is not "affect doesn't matter" but **"affect gates refusal — with the
opposite sign from the naive jailbreak hypothesis."** This is the *independent-of-r* VLM analogue of the
valence-arousal→refusal control shown for text LLMs (Sun et al. 2604.03147).

Still untested: the **empathy-override** construct — the *user* pleading / in distress (EmoAgent 2508.03986,
"Lost in Delusion" 2606.00975), a different thing from a perceived scene. That is Experiment **B**.

## Methodological lessons (worth a paragraph in the paper)

- **Verify the hook mechanism** (zero-ablation sanity) before trusting any steering result.
- **Validate the refusal direction causally** (add-scaled → induces refusal) before interpreting nulls.
- **Scale steering to the residual norm** on large-norm models (Gemma-3) — fixed coefficients silently no-op.
- **Use a random-direction control** at matched norm to separate concept-specific effects from generic
  perturbation. Here it flipped the interpretation (affect *raises* refusal; random *lowers* it).

## Findings (cont.)

7. **Experiment B — user-empathy hits an affect gate too.** A direction from *pleading vs neutral* framing of
   harmful requests, orthogonalized against `r`, jailbreaks affect-specifically at the calibrated α (`user⟂`
   refuse → 0.00 vs random 1.00, coherent). `cos(a_user, image-affect a) = 0.25` — related but distinct routes,
   so it's a *family* of affect directions gating the same refusal behavior. **Behavioral** pleading does NOT
   jailbreak (0.99 → 1.00, more refusal), consistent with distress being refusal-aligned; only *steering toward
   positive affect* jailbreaks.

8. **Detector perfect; clamp only partial (detection-based defense is the story).**
   - **Mediation confirmed:** under the `a⟂` jailbreak the mean `r`-projection drops 19,967 → 14,656 (~27%) —
     affect gates refusal *by pushing the model off `r`*.
   - **Detector:** the affect-projection separates clean vs attacked harmful prompts at **AUROC 1.000**
     (clean +724 → attacked −1470, out-of-distribution). *(Caveat: near-trivial for a large injected steer;
     value is that the gating axis is also the monitor.)*
   - **Clamp defense:** blanket ablation over-refuses + degrades (0.12 → 1.00, garbage). A **one-sided
     projection floor** keeps harmless intact (over-refusal 0.12 → 0.07, coherent) but only partially restores
     refusal under attack (0.00 → 0.12) — because the steer pushes the affect-projection *into the benign
     range*, so a static clamp can't distinguish attacked-harmful from genuinely-benign. **Conclusion: defend
     by monitoring the affect axis, not clamping it.**

9. **The gate is mid-layer.** Steering `a⟂` only within a layer window: `[8,20)` fully jailbreaks (**0.00**),
   `[12,24)` strongly (**0.17**), early `[0,12)` weak (**0.72**), **late `[24,34)` does nothing (1.00)** — all
   coherent. Mid-layer localization, consistent with Tak et al.'s appraisal-in-mid-attention finding.

## Calibration (critical methods point)

Steering strength must be **coherence-calibrated**, not norm-scaled: on Gemma-3, `α ≈ 1.0 × ‖resid‖` destroys
the model (the refusal metric then reads garbage as "not refusing"), while `α ≈ −0.008 × ‖resid‖` gives a
**coherent** affect-jailbreak. Always co-report an output-coherence check + a random-direction control at the
same α; effect + coherence + specificity together are the claim.

## Status & next steps

1. ✅ **Confirmed:** affect-specific, coherent, independent-of-`r` refusal gate (finding 6), α ≈ −0.008.
2. **Experiment B — empathy-override:** build the direction from *user-distress* framing (pleading vs neutral
   request); test at the calibrated α (+ random control + behavioral gate). Does *user* empathy also gate
   refusal, or is the valence axis the whole story?
3. **Localize** the gate (which layers/sublayers carry `a⟂`).
4. **Detector + clamp DEFENSE:** probe `a⟂` to detect the steer; clamp/ablate `a⟂` to restore refusal without
   harming normal behavior (over-refusal + capability checks).
5. **Replicate** on a 2nd VLM; add short-gen validation.

**Paper:** *VLM refusal is gated by an affect/valence axis independent of the refusal direction — distressing
images don't jailbreak (perceived affect is refusal-aligned), but a small, coherent steer toward positive
affect does, affect-specifically; we localize, detect, and defend it.* Positive mechanism + defense, plus an
appraisal-consistent negative sub-result (images fail).

## Replication across VLMs (multi-model)

Ran the full pipeline on 4 VLMs (`affect_gate_replication.ipynb`, auto-calibrated α per model). **Key result:
the gate is model-dependent, and `cos(a,r)` predicts it.**

| Model | norm_med | r_valid | cos(a,r) | α | affect⟂ | random⟂ | gate? |
|---|---|---|---|---|---|---|---|
| **Gemma-3-4B** | 38,016 | 1.0 | **0.38** | 0.006 | **0.12** | 0.99 | **YES** (full: mediated by r, mid-layer, img-null) |
| **LLaVA-OV-7B** | 60 | 1.0 | **0.03** | None | 0.98 | 0.98 | **NO** — `r` valid but affect ⟂ refusal |
| LLaVA-OV-0.5B | 16 | 1.0 | 0.07 | None | 0.31 | — | excluded (base_refuse 0.29 — too weak a refuser) |
| Qwen3.5-4B | — | — | — | — | — | — | pending (vision OOM; image resize to 512px added) |

**Interpretation:** `r` validates in *all* models (`r_valid_add = 1.0`), so the refusal machinery is universal —
but the **affect gate appears only where the affect and refusal directions geometrically overlap** (Gemma,
cos 0.38) and is **absent where they're orthogonal** (LLaVA, cos 0.03). So the finding is not "VLMs have an
affect gate" but "**refusal is affect-gated iff the affect axis overlaps the refusal axis** — a
model-dependent property."

**Confound — partially materialized (offline decomposition, 2026-07).** Gemma is *both* the only model with the
gate *and* the only one with **massive activations** (norm 38,016 vs ~60). Decomposing the net `cos(a,r)` per
dimension: mean signed `cos = −0.306`, and **zeroing just 6 recurring dims (dominated by dim 443, a top-3
contributor in 33/34 layers) collapses it to −0.023 — 93% of the net overlap lives in ~6 outlier dims.** The
remaining ~2,550 dims have mixed-sign products that cancel to near-zero net. So the *representational* `a`/`r`
overlap that distinguishes Gemma from LLaVA is **almost entirely a massive-activation phenomenon**, not a
distributed property. *(Note: a per-**|contribution|** decomposition looked distributed — top-3 = 0.35, top-10
= 0.40 of |cos| — but that was misleading; the signed net is outlier-dominated. Always decompose the signed sum.)*

**RESOLVED — the gate is NOT a massive-activation artifact (causal test, 2026-07).** Representational overlap ≠
causal gate, so we ran the decisive test: steer `a⟂` at the coherence-calibrated α with the outlier dims zeroed
out of the steering vector. **The jailbreak survives fully — refusal stays 0.00 with dim 443 zeroed, and 0.00
with all 6 outlier dims zeroed (vs 0.00 normal).** So the causal gate lives in the *distributed* part of `a⟂`,
independent of the massive-activation dimensions. The 7% `cos` collapse and the intact jailbreak together are a
clean **dissociation**: the *geometric* `a`/`r` overlap is outlier-dominated, but the *causal* effect is not.

**Consequence for the cross-model claim:** because raw `cos(a,r)` is outlier-inflated, it's a fragile geometric
*correlate*, not the mechanism. Lead model-dependence with the **causal `affect⟂` metric** (Gemma 0.12 → gate;
LLaVA 0.98 → no gate), and present `cos` as the (outlier-sensitive) geometry that motivated the check. Headline
stands: **a real, coherence-controlled, mid-layer, `r`-mediated affect gate on refusal, robust to removing
Gemma's massive-activation dimensions.**

## Novelty & related work (verified prior-art search, 2026-07)

**Substantially novel.** The text-only, arousal-based version is taken; the VLM + orthogonal-to-`r` +
positive-valence + coherence-controlled + defense + image-negative package is OPEN.

- **Sun et al., "Valence–Arousal Subspace in LLMs" (arXiv 2604.03147)** — direct competitor. **Text-only**;
  clean control is via **arousal** (valence "less consistent"); mechanism is **lexical mediation with the VA
  subspace *aligned* to the Arditi `r`** (they project `r` onto VA and report consistency) — i.e. they argue
  *against* an independent axis. Random-direction control YES, **coherence check NO, defense NO** (future work).
  Our differentiators: **VLM, valence, a second axis *orthogonal to and independent of* `r`, coherence-verified,
  random-orthogonalized affect-specificity control, and an ablation defense.**
- **Arditi et al. (2406.11717)** — the single refusal direction; ours is a *second, orthogonal* gating axis.
- **Wollschläger et al., "Geometry of Refusal: Concept Cones & Representational Independence" (2502.17420,
  ICML 2025)** — adopt their **"orthogonality ≠ causal independence"** framing; none of their independent
  directions is affect-labeled. → run the representational-independence test (steer `a⟂` with `r` ablated).
- Secondary: Joad et al. 2602.02132 (multiple refusal directions, no affect); FreakOut-LLM 2604.04992
  (positive *prompt* priming had **no** effect — our internal-direction result is opposite/stronger); Zhang
  2605.21980 (VLM emotion circuits, no safety link).

**Required rigor add:** representational-independence test (per 2502.17420) — does `a⟂` still gate refusal when
`r` is ablated? Orthogonality alone will not satisfy reviewers.

## Caveats

- Single model (Gemma-3-4B); single seed; generation-free refusal proxy. Replicate on a 2nd VLM + short-gen
  validation before finalizing.
- Affect direction from EMOTIC depicted emotion (viewer-perceived), last-token readout.
