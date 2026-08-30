# Results (draft)

*Preliminary results section assembled from what we currently have: the white-box mechanism (gemma-3-12b), Arnav's documented black-box battery (gemma-4-E4B, exploratory), and the cross-model replication grid. Numbers marked **[exploratory]** predate pre-registration; the multi-model sweep numbers still need a clean persisted re-run (see Limitations).*

---

## 4.1 An internal affect axis causally shifts behavior (white-box, gemma-3-12b)

We recover a per-layer **affect/valence direction** `a` (diff-in-means of last-token residuals, low- vs. high-valence images) and steer it with norm-scaled activation additions.

- **Dose-response.** Steering `a` toward "benign" collapses refusal along a clean monotone curve (**refusal 1.00 → 0.00** as α grows), while a **random direction at matched norm stays pinned at 1.00** — the effect is affect-specific, not generic degradation.
- **Causal mediation.** The affect steer reduces the projection onto the Arditi refusal direction by ~30%; **restoring that projection to its clean baseline recovers refusal fully (0.25 → 1.00)**. The affect effect is routed *through* the refusal direction.
- **Soft-task behavior.** On non-safety decision probes the same steer moves behavior monotonically (e.g. outlook judgment **−14.3 → +24.1**; risk-taking **4.68 → 8.46** from negative→positive valence).
- **Images reach the axis, context-dependently.** A valenced image moves `a` by ~15–23% of a full steer in a neutral/task context but only ~0.7% under a harmful prompt — so **images cannot jailbreak refusal (text-dominated) but can move softer behavior.**
- **Model/scale-dependence.** The refusal *gate* appears in gemma-3-12B (interleaved fusion) and is absent in gemma-3-4B, Qwen2.5-VL, and InternVL3; `cos(a,r)` does not predict it. The **behavioral** effects are the robust, cross-model part of the story.

## 4.2 Task-irrelevant affective images shift decisions (black-box battery, gemma-4-E4B) [exploratory]

Each experiment prepends an EMOTIC image to an established decision task and scores a forced choice. Effects are reported as Δ vs. a neutral/no-image baseline with bootstrap CIs.

**Positive findings**
- **Cross-modal dissociation (exp03, dictator).** Real photos shifted generosity (**Δ ≈ −0.98, 95% CI excludes 0, n = 8**), while matched **captions, emotion labels, and "you should feel sad" narratives did not** (CIs include 0). The image channel does something the text channel does not — the model is not simply reading an emotion word off the picture. *(Weak: n = 8, smoke tier.)*
- **Over-refusal / safety calibration (exp09, XSTest).** Benign refusal rose from **~66% (no image) to ~91–94% with any photo — including neutral photos** — a global caution shift rather than improved safety reasoning. *(Strongest single effect.)*
- **Fairness (exp05, ultimatum).** Anger images → more rejection of unfair offers vs. other photos; dictator give-more/less did not hold. *(Partial.)*
- **Present bias (exp06, temporal discounting).** Any photo → more "money now" than the no-photo baseline; no graded emotion ladder. *(Partial.)*
- **Representation mediation (exp10).** A risk shift was present and correlational mediation through the affect direction held (dictator arm invalid). *(Partial; `DIAGNOSTIC_NOT_V3`.)*

**Specificity / negative controls (nulls that strengthen the claim)**
- **Discrete-emotion risk (exp01).** Fear vs. anger on risky lotteries: **Δ ≈ −0.09, CI [−0.50, 0.30] includes 0** — no discrete-emotion effect. Points away from category-specific and toward a valence account.
- **Capability control (exp07, TruthfulQA/MMLU).** Accuracy flat (**~0.68–0.72**) across affect conditions — affect does **not** degrade plain knowledge, so the behavioral effects are not generic distraction.
- **Stimulus control (exp08, EMOTIC vs. OASIS).** Matching nulls across both stimulus sets (do not pool).
- **Sycophancy (exp04, Perez).** Mean Δ vs. neutral includes 0.

## 4.3 Cross-model replication (Gemma-4-E4B, Qwen3-VL-2B, Qwen3-VL-4B)

The full battery ran across five VLMs (Gemma-4 4B/12B, Qwen3-VL 2B/4B/9B). The **three 4B-class models completed the battery**; the 12B and 9B models ran the light experiments but OOM'd on the image-heavy ones (Limitations). Two effects replicate across **both model families**.

**Over-refusal (exp09, XSTest) — the strongest, most consistent effect.** A task-irrelevant photo (any category, including neutral) sharply raises benign refusal:

| Model | no-image refuse | fear-photo refuse | Δ |
|---|---|---|---|
| Gemma-4-E4B | 0.66 [0.60, 0.72] | 0.92 [0.88, 0.96] | +0.26 |
| Qwen3-VL-2B | 0.71 [0.65, 0.76] | 0.97 [0.96, 0.99] | +0.26 |
| Qwen3-VL-4B | 0.80 [0.74, 0.84] | 0.96 [0.95, 0.98] | +0.16 |

CIs are non-overlapping and the direction is identical across families — the model becomes globally more cautious after seeing an unrelated image, not more accurate at safety.

**Cross-modal dissociation (exp03, dictator/perez).** On Gemma-4-E4B (`CROSS_MODAL_DIVERGE`, gates 9/9) the **photo** moves behavior (`perez:pixels −0.69 [−1.23, −0.09] n=16`) while a **narrative** of the same content moves it the other way (`+0.58 [+0.21, +1.00]`) — image ≠ text. Qwen3-VL-2B shows the same image-channel signal (`risk:deaffect +0.80 [+0.06, +1.55]`; `dictator:label −1.12 [−2.09, −0.19]`); Qwen3-VL-4B was inconclusive at n=8.

**Generosity (exp05, dictator).** Small, model-dependent shifts (e.g. Qwen3-VL-4B: anger 56.1 vs. no-image 60.2 — anger lowers giving ~4 pts); no consistent cross-model direction at smoke n.

## 4.4 Summary

Across a white-box mechanism and a behavioral battery, an **internal affect direction — shared between image and text but reachable by images — causally shifts a VLM's decisions**. The most reliable effect is a **global caution shift**: an unrelated photo raises benign over-refusal from ~66–80% to ~92–97%, replicated across two model families (Gemma-4, Qwen3-VL). Behavior also shifts through the **image channel in a way text cannot reproduce** (the cross-modal dissociation), while plain capability is left intact (exp07 control). Discrete fear-vs-anger contrasts are null, consistent with a **dimensional (valence) rather than categorical** account.

## Limitations

- Battery numbers are **exploratory** (no pre-registration, **smoke tier**, small n — exp03 n = 8, exp09 n = 250/condition). The headline effects (exp09 over-refusal, exp03 cross-modal) warrant a pre-registered, full-tier rerun for tighter CIs.
- **exp06 (temporal discounting) is excluded** — its k-estimator returns a degenerate value (identical `k` across conditions on every model), a bug in the estimator rather than a null result.
- **The 12B and 9B models OOM'd** on the image-heavy experiments (exp05–exp09) even with images capped at 512px; the cross-model claim rests on the three 4B-class models (two families). The large models complete the light experiments and can be added with a smaller image cap / lighter batch.
- The refusal **gate** is model-specific (gemma-3-12B); the paper is anchored on the **behavioral** results, which are the cross-model part.
- Defensive framing throughout: refusal is measured generation-free; only existing benchmarks are used; no crafted attacks.
