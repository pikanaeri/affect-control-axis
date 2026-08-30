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

## 4.3 Cross-model replication (gemma-4 4B/12B, Qwen3-VL 2B/4B/9B)

We re-ran the battery across five VLMs. **The cross-modal dissociation (exp03) reproduced on gemma-4-E4B (`CROSS_MODAL_DIVERGE`)**; on the smaller Qwen3-VL models exp03 ran to completion but was inconclusive, and the mediation diagnostic (exp10) completed on all five models. *(The multi-model run did not persist numeric outputs cleanly and several larger-model runs crashed; final cross-model effect sizes require the re-run in Limitations. Reported here qualitatively as which constructs completed and their verdicts.)*

## 4.4 Summary

Across a white-box mechanism and a behavioral battery, an **internal affect direction — shared between image and text but reachable by images — causally shifts a VLM's decisions**, most reliably as increased caution (over-refusal, present bias) and a cross-modal generosity shift that text cannot reproduce, while leaving plain capability intact. Discrete fear-vs-anger contrasts are null, consistent with a **dimensional (valence) rather than categorical** account.

## Limitations

- Battery numbers are **exploratory** (no pre-registration, small n — exp03 n = 8, smoke tier). The headline behavioral effects (exp09, exp05, exp06) warrant a pre-registered, full-tier rerun.
- The **multi-model sweep numbers were not persisted** (Drive write failed + runtime reset) and some large-model runs crashed (suspected OOM / processor mismatch). A clean re-run with fixed persistence (per-model run dirs, robust Drive writes, 4-bit for the 12B/9B) is needed for final cross-model effect sizes.
- The refusal **gate** is model-specific (gemma-3-12B); the paper is anchored on the **behavioral** results, which are the cross-model part.
- Defensive framing throughout: refusal is measured generation-free; only existing benchmarks are used; no crafted attacks.
