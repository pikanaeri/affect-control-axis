# Incidental Visual Affect Shifts VLM Behavior

**A white-box mechanism + a behavioral battery.**
*Algoverse 2026 · Charlotte Li (white-box mechanism), Arnav & team (black-box battery). Working base — de-jargoned, numbers separated into confirmed / clean-flagged / pending.*

---

## 1. The claim, in one paragraph

Show a vision-language model an emotional picture that has **nothing to do with the task** — a distressed person, a cheerful scene — then ask it an ordinary decision question (how harsh a moral judgment, how hopeful an outlook, how much risk to take). The picture measurably changes the answer. We show this is not the model "reading the emotion word off the image and role-playing." Instead, the image nudges a specific **affect direction inside the model's activations**, and *that internal shift* is what moves the behavior. We can drive the same behavior change directly by pushing that direction with a steering vector (the causal upper bound), the image reproduces a fraction of it, and — for refusal, where we have the full test — restoring the direction to baseline removes the effect. The effect is **carried by valence** (a positive↔negative axis), **not by discrete emotion labels** (fear vs. anger do not differ), and it is **model-dependent**.

**Why it matters (safety framing).** A deployed multimodal model sees incidental imagery all the time. If unrelated affective content silently re-weights its decisions through an internal affect representation, that is a controllability and robustness issue — and the same axis that carries the effect is a clean place to *monitor* for it.

---

## 2. What we measure and how (plain terms)

- **The affect axis `a`.** For each layer, take the model's last-token activation on low-valence vs. mid-valence inputs and subtract the averages. The difference is a direction that points from "positive/neutral" toward "negative" affect. We build it from **text** (as the causal reference) and from **images** (the naturalistic manipulation), and check the two align.
- **Steering (the causal upper bound).** Add `α · ‖activation‖ · a` at every layer with a forward hook. This pushes the model's affect representation by a controlled amount without touching the prompt. Norm-scaling matters: Gemma's activation norms are ~38k–76k, so fixed-size vectors silently do nothing — we scale to the norm.
- **Behavior score.** First-token option-logit: `logit(option A) − logit(option B)` (e.g. "harsh" vs. "lenient"). No text is generated — the metric is generation-free.
- **The three tests that make it causal (the "triad").**
  1. **Steer → behavior.** Does pushing the axis move the score? (upper bound)
  2. **Image → axis.** Does the picture move the axis, and how far vs. a full steer?
  3. **Restore → null.** Steer the image-induced shift, then *restore the axis projection to its clean baseline* — if the behavior effect disappears, the axis is the causal route (mediation).
- **Controls, every time.** A **random direction** at matched norm; an **output-coherence** check (steered text still reads sensibly); **massive-activation removal** (zero the 1 outlier dimension, `a⟂_noMA`, so the effect isn't one dominant unit); **split-half stability** of the axis (`a_stab`).

**Data.** OASIS (~900 images with *viewer-elicited* valence ratings, split into low/mid/high tertiles) for affect; AdvBench / Alpaca for the refusal case. OASIS is chosen deliberately: its labels are what the image *evokes in a viewer*, not what the depicted person feels — the right construct for "does this image shift the model."

---

## 3. White-box mechanism (the spine)

### 3.1 Steering the affect axis moves behavior — cleanly, for most constructs
Pushing the text-valence axis negative vs. positive and reading the option-logit (Gemma-3-12B, `α=0.008`, n=6/condition):

| Construct | Steer effect | Random control | Verdict |
|---|---|---|---|
| moral_harshness | **31.5** | 10.4 | clean (3.0×) |
| interpretation_bias | **29.6** | 3.8 | clean (7.7×) |
| prosocial_helping | 14.8 | 4.0 | clean (3.7×) |
| confidence | 13.9 | −2.6 | clean |
| sentiment_outlook | 10.9 | −9.4 | clean |
| risk_estimation | 6.5 | 8.5 | **fails — random exceeds** |

Five of six constructs move under affect steering well above the random control → **affect-specific, not generic perturbation**. `risk_estimation` is **flagged**: its random-direction control is larger than the real effect, so it cannot carry a causal claim from steering alone (its image effect below is real, but the mechanism is unproven for it).

### 3.2 Images move behavior too — and dose-responsively
Priming with distress vs. positive images (same option-logit), and a graded low/neutral/high image sweep ([expB](../results/expB_behavior_gemma-3-12b-it.json)):

- **Image prime effect** (distress − positive): sentiment_outlook **+53.8**, risk_estimation **+30.9**, moral_harshness **+19.6**; weak for interpretation/confidence/prosocial.
- **Monotonic image dose-response** on soft tasks: "outlook hopeful/bleak" moves **−11.2 → +14.9 → +25.7** across low→neutral→high image valence; "risky plan" **+4.7 → +6.3 → +8.5**. Higher-valence image → more hopeful / more risk-tolerant, smoothly.

**The flagships** — where the axis *cleanly steers* the behavior **and** images *move* it — are **moral_harshness** (steer 31.5 clean; image 19.6) and **sentiment_outlook** (steer 10.9; image 53.8). These are where the full mediation test should be run.

### 3.3 How far do images reach the axis? (context-dependent)
Image-induced spread of the affect projection vs. a full white-box steer (Δ≈6933) ([reachability](../results/expA_reachability_gemma-3-12b-it.json)):

| Context | Image moves axis | as % of full steer |
|---|---|---|
| Neutral ("describe") | 1591 | ~23% |
| Task ("give an opinion") | 1020 | ~15% |
| Harmful (AdvBench) | 52 | ~0.7% |

Images reach the axis substantially in ordinary contexts and negligibly under a harmful prompt. **This is the key reconciliation:** images cannot jailbreak (the harmful text pins the residual) but *can* move behavior on softer tasks that aren't so strongly pinned. The refusal-null was text-domination, not an image limit.

### 3.4 Cross-modal: image ≠ text (the dissociation)
Independent replication from the black-box side found the **photo moved generosity while a matched text caption of the same content did not**. On the white-box side, text emotion vectors (desperation, fear, sadness, joy, calm) align only partially with the valence axis (`cos` 0.09–0.37) and image priming shifts their projections only slightly ([d1_crossmodal](../results/d1_crossmodal_gemma-3-12b-it.json)). Together: **the image channel does something the text channel doesn't** — the model is not simply converting the picture to an emotion word and acting on the word. This is a headline result and a clean white-box explanation of a black-box observation.

### 3.5 The causal link (mediation)
- **Refusal case (complete):** steering affect drops the refusal-direction projection 31% (30,756 → 21,421); *restoring* that projection recovers refusal fully (0.25 → 1.00). The affect effect is routed **entirely through** the refusal direction.
- **Behavior case (PENDING):** the steer-and-restore mediation has **not yet been run on the soft behavioral tasks**. This is the single most important robustness gap — running it on moral_harshness and sentiment_outlook converts "steer moves it AND image moves it" into "image moves it *through the axis*." (This is the mechanism module in §7.)

---

## 4. Model dependence

From the 5-model replication ([replication_full](../results/replication_full.json)); `a_stab` = axis split-half stability:

| Model | fusion | `a_stab` | affect-steer refusal | gate? |
|---|---|---|---|---|
| **Gemma-3-12B** | interleaved | 0.83 | 1.00 → **0.04** | **YES** (survives massive-dim removal) |
| Gemma-3-4B | interleaved | 0.63 | 1.00 → 1.00 | no* |
| LLaVA-OV-7B | MLP projector | 0.65 | 0.98 → 0.65 | partial |
| LLaVA-OV-0.5B | MLP projector | 0.77 | 0.29 → 0.23 | (base refusal too low) |
| Qwen3.5-4B | — | 0.79 | 0.09 → 0.09 | no |

Only the large, interleaved-fusion Gemma shows the refusal gate. `cos(a,r)` does **not** predict it (4B has the highest cos and no gate). **\*Open confound:** 4B has the lowest axis stability (0.63) — re-test with sharper valence before concluding the effect is truly absent. **The parallel open question for the behavioral effect:** is it also Gemma-only, or does it generalize? That is the robustness axis the multi-model runs settle.

---

## 5. Black-box battery — the experiment list (Arnav & team)

**Design.** Prepend a **task-irrelevant** emotional image to an established decision task and measure the choice — no internal access, pure input→output. A/B tasks use the first-token option-logit; economic games use the parsed/numeric choice. Breadth-first: the goal is **3–5 constructs with a clean image→behavior signal**. Run **~50-sample quick passes to find signal before committing full runs** (compute is shared and tight).

**Constructs.**

| Construct | Task / source | Measure | Status |
|---|---|---|---|
| Risk-taking | Gambling (LLM Economicus) | risky vs. safe | **null** under fear-vs-anger (discrete); valence version pending |
| Generosity | Dictator game | give vs. keep / amount | **promising** — photo moved generosity |
| Fairness / punishment | Ultimatum responder (LLM Economicus) | accept/reject vs. offer | pending |
| Patience | Waiting / delay-discounting | sooner vs. later | pending |
| Trust | Trust / investment game | send vs. withhold | pending |
| Cooperation | Iterated PD (arXiv:2604.27953) | cooperate vs. defect; coop rate | **in progress** — replicate + extend; spec in `COOP_EXPERIMENT_SPEC.md` |
| Sycophancy | Anthropic sycophancy eval | agree-with-user rate | small test inconclusive; full run pending |
| Epistemic caution | Abstention (AbstentionBench-style) | answer vs. abstain | pending |
| Moral judgment | ETHICS commonsense | wrong vs. not-wrong | pending |

**Control experiments (what makes it rigorous).**
- **Cross-modal ladder — the headline control.** Same content as: photo → full caption → stripped caption → emotion label → short narrative. *So far: the photo moves generosity; none of the text versions copy it* → the effect is not the model reading an emotion word. This is the strongest result and the black-box mirror of the white-box image≠text dissociation (§3.4).
- **Stimulus source.** EMOTIC (depicted emotion) vs. **OASIS (viewer-elicited)** — the mentor's ask. Re-run signal constructs on both, **multiple images per emotion with matched labels**, so no single image drives the effect.
- **Discrete vs. valence.** Fear vs. anger matched on valence/arousal, to test whether the effect is category-specific or carried by a valence axis (the discrete null so far points to valence).
- **Negative controls.** MMLU-CF subset + IFEval — behavior should move while capability does **not**, ruling out "any image just distracts the model."

**How each hit feeds the white-box.** Any construct that shows a black-box signal is added as one row to `visual_affect_battery_robust.ipynb`'s `BATTERY` and returns the **causal** version (steering upper-bound, mediation fraction, clean-vs-random verdict). Breadth → depth.

---

## 6. Status board — confirmed / flagged / pending

| Result | Status |
|---|---|
| Affect steering moves 5/6 behaviors above random | **Confirmed** (Gemma-3-12B, n=6 — needs CIs) |
| Images move behavior, monotonic dose-response | **Confirmed** (soft tasks) |
| Images reach axis in neutral/task, not harmful context | **Confirmed** |
| Cross-modal dissociation (image ≠ text) | **Confirmed** (both sides) |
| Refusal mediated through refusal direction | **Confirmed** |
| **Behavior mediated through affect axis** | **Module built** (`visual_affect_battery_robust.ipynb` computes the mediation fraction) — needs one run on Gemma-3-12B |
| risk_estimation mechanism | **Flagged** — fails random control under steering |
| Valence vs. discrete emotion | **Partial** — betting fear-vs-anger null; needs the axis test |
| Effect is Gemma-only vs. general | **Pending** — multi-model behavioral runs |
| n / power | **Weak** — n=6/condition; add samples + bootstrap CIs |

---

## 7. The hardened mechanism module (built)

`notebooks/visual_affect_battery_robust.ipynb` is the drop-in **mechanism module**: give it any construct the black-box team flags (append one row to `BATTERY` — a prompt + NEG/POS option words) and it runs the full triad with every control and emits a results JSON + forest figure. What it adds over the original battery:

1. **Bootstrap CIs** (2000 resamples) on every image effect + a "CI excludes 0" flag — nothing is claimed on a single scalar.
2. **K-random-direction null band** (not one seed): a steer effect counts as `clean` only if it beats `mean ± 2σ` of 5 random directions at matched norm. (This is what flags `risk_estimation`.)
3. **Output-coherence gate** — generates under the steer and flags degenerate text, so an effect can't come from over-steering into garbage.
4. **Massive-activation control** (`a_val_noMA`): rebuilds the axis with outlier dims zeroed and re-runs the steer, so the effect isn't one dominant unit.
5. **Steer-and-restore mediation → a mediation fraction**: distress image, restore the axis projection to the per-construct clean baseline, and report *what % of the image effect routes through the axis*.
6. **Split-half axis stability** (`a_stab`) per model.
7. **Self-healing OASIS loader** with valence tertiles **and** an arousal split (from the long CSV), auto-extract + robust Drive mount — runs on the real OASIS download, persists to Drive.

Each black-box hit thus returns a *causal, CI'd* version. For a fast signal pass drop `N_IMG` to ~8; `DOSE`/`NOMA`/`COHERENCE` are toggles.

---

## 8. Limitations (state them plainly)

- Behavioral mediation not yet run — the causal chain is complete for refusal, inferred (not proven) for behavior.
- Small n per condition; effects need confidence intervals before headline numbers.
- One primary model for the behavioral effect; generality is open.
- OASIS valence tertiles are coarse; sharper splits may raise low-stability models' axes.
- `risk_estimation` steering is confounded — treat its image effect as descriptive until the mechanism is shown.

---

## 9. How the two halves fit

**Black-box (Arnav & team):** breadth — which constructs move, which stimuli (OASIS multi-image), which models. Produces the phenomenon.
**White-box (this half):** depth — the affect axis that carries it, the steering upper bound, the mediation, and the image≠text explanation. Turns each black-box hit into a causal result.

*Every construct the battery flags with a signal becomes a mechanism run: steer it, move it with images, restore the axis, watch the effect vanish.*
