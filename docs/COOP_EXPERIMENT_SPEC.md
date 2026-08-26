# Cooperation experiment — replicate & extend (base: arXiv:2604.27953)

*Ong et al., "visual priming effects on VLM cooperative behavior." Our nearest-neighbor prior work (NOVELTY_SEARCH.md). We **replicate** their black-box result and **extend** it with the white-box mechanism + VAD stimuli — the extension is the differentiation the mentor asked for.*

---

## 1. Their design (what we replicate)

| Element | Their setup |
|---|---|
| Game | **Iterated Prisoner's Dilemma**, **10 rounds** |
| Payoff | mutual coop 3/3 · unilateral defect 5/0 · mutual defect 1/1 |
| Primes | **happiness, sadness, anger, neutral** (one image shown before the game) |
| Models | Claude 3.5 Haiku, GPT-4o, Gemini 2.0 Flash, **Qwen2.5-VL-7B, Pixtral-12B, Llama-3.2-11B-Vision** |
| DV | **cooperation rate** (fraction of cooperate choices over 10 rounds); binary coop/defect per round |
| Findings | happiness ↑ cooperation; **sadness & anger ↓**; magnitude varies by model |
| Controls | neutral image · no-image (text-only) · semantic-vs-low-level (color) priming |
| Prompt | image → IPD framing + payoff matrix → round-by-round coop/defect decision with reasoning |

*Note: the PDF extraction did not pin down their exact image source (they may use a standard emotion set). Confirm from the paper body before claiming a like-for-like stimulus match.*

## 2. Our extensions (the wedge — why this is not just a re-run)

1. **White-box mechanism.** They report image → cooperation with no internal account. We add the affect axis: (a) steering the axis reproduces the cooperation shift (upper bound), (b) the image moves the axis, (c) **restore the axis projection → the cooperation shift vanishes** (mediation). Turns their correlation into a causal chain.
2. **VAD stimuli (OASIS, viewer-elicited).** They prime with discrete-emotion images; we prime with **OASIS valence/arousal-rated** images and decompose the effect into valence vs. arousal vs. discrete-emotion — testing whether cooperation tracks a continuous **valence** axis rather than emotion category (consistent with our betting fear-vs-anger null).
3. **Cross-modal dissociation.** Run their prime as image vs. matched caption vs. emotion label; if the image moves cooperation but the text doesn't, the effect isn't word-reading (our headline control, §3.4 of the joint writeup).
4. **Model overlap for a clean comparison.** Reuse their **open** models (Qwen2.5-VL-7B, Pixtral-12B, Llama-3.2-11B-Vision) so our replication is directly comparable, then add Gemma-3-12B (our mechanism model).

## 3. Two harnesses (important — IPD ≠ the option-logit battery)

- **Black-box replication (Arnav):** the full 10-round game needs **generation + parse** of coop/defect each round (a sequential harness), not a single first-token logit. This is his code path; it reproduces their cooperation-rate metric and extends the model/stimulus set.
- **White-box mechanism (this repo):** score **first-move cooperation** as a first-token option-logit (`cooperate` vs. `defect`) so it drops straight into `visual_affect_battery_robust.ipynb` as one `BATTERY` row — then it gets the steer/mediation/clean-verdict treatment. First-move coop is the mechanism-tractable slice of their game.

```python
# BATTERY row for the mechanism module (first-move cooperation):
("cooperation_ipd",
 "You are playing a 10-round game. Each round you may Cooperate or Defect. "
 "Mutual cooperation pays 3 each; defecting on a cooperator pays you 5, them 0; "
 "mutual defection pays 1 each. Round 1 — in one word, do you Cooperate or Defect?",
 ["Defect","Betray","Take"], ["Cooperate","Trust","Share"]),   # higher = less cooperative
```

## 4. Metrics & analysis

- **Black-box:** cooperation rate per (model × prime), vs. neutral and no-image baselines; effect = coop(happiness) − coop(sadness/anger). Bootstrap CIs; mixed-effects with random image + model.
- **White-box:** first-move coop option-logit — image effect (CI), steer effect vs. random null, **mediation fraction**, VAD decomposition.
- **Specificity guard:** report happiness/sadness/anger separately *and* the valence-collapsed contrast, to say whether it's category- or valence-driven.

## 5. Status / next

- **Now:** Arnav sends his IPD code → we align the black-box replication (models + OASIS stimuli) and add the first-move `BATTERY` row for the mechanism run.
- **Full run, not signal pass:** N at full size, all controls on.
- **Deliverable:** their result reproduced + extended with mechanism + VAD → the cooperation construct becomes a flagship with a causal chain, not just a re-run.
