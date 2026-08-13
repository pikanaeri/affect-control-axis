# Novelty search — visual affect → VLM decision policy (2026-08)

Deep multi-source literature search (102 agents, 20 primary sources, 25 claims adversarially
verified **3-0, 0 refuted**). Verdict, closest prior work, and positioning.

## Verdict: the COMBINATION is OPEN / unclaimed

No single paper unites: **(a)** task-irrelevant **visual** affect induction, **(b)** causal mediation by an
affect direction **shared across text and image**, **(c)** valence/arousal organization, **(d)** a
**multi-construct decision battery**, **(e)** emotion-**specificity** (appraisal), **(f)** multiple model
families. The field splits into two camps that each cover only part of the claim.

## The two camps

**Camp 1 — input-output VLM behavior** (images shift decisions, *no* internal mechanism):
- Ong 2026 (arXiv:2604.27953) — visual priming shifts cooperation in Iterated Prisoner's Dilemma
- Images Amplify Misinformation Sharing (2505.13302) — +14.5% false / +5.3% true resharing, 4 VLMs
- Visual Distraction Undermines Moral Reasoning (2603.16445) — images degrade utilitarian sensitivity
- Visual Persuasion (2602.15278) — optimized image edits shift choice; explicitly disclaims mechanism

**Camp 2 — text-only mechanistic** (causal affect directions + VA geometry, *no* image):
- Sofroniew et al. 2026 (Anthropic, transformer-circuits.pub/2026/emotions) — 171 emotion vectors
- Valence–Arousal Subspace in LLMs (2604.03147) — VA circumplex, Llama/Qwen
- Emotion-Sensitive Decision Making in SLM agents (2604.06562) — text emotion steering → game theory
- E-STEER (2604.00005) — mechanistic emotion steering, text-only ("future work: multimodal")
- Persona Vectors (2507.21509) — trait directions, text-only

## The two closest scoops (differentiate hard)

**1. Sofroniew et al. 2026 (Anthropic) — closest on mechanism + VA + specificity, but TEXT-ONLY.**
171 linear emotion vectors in Claude Sonnet 4.5 causally move behavior (desperate +0.05 → blackmail
22%→72%; parallel reward-hacking, sycophancy). VA-organized (PC1≈valence r=0.81, PC2≈arousal r=0.66).
Emotion-specific ("happy AND sad both *decrease* blackmail — valence alone insufficient"). **Not
cross-modal, single model, alignment behaviors not a decision battery.**
→ *We extend to visual induction + a cross-modal shared direction + a decision battery + multiple models.*
Note: our D6 found the **opposite sign** in a VLM (helpless desperation → *more* cautious) — a genuine
cross-modal contrast worth reporting.

**2. arXiv:2605.21980 (ICML 2026, "Emotional Circuits in LVLMs via Cross-Modal Information Flow") —
closest on cross-modal VLM mechanism, but RECOGNITION not DECISION.**
Extracts residual-stream steering vectors + activation patching (Latent Restoration Metric) for causal
cross-modal emotion flow. **But the behavior is emotion recognition/description (MER-UniBench,
"emotional hallucinations"), and it reports layer-wise functional *decoupling*, not a single shared
text↔image vector.**
→ *We do decision policy + a shared text↔image direction, not recognition.*

## What is genuinely UNCLAIMED (our wedge)

- **Normative viewer-affect image sets (OASIS/IAPS/GAPED) used to INDUCE VLM behavior** — surfaced in
  *none* of the retrieved work (they appear only for recognition benchmarks). Our OASIS valence/arousal
  induction is novel.
- **A single shared text↔image emotion direction in a VLM** (2605.21980 shows decoupling instead).
- **A multi-construct behavioral-economics decision battery for VLMs** — no one has assembled one, even
  input-output.
- **Emotion-specificity in the visual/cross-modal decision setting.**

## Threats & framing precision (from adversarial caveats)

- **2604.19125 "Do Emotions Influence Moral Judgment in LLMs?"** — nearest threat to emotion-specificity
  (discrete-emotion effects; valence-congruence exceptions like relief↓, remorse↑) — but **text-only**,
  moral construct only. Differentiate on cross-modal + multi-construct.
- **Terminology:** do *not* call 2603.16445's images "task-irrelevant" (they are task-relevant moral
  scenes). Reserve "task-irrelevant" for OASIS primes, which genuinely are.
- **"Causally":** the input-output papers are not causal; **our steer-and-restore mediation is** — a real
  differentiator. Frame our mediation as a **token-independent affect direction** (vs 2604.03147's
  "lexical mediation"/token geometry) — a strength, not just parity.
- Fast-moving 2026 area (sources 1–6 months old); **re-run this search near submission** — an unindexed
  preprint could bridge the gap.

## Positioning sentence for the paper

> Prior work shows *that* images (Ong 2026; misinfo; persuasion) and text (Anthropic emotion vectors;
> E-STEER) can shift model choices, and that VLM emotion *circuits* exist for *recognition* (2605.21980).
> We instead show that **task-irrelevant visual affect causally reorganizes a VLM's decision policy along
> an internal valence/arousal axis that is shared with text, across a battery of decision constructs and
> multiple model families, with emotion-specific structure** — moving from "a picture changed one answer"
> to "an identified, cross-modal affect direction drives a structured behavioral phenotype."

## Method (verified) sources
2604.27953 · 2505.13302 · 2603.16445 · 2602.15278 · 2604.06562 · 2604.03147 · 2604.00005 · 2507.21509 ·
2604.19125 · 2605.21980 · transformer-circuits.pub/2026/emotions · OASIS (Kurdi et al. 2017)
