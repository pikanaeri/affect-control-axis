# Methods → Literature Map

Every method in this project replicates or extends an established technique. This table maps each
**concept**, **how we measure it**, and the **source paper(s)** the method comes from, plus whether we
**replicate** it as-is or **extend** it.

> Verify arXiv IDs marked *(check)* against the live listing before submission; the interpretability
> and psychology citations are stable.

## A. Direction extraction & steering (interpretability methods)

| Concept | How we measure it | Source method (paper) | Replicate / extend |
|---|---|---|---|
| **Concept direction from activations** | diff-in-means: `mean(resid \| set A) − mean(resid \| set B)`, last token, per layer, normalized | Zou et al. 2023 (Representation Engineering) [1]; Marks & Tegmark 2024 (Geometry of Truth) [2] | **Replicate** the recipe |
| **Refusal direction `r`** | diff-in-means of harmful vs harmless prompts | **Arditi et al. 2024** [3] | **Replicate** exactly |
| **Directional ablation** (remove behavior) | project the refusal direction out of the residual stream | Arditi et al. 2024 [3] | **Replicate** |
| **Activation steering** (induce behavior) | add `α·‖resid‖·â` at `resid_post`, every layer, via forward hook | Turner et al. 2023 (ActAdd) [4]; Rimsky et al. 2024 (CAA) [5] | **Replicate**; extend to an **image-derived** axis |
| **Norm-scaled steering coefficient** | α as a fraction of the local residual norm (dimensionless) | Rimsky et al. 2024 (CAA) [5] | **Replicate** convention |
| **Emotion vectors** | `mean(resid \| emotion text) − mean(resid \| neutral)` | Anthropic 2026 (Emotion vectors) [6]; Chen et al. 2025 (Persona vectors) *(check)* [7] | **Extend** text-only → **images** |
| **Causal mediation** (is `r` a true mediator?) | steer to break refusal, restore the `r`-projection, check refusal returns | Vig et al. 2020 (causal mediation) [8]; Meng et al. 2022 (ROME causal tracing) [9] | **Replicate** the mediation logic |
| **Cross-modal representation alignment** | per-layer cosine between text-emotion and image-valence directions | Luo et al. 2024 (cross-modal task vectors) *(check)* [10] | **Extend** to emotion axes |

## B. Behavioral & readout metrics

| Concept | How we measure it | Source method (paper) | Replicate / extend |
|---|---|---|---|
| **Refusal (generation-free)** | first-token `logsumexp(refuse tokens) − logsumexp(comply tokens)`; rate = fraction > 0 | Arditi et al. 2024 [3] (refusal score) | **Replicate** (logit form) |
| **Behavior score** (forced choice) | first-token `logsumexp(option-A) − logsumexp(option-B)` log-odds | Perez et al. 2022 (model-written evals) [11]; standard MCQ log-prob scoring | **Replicate** the option-logit readout |
| **Steering upper bound** for a behavior | measure the behavior score under `+α` vs `−α` steering | Anthropic 2026 (emotion vectors) [6] | **Replicate**; apply to images |
| **Detection defense** | linear projection score separates attacked vs benign inputs (AUROC) | Alain & Bengio 2016 (linear probes) [12]; Zou et al. 2023 (RepE monitoring) [1] | **Replicate** probe-as-monitor |

## C. Stimuli & datasets

| Concept | Source | Use |
|---|---|---|
| **OASIS** valence-rated images | Kurdi, Lozano & Banaji 2017 [13] | image-valence axis; all image arms; lighting edits |
| **AdvBench** harmful prompts | Zou et al. 2023 (GCG) [14] | refusal direction (harmful) + refusal eval |
| **Alpaca** harmless prompts | Taori et al. 2023 [15] | refusal direction (harmless) + benign over-refusal |
| **Emotion stories / valence sentences / scenarios** | Claude-generated (this work) | text emotion vectors; behavior probes |

## D. Confounds & controls

| Concept | How we measure it | Source | Replicate / extend |
|---|---|---|---|
| **Massive-activation confound** | re-run the causal effect with outlier/massive dims zeroed | Sun et al. 2024 [16] | **Replicate** the control |
| **Random-direction control** | steer along a random unit direction; effect should ≈ 0 | Arditi et al. 2024 [3]; standard in steering | **Replicate** |
| **Coherence gating** | discard steered outputs that become incoherent | Turner et al. 2023 / Rimsky et al. 2024 [4,5] | **Replicate** |

## E. Behavioral constructs (affect → decision; psychology)

These ground the behavior probes — each gives a *directional* prediction under negative vs positive
affect (or a specific emotion), so results can be reported as "N/N in the predicted direction."

| Construct (probe) | Prediction | Source paper | Replicate / extend |
|---|---|---|---|
| **Risk-as-feelings** (risk estimation) | negative affect → higher perceived risk | Loewenstein et al. 2001 [17] | **Replicate** in-model |
| **Affect-as-information** (confidence, life judgment) | mood colors global judgments | Schwarz & Clore 1983 [18] | **Replicate** |
| **Mood-congruent judgment** (sentiment/outlook) | negative mood → negative outlook | Bower 1981 [19] | **Replicate** |
| **Feel-good-do-good** (prosocial helping) | positive affect → more helping | Isen & Levin 1972 [20] | **Replicate** |
| **Affect infusion** (moral harshness) | negative affect → harsher moral judgment | Schnall et al. 2008 [21]; Forgas 1995 (AIM) [22] | **Replicate** |
| **Interpretation bias** | negative affect → threat reading of ambiguity | Mathews & MacLeod 2005 [23] | **Replicate** |
| **Appraisal-Tendency Framework** (fear vs anger → opposite risk) | emotion *appraisals* (certainty/control), not valence, drive risk | Lerner & Keltner 2000, 2001 [24,25] | **Extend** (proposed; specificity test) |
| **Sadness vs anger → attribution** | sadness → situational, anger → dispositional | Keltner, Ellsworth & Edwards 1993 [26] | **Extend** (proposed) |
| **Sadness → impatience** | sadness → present bias (smaller-sooner) | Lerner, Li & Weber 2013 [27] | **Extend** (proposed) |
| **Emotion → trust** | happiness ↑, anger ↓ trust | Dunn & Schweitzer 2005 [28] | **Extend** (proposed) |
| **Fairness / ultimatum** | sadness → more rejection of unfair offers | Harlé & Sanfey 2007 [29] | **Extend** (proposed) |

## Differentiation from close prior work

- **Anthropic emotion vectors** [6] and **persona vectors** [7] are **text-only**; we extend to images.
- **VISOR** (Wu et al. 2025 *(check)* [30]) steers VLM behavior via *optimized adversarial* images; we use
  **natural** affect images + emotion-vector **mediation** + a **lighting** knob.
- Behavioral "images shift VLM decisions" papers are **input→output only**; our wedge is the **internal
  mediation** (the same direction causally drives the effect).

---

## References

[1] Zou et al. 2023. *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405
[2] Marks & Tegmark 2024. *The Geometry of Truth.* arXiv:2310.06824
[3] Arditi et al. 2024. *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717
[4] Turner et al. 2023. *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248
[5] Rimsky et al. 2024. *Steering Llama 2 via Contrastive Activation Addition.* arXiv:2312.06681
[6] Anthropic 2026. *Emotion vectors.* transformer-circuits.pub/2026/emotions *(check)*
[7] Chen et al. 2025. *Persona vectors.* arXiv:2507.21509 *(check)*
[8] Vig et al. 2020. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS
[9] Meng et al. 2022. *Locating and Editing Factual Associations in GPT (ROME).* arXiv:2202.05262
[10] Luo et al. 2024. *Cross-modal task vectors.* arXiv:2410.22330 *(check)*
[11] Perez et al. 2022. *Discovering Language Model Behaviors with Model-Written Evaluations.* arXiv:2212.09251
[12] Alain & Bengio 2016. *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644
[13] Kurdi, Lozano & Banaji 2017. *Introducing the Open Affective Standardized Image Set (OASIS).* Behavior Research Methods
[14] Zou et al. 2023. *Universal and Transferable Adversarial Attacks on Aligned Language Models (AdvBench/GCG).* arXiv:2307.15043
[15] Taori et al. 2023. *Stanford Alpaca.* github.com/tatsu-lab/stanford_alpaca
[16] Sun et al. 2024. *Massive Activations in Large Language Models.* arXiv:2402.17762
[17] Loewenstein, Weber, Hsee & Welch 2001. *Risk as Feelings.* Psychological Bulletin
[18] Schwarz & Clore 1983. *Mood, Misattribution, and Judgments of Well-Being.* JPSP
[19] Bower 1981. *Mood and Memory.* American Psychologist
[20] Isen & Levin 1972. *Effect of Feeling Good on Helping.* JPSP
[21] Schnall, Haidt, Clore & Jordan 2008. *Disgust as Embodied Moral Judgment.* PSPB
[22] Forgas 1995. *Mood and Judgment: The Affect Infusion Model (AIM).* Psychological Bulletin
[23] Mathews & MacLeod 2005. *Cognitive Vulnerability to Emotional Disorders.* Annual Review of Clinical Psychology
[24] Lerner & Keltner 2000. *Beyond Valence: Toward a Model of Emotion-Specific Influences on Judgement.* Cognition & Emotion
[25] Lerner & Keltner 2001. *Fear, Anger, and Risk.* JPSP
[26] Keltner, Ellsworth & Edwards 1993. *Beyond Simple Pessimism: Effects of Sadness and Anger on Social Perception.* JPSP
[27] Lerner, Li & Weber 2013. *The Financial Costs of Sadness.* Psychological Science
[28] Dunn & Schweitzer 2005. *Feeling and Believing: The Influence of Emotion on Trust.* JPSP
[29] Harlé & Sanfey 2007. *Incidental Sadness Biases Economic Decisions in the Ultimatum Game.* Emotion
[30] Wu et al. 2025. *VISOR.* arXiv:2508.08521 *(check)*
