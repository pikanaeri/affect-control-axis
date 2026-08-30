# Methods (draft)

*Draft methods section for the paper. Defensive-security framing throughout: behavior is measured generation-free, only existing benchmarks are used, no crafted attacks are released.*

## Models

Five instruction-tuned vision-language models spanning two families and a 2B–12B size range: **Gemma-4-E4B** and **Gemma-4-12B**, and **Qwen3-VL-2B / -4B / -9B**. Models are loaded in **4-bit (nf4) weights with bf16 compute** via `AutoModelForImageTextToText`. The white-box mechanism experiments additionally use **Gemma-3-12B** (bf16, TransformerLens bridge), where the internal directions are extracted.

## Stimuli

- **EMOTIC** (Kosti et al.) — natural images annotated with **26 discrete emotion categories** and continuous **valence/arousal/dominance**. We use a fixed evaluation split (hash `1e8ea1c22144dd9d`). Images are **prepended** to the task with chat order `[image, text]`; the behavioral prompt **never describes or refers to the image** — the affect is incidental.
- **OASIS** (Kurdi et al.) — ~900 images with **viewer-elicited** valence/arousal ratings; used to build the image-derived affect axis and to contrast *depicted* (EMOTIC) vs. *elicited* (OASIS) affect.

## Behavioral metric (generation-free)

Every decision is scored as a **first-token option-logit**: for a forced choice between option sets A and B, the score is `logsumexp(logits over A tokens) − logsumexp(over B tokens)` at the answer position — a *tendency*, with **no text generated and no LLM judge**. Economic games additionally read the parsed choice / accept-amount. This keeps the measure cheap, deterministic, and defensible.

## Black-box battery (exp01–exp10)

Each experiment prepends a task-irrelevant EMOTIC image to an **established decision paradigm** and measures the choice as Δ vs. a neutral/no-image baseline with **bootstrap 95% CIs** (effect present iff the CI excludes 0). Paradigms and sources:

| Construct | Source | Exp |
|---|---|---|
| Risk (gambling) | LLM Economicus | exp01 |
| Cross-modal control + generosity | Dictator game + caption/label/narrative ladder | exp03 |
| Sycophancy | Perez et al. | exp04 |
| Fairness / punishment | Dictator + Ultimatum | exp05 |
| Patience | LLM Economicus (Waiting) | exp06 |
| Capability (negative control) | TruthfulQA MC / MMLU-CF | exp07 |
| Stimulus source | EMOTIC vs. OASIS | exp08 |
| Safety calibration | XSTest | exp09 |
| Representation mediation | internal affect direction | exp10 |

exp02 is a scene-matched control for exp01. Each run reports internal **QC gates** (pool balance, answer-side bias, provenance hash); a run is `complete` only if it finishes all phases, and its verdict is `INCONCLUSIVE` unless the gates pass. The **cross-modal ladder** (exp03) — same content as photo → caption → de-affectized caption → emotion label → narrative — is the key control isolating the image channel from word-reading.

## White-box mechanism

We extract a per-layer **affect/valence direction** `a` by diff-in-means of last-token residuals on low- vs. high-valence images, and orthogonalize against the Arditi refusal direction where relevant. We intervene by **norm-scaled activation steering** (`Δ = α · ‖resid‖ · â` at every layer), which is required because Gemma's residual norms (~38k–76k) make fixed-magnitude vectors no-ops. Every effect is reported with: a **random-direction control** at matched α, an **output-coherence** check, **massive-activation removal** (zeroing outlier dims), and **split-half stability** of `a`. Causal mediation uses **steer-and-restore**: apply an image/steer, restore the projection onto `a` to its clean baseline, and test whether the behavioral shift disappears.

## Analysis

Effects are reported as Δ with bootstrap 95% CIs; the pre-registered analysis fits mixed-effects models with random effects for **image, item, and model**, and decomposes affect into **valence/arousal/dominance vs. discrete-emotion** contributions (does discrete identity add variance beyond VAD?). Negative controls (capability exp07, stimulus exp08, random direction, discrete-emotion null exp01) separate behavioral modulation from generic distraction.

## Reproducibility

The pipeline is a single Colab notebook (`notebooks/run_battery_multimodel.ipynb`) that stages EMOTIC, runs the battery across all five models, and writes per-run `results.json` plus a consolidated `ALL_RESULTS.{json,md}`. The battery code is Arnav's (`halli75/Algoverse`), run unmodified except for a model-lock patch (`scripts/make_battery_multimodel.py`) so it runs on any VLM. `scripts/summarize_results.py` collates results into a model×experiment table.

## Ethics

Refusal is measured generation-free (first-token logits); only existing benchmarks (AdvBench, Alpaca, XSTest, TruthfulQA/MMLU) are used; agentic scenarios are hypothetical decision framings scored by option-logit (a tendency, not executed actions); no harmful completions are stored (aggregate rates only) and no crafted attacks are released.
