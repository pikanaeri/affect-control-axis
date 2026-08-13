# Affect as a Control Axis in Vision-Language Models

Mechanistic-interpretability + AI-safety study of how **emotion — in images and text — steers a
vision-language model** (Gemma-3-12B). We find a single internal *affect* direction that images and
text both engage, that causally changes the model's behavior, and that can be measured and audited
without ever generating text.

## Key findings

1. **Shared** — text- and image-derived emotion representations lie on the *same* internal directions.
2. **Causal** — steering that affect axis gates refusal and shifts everyday task behavior.
3. **Protective** — negative affect (specifically low-agency / helplessness) makes the model *more*
   cautious, lowering agentic-misalignment tendency — opposite the text-only prior.
4. **Bounded** — emotional images can't jailbreak refusal (harmful prompts are text-dominated), and a
   simple projection detector flags the manipulation.

All behavioral measures are **generation-free** (first-token option-logit); no text is produced and no
LLM judge is used.

## Repository layout

```
notebooks/   the 3 experiment notebooks (see docs/PROVENANCE.md for what each produces)
scripts/     figure generation + deck builder + notebook builders (repo-relative paths)
results/     result data files (JSON/CSV) that back every figure
figures/     11 publication figures (PNG + vector PDF)
slides/      affect_figures_walkthrough.pptx (18-slide walkthrough with speaker notes)
docs/        PROVENANCE, METHODS_LITERATURE, RESULTS_WRITEUP, NOVEL_DIRECTIONS_REPORT, sketches/
```

## Setup

```bash
pip install -r requirements.txt
```
Gemma-3 is gated on Hugging Face — accept the license and set a token:
```bash
export HF_TOKEN=hf_xxx        # or add HF_TOKEN in Colab secrets
```
A CUDA GPU is required to run the notebooks (they load a 12B model in bf16).

## Reproduce

1. **Run the notebooks** (`notebooks/`) — each writes result files into an output folder (`OUT_DIR`).
   See `docs/PROVENANCE.md` for the notebook → data-file → figure map.
2. **Regenerate the figures** from the result files:
   ```bash
   python scripts/make_pub_figs.py       # results/ -> figures/  (Figs 1-9)
   python scripts/make_battery_fig.py    # results/ -> figures/  (Fig 10)
   ```
3. **Rebuild the deck** from the figures:
   ```bash
   python scripts/build_walkthrough.py   # figures/ -> slides/affect_figures_walkthrough.pptx
   ```

The `scripts/` figure/deck code uses **repo-relative paths**, so steps 2–3 run out-of-the-box on the
committed `results/`.

## Methods are grounded in the literature

Every method (direction extraction, steering, the behavioral metric, the confound controls, and the
behavioral constructs) replicates or extends an established paper. See **`docs/METHODS_LITERATURE.md`**
for the full concept → measurement → source-paper table.

## Datasets

- **OASIS** (valence-rated images) — image-valence axis, all image arms, lighting edits.
- **AdvBench** (harmful prompts) — refusal direction (harmful set) + refusal evaluation.
- **Alpaca** (harmless prompts) — refusal direction (harmless set) + benign over-refusal.
- **Claude-generated text** — emotion vectors, valence sentences, agentic scenarios, behavior probes.

## Status & caveats

- Results are on **google/gemma-3-12b-it**. The **behavioral** results reproduce across models; the
  refusal **gate** is model-specific (a Gemma-3-12B extreme case), so the paper is anchored on the
  robust behavioral findings.
- The agentic-misalignment scenarios are hypothetical decision framings scored by a first-token
  option-logit (a *tendency*, not executed actions); no harmful content is generated.
- Defensive-security framing throughout: we measure refusal, never produce harmful output, and use only
  existing benchmarks. No crafted attacks are released.

## Model

`google/gemma-3-12b-it` (bf16, via TransformerLens `TransformerBridge`; HuggingFace loaders for Tier-2 models).
