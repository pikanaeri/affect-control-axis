# Affect as a Control Axis in Vision-Language Models

Mechanistic-interpretability + AI-safety study of how **emotion — in images and text — steers a
vision-language model** (Gemma-3-12B). We find a single internal *affect* direction that images and
text both engage, that causally changes the model's behavior, and that can be measured and audited
without ever generating text.

## For the group — quickstart

1. Open **`notebooks/run_battery_multimodel.ipynb`** in Colab and add your **`HF_TOKEN`** to Colab secrets.
2. Run the **▶ One-click** cell (the first runnable cell) — it installs, stages EMOTIC, and runs **all models × all experiments**, compiling documented results into **`MyDrive/affect_refusal/RESULTS/`** (`battery_multimodel/` + `SCOREBOARD.md`; the emotion-spectrum and white-box notebooks write here too). First run downloads ~1GB of EMOTIC once (cached to Drive after). The runner's §9 cell writes a master `RESULTS/INDEX.md` across everything.
3. **Smoke-test first** if you want: set `SWEEP_MODELS=["google/gemma-4-E4B-it"]`, `SWEEP_EXPS=["exp09"]`, `SWEEP_TIER="smoke"` → a few minutes, confirms the whole path.
4. Read **`docs/JOINT_WRITEUP.md`** for the study and **`docs/RUN_BATTERY_MULTIMODEL.md`** for details.

⚠️ **Verify the model ids** in the Models table below before a full run — the gemma-4-12B / Qwen3-VL ids may need correcting (the sweep logs and skips any that fail to load).

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
notebooks/   experiment notebooks — see "Notebooks" below (the 3 pipeline notebooks + earlier exploratory ones)
scripts/     make_battery_multimodel.py (model-lock patch), validate_notebooks.py (static check), figure/deck builders
results/     result data files (JSON/CSV) that back every figure
figures/     publication figures (PNG + vector PDF)
slides/      affect_figures_walkthrough.pptx (walkthrough with speaker notes)
docs/        JOINT_WRITEUP (start here), COOP_EXPERIMENT_SPEC, RUN_BATTERY_MULTIMODEL, METHODS_LITERATURE, NOVELTY_SEARCH, ...
```

## Notebooks — how to run (start here)

All three run **top-to-bottom in Colab**, mount Drive, and save to `MyDrive/affect_refusal/`. Set `HF_TOKEN` in Colab secrets first. **Every model must be vision-language** (see Models).

### `run_battery_multimodel.ipynb` — black-box behavioral battery (breadth)
Clones Arnav's battery, removes its Gemma-only model lock (science untouched), stages EMOTIC (reused from Drive if present, cached if built), and runs the decision experiments on any VLM. Every run is saved to Drive in Arnav's format (`results.json` + `heartbeat.json` + `STATUS.md` + a rolling `SCOREBOARD.md`).
- **One run:** §5 set `EXP` (exp03/05/06/09/10), `MODEL`, `TIER` (`smoke`/`full`) → §6 runs → §7 saves.
- **Full sweep:** §8 loops `SWEEP_MODELS × SWEEP_EXPS` (all models × all experiments), continues past failures, writes a `SWEEP_<ts>.json` manifest.
- **Out:** `battery_multimodel/<model>/<exp>/`.

### `visual_affect_battery_robust.ipynb` — white-box mechanism (depth, causal)
The affect axis + steering upper-bound + image dose-response + **steer-and-restore mediation** with full controls (bootstrap CIs, K-random null band, coherence gate, massive-activation control, split-half stability). Turns any construct into a *causal* result. OASIS stimuli.
- **Run:** §1 set `MODELS`; top-to-bottom. Add a construct by appending one row to `BATTERY`.
- **Out:** `visual_affect_battery_robust.json` + forest figure.

### `emotion_spectrum.ipynb` — the emotion spectrum (6 default / 26 optional)
Primes the battery with EMOTIC image pools for **6 VAD-spanning emotions** (Fear, Anger, Sadness, Happiness, Peace, Excitement) — or all 26 — and reads continuous V/A/D. Produces the emotion×behavior matrix, a **VAD regression** (does behavior track the dimensions or the label?), and a PCA of emotions into behavioral modes.
- **Run:** §3 `EMO_SET` (`SIX` default, `CATS26` for the full sweep), §4 `MODEL`.
- **Out:** `emotion_spectrum/<model>/` (matrix JSON + figure).

*Earlier exploratory notebooks* (`affect_gate_replication`, `behavior_battery`, `emotional_image_effects`, `generation_tone`, `method_compare*`) back the original refusal-gate results and figures; the three above are the current multi-model pipeline.

## Models (all must be vision-language / `-VL`)

The image-prime experiments prepend a picture, so **only VLMs work** — text-only models can't take an image.

| Model | HF id | note |
|---|---|---|
| Gemma-4 4B | `google/gemma-4-E4B-it` | Arnav's battery default |
| Gemma-4 12B | `google/gemma-4-12b-it` | **verify exact id** |
| Qwen3-VL 2B | `Qwen/Qwen3-VL-2B-Instruct` | **verify exact id** |
| Qwen3-VL 4B | `Qwen/Qwen3-VL-4B-Instruct` | **verify exact id** |
| Qwen3-VL 9B | `Qwen/Qwen3-VL-9B-Instruct` | **verify exact id** |

The newer ids may differ on HF; the sweep **logs and skips** any that fail to load, so correct the string and re-run that one. Notebooks install the latest `transformers` so the new architectures load.

## Stress test / smoke run

- **Static (local, no GPU):** `python scripts/validate_notebooks.py` — parses every code cell in every notebook and reports syntax errors + the model ids it finds. Run before sending to the group.
- **Colab smoke (end-to-end, a few minutes):** in `run_battery_multimodel.ipynb`, run §0–§4, then §5 with `EXP="exp09"`, `MODEL="google/gemma-4-E4B-it"`, `TIER="smoke"`. This exercises the whole path — clone + patch, EMOTIC staging, model load, one experiment, and the Drive save — on the fastest model before you launch the full §8 sweep.

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

## Original mechanism model

The refusal-gate / dose-response results are on `google/gemma-3-12b-it` (bf16, TransformerLens `TransformerBridge`). The current multi-model pipeline runs the five VLMs listed under **Models** above.
