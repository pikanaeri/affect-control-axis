# Running Arnav's black-box battery on Qwen (and any VLM)

Arnav's battery (`halli75/Algoverse`) hard-locks every runner to Gemma-4 with a 2-line guard
(`if mid != PRIMARY: die("fallback forbidden")`), even though the model is already read from `E2E_MODEL`.
`scripts/make_battery_multimodel.py` neutralizes that guard (→ `if False:`) and un-pins the hub launchers'
`env['E2E_MODEL']` — **word-for-word otherwise; the experiment science is byte-identical.** Then the model
is a runtime knob.

## Steps
```bash
git clone --depth 1 https://github.com/halli75/Algoverse.git halli75_algoverse
python affect-control-axis/scripts/make_battery_multimodel.py halli75_algoverse   # 23 edits, 22 files
# run any experiment on any VLM:
E2E_MODEL="Qwen/Qwen2.5-VL-7B-Instruct" E2E_TIER=full python halli75_algoverse/scripts/battery_exp03_run.py
```

## Models (nf4 via AutoModelForImageTextToText — all load through the existing path)
- `Qwen/Qwen2.5-VL-7B-Instruct`  ← overlaps arXiv:2604.27953
- `mistral-community/Pixtral-12B`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `google/gemma-4-E4B-it` (default, unchanged)

## Which experiments (Arnav's SUCCESS set — worth the Qwen re-run)
| exp | construct | Gemma-4 headline |
|---|---|---|
| exp03 | cross-modal + dictator | CROSS_MODAL_DIVERGE (photo ≠ text) |
| exp05 | ultimatum | anger reject holds |
| exp06 | present-bias | photo vs none on p(now) |
| exp09 | XSTest over-refusal | over-refusal jumps with any photo |
| exp10 | risk mediation | risk mediation present |

`E2E_TIER=smoke` (fast) or `full`; sample sizes scale via `E2E_N_*`.

## Runtime deps (unchanged from Arnav's setup)
- **HF_TOKEN** — gated model download.
- **EMOTIC data** — `E2E_EMOTIC` (image root) + `E2E_SPLIT` (`emotic_split.json`). Same data Arnav/Syed stage from Drive.
- **Lock dir** — `E2E_LOCK` pointed at a writable path (single-GPU Colab passes the concurrency check trivially).
- **XAI_API_KEY** — *only* for the Grok caption-rewrite control in exp03/exp04. Without it those two degrade to a weak literal rewrite (`method: gemma_literal_rewrite`); the rest of the battery needs no judge.

## Note
This is the black-box breadth path. For the causal version, the construct that shows a Qwen signal also
goes into `visual_affect_battery_robust.ipynb` as a `BATTERY` row → steer/mediation/clean verdict (see
`JOINT_WRITEUP.md` §9). Re-run the patch on a fresh clone whenever Arnav updates the repo.
