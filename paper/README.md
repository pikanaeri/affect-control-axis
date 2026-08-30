# Paper (Overleaf)

- `main.tex` — the draft. Combines two runs, tagged inline: **[A]** = Arnav's original Gemma-4-E4B battery (`github.com/halli75/Algoverse`), **[M]** = this work's multi-model sweep (`github.com/pikanaeri/affect-control-axis`).
- `figures/` — `fig1_overrefusal_crossmodel` (exp09, cross-model) and `fig2_crossmodal_dissociation` (exp03), as PDF (for LaTeX) + PNG (preview).
- `make_paper_figs.py` — regenerates the figures. Numbers are hard-coded from `RESULTS/battery_multimodel/ALL_RESULTS.json` + halli75 `battery_results.md`; update them there for camera-ready.

**To use in Overleaf:** upload the `paper/` folder (or `main.tex` + `figures/`). Swap `\documentclass` for the workshop's style file when chosen. Prose sections marked *[Stub]* (Intro, Related Work, Discussion) are for you to expand; Methods, Results, the battery table, and the emotions table are filled from the results.

Content sources: Methods ← `docs/METHODS_DRAFT.md`; Results ← `docs/RESULTS_DRAFT.md`; battery provenance ← `docs/EXPERIMENTS.md`; positioning ← `docs/NOVELTY_SEARCH.md`.
