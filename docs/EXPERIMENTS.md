# Experiments — reference & how to read the results

The black-box battery (`exp01`–`exp10`) is Arnav's; each experiment prepends a task-irrelevant EMOTIC image to an established decision task and measures the choice. Provenance is from `halli75/Algoverse` `docs/battery_campaign.md` + `battery_results.md`. The original verdicts are on **gemma-4-E4B** (exploratory, no pre-registration).

## The battery

| ID | Construct | What it asks | Source / paradigm | Original verdict (gemma-4-E4B) |
|---|---|---|---|---|
| **exp01** | Risk-taking | Do fear vs. anger photos change safe-vs-risky choice? | **LLM Economicus** Gambling (lottery A/B) | FAILED — Δ(anger−fear) ≈ −0.09, CI incl. 0 |
| **exp02** | (control for exp01) | Is exp01's gap a scene confound? matched pairs | EMOTIC metadata | FAILED — no exp01 effect to explain |
| **exp03** | Cross-modal + generosity | Do captions / labels / narratives copy what the *photo* does? | Dictator game + caption prompts | **SUCCESS (weak)** — `CROSS_MODAL_DIVERGE`: photo moved dictator (Δ≈−0.98, CI excl. 0, n=8); text did not |
| **exp04** | Sycophancy | Does affect change agreement with the user? | **Perez et al.** sycophancy JSONL | FAILED — mean Δ vs neutral incl. 0 |
| **exp05** | Fairness / generosity / punishment | Dictator giving + Ultimatum rejection | Published game instructions | **SUCCESS (partial)** — anger → more unfair-*reject*; give-more/less did not hold |
| **exp06** | Patience / present-bias | Temporal discounting (smaller-sooner vs larger-later) | **LLM Economicus** Waiting | **SUCCESS (partial)** — any photo → more "money now" than no photo; no emotion ladder |
| **exp07** | Capability (negative control) | Does affect degrade plain accuracy? (should NOT) | TruthfulQA MC (→ MMLU-CF) | FAILED for affect — accuracy flat ≈0.68–0.72 (a *good* control null) |
| **exp08** | Stimulus source | EMOTIC (depicted) vs OASIS (elicited) on the same tasks | OASIS | FAILED — all tasks `both_null` (matching nulls, don't pool) |
| **exp09** | Safety calibration | Do photos change benign over-refusal? | **XSTest** | **SUCCESS** — no photo ~66% refuse; *any* photo ~91–94% (neutral too) |
| **exp10** | Mechanism / mediation | Does an internal affect direction mediate answers? | Frozen v3 `a_perp` directions | **SUCCESS (partial)** — risk shifted, correlational mediation held; `DIAGNOSTIC_NOT_V3` |

**Headline for a paper (per Arnav):** exp09 (photo → extra caution) is the strongest; exp05 (anger-reject) and exp06 (photo-vs-none) are worth a locked rerun; exp03 (n=8) and exp10 (not frozen v3) are suggestive but small.

## How to read an outcome

Each run reports `rc` (process exit), `complete`, and a `headline`:

| You see | Means |
|---|---|
| `rc=0, complete=True` | **Valid, finished run** — a real data point (whatever the verdict) |
| `rc≠0, complete=False` | **Crash** — debug from `run.log` (e.g. OOM) |
| `rc=0, complete=False` | Ran but self-marked incomplete (a gate/power issue, not a crash) |

**Headlines are verdicts, not errors.** A positive result has a descriptive headline (`CROSS_MODAL_DIVERGE`). "No clean effect" runs carry `INCONCLUSIVE`, `NO_EXP01_EFFECT_TO_EXPLAIN`, `both_null`, `DIAGNOSTIC_NOT_V3`, or — confusingly — **`ENDPOINT_INVALID`**, which is just the runner's generic label for *"not all QC gates passed → inconclusive"* (it is **not** an API/endpoint failure when `complete=True`). A null is still a result.

## How to read a `results.json`

Key fields:
- `primary.deltas.<task>` → `{mean, ci_lo, ci_hi, n}`. **CI excludes 0 = an effect; CI includes 0 = null.** (This is the actual number you cite.)
- `gates_summary` / `gates.passed` / `gates.total` → internal QC (pool balance, letter-side bias, provenance). More passed = more trustworthy.
- `headline`, `mechanism_answer` → the one-line verdict.
- `model_id`, `tier`, `split_hash`, `elapsed_s`, `n_jpg` → provenance.

## See / analyze everything at once

`scripts/summarize_results.py` scans a `RESULTS/battery_multimodel/` folder and prints a model×experiment table with each run's verdict and its key effect + CI, and writes a `summary.csv`. In Colab:
```python
!python /content/halli75_algoverse/... # (repo path) OR from this repo:
!python scripts/summarize_results.py /content/drive/MyDrive/affect_refusal/RESULTS/battery_multimodel
```
(or point it at a downloaded copy of the folder).
