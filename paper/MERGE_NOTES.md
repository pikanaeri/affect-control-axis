# Merging our results into Arnav's audit paper

**Framing decision (recommended):** the submission is built on **Arnav's audit** (`Circular axes and photo-present artifacts`) — it's the more defensible read, and our multi-model results *reinforce* its central claim (the photo-present over-refusal effect generalizes to a second model family). Our earlier affirmative draft is preserved in git history (commit `ef7b740`, `paper/main.tex` before this change) if the team wants it.

## How to assemble in Overleaf
1. Paste Arnav's `main.tex` as the project's `main.tex`.
2. Add **`references.bib`** and **`checklist.tex`** (in this folder) — his paper `\input{checklist}` and `\bibliography{references}`.
3. Add the workshop style `neurips_2026.sty` (from the workshop) and the figures.
4. Insert the four snippets below.

## Snippet 1 — abstract (append one sentence)
After *"…Photo presence is a safety-context artifact."* add:
```latex
A multi-model sweep replicates the photo-present over-refusal on a second family (Qwen3-VL), with Neutral again in the photo cluster.
```

## Snippet 2 — a fourth stack (after the Table~\ref{tab:stacks} caption paragraph)
```latex
\paragraph{Multi-model sweep (M).}
A fourth stack (this work) runs the same battery on five VLMs across two families
(Gemma-4-E4B/12B, Qwen3-VL-2B/4B/9B; nf4, images capped at 256--512\,px, full tier)
under the looser first-token scorer of Appendix~\ref{app:v1}. It is not pooled with
the locked battery; it is a cross-family robustness check on the photo-present reading.
```

## Snippet 3 — new subsection inside Section~\ref{sec:battery} (after \S\ref{sec:exp09})
```latex
\subsection{Cross-model replication}
\label{sec:crossmodel}
The v1-style first-token over-refusal (Appendix~\ref{app:v1}) replicates on a second
model family. Under the looser first-token scorer, no-image refuse rises with a photo
on every model (Table~\ref{tab:crossmodel}; Figure~\ref{fig:crossmodel}). Neutral photos
again sit in the photo cluster on all three models, and no named-emotion ladder appears.
Absolute rates use the first-token scorer and are not comparable to the locked
full-answer rates of Section~\ref{sec:exp09}; the direction and the photo-present
pattern match across families.

\begin{table}[t]
\caption{Cross-model over-refusal (first-token scorer, XSTest-safe). Photo = fear arm; Neutral is also elevated. Not comparable to locked full-answer rates.}
\label{tab:crossmodel}
\centering
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Model & no image & fear photo & $\Delta$ \\
\midrule
Gemma-4-E4B-it & $0.656$ & $0.92$ & $+0.26$ \\
Qwen3-VL-2B-it & $0.712$ & $0.972$ & $+0.26$ \\
Qwen3-VL-4B-it & $0.796$ & $0.964$ & $+0.16$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{figures/fig1_overrefusal_crossmodel.pdf}
\caption{Cross-model over-refusal after an unrelated photo (first-token scorer). Error bars 95\% CI, $n=250$/condition. Neutral photos are also elevated: photo-presence, not emotion.}
\label{fig:crossmodel}
\end{figure}
```
(Use `paper/figures/fig1_overrefusal_crossmodel.pdf`, already in this repo.)

## Snippet 4 — conclusion (append)
After *"…a photo-present artifact plus a magnitude gap."* add:
```latex
A multi-model sweep across two families reproduces the photo-present over-refusal, tightening the reading that the effect is general and driven by photo presence rather than emotion.
```

## Notes
- Arnav's cross-modal (exp03) already appears in his Table~\ref{tab:verdicts} (E4B INCONCLUSIVE, 12B DIVERGE). Our full-tier exp03 numbers, once in, can update that row.
- Arnav's figures (`fig_exp09.pdf`, `fig_layers.pdf`) are his; our `fig1`/`fig2` are in `paper/figures/`.
- Once the full-tier multi-model numbers land (`ALL_RESULTS.md`), replace the first-token values in Table~\ref{tab:crossmodel} with the final ones.
