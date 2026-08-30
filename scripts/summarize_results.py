#!/usr/bin/env python
"""Summarize a RESULTS/battery_multimodel/ folder into a model x experiment table + summary.csv.

    python scripts/summarize_results.py <path-to>/RESULTS/battery_multimodel

Prints, per run: model, exp, complete, headline, gates passed/total, and any effect whose 95% CI
excludes 0 (task: mean [lo,hi] n). Writes summary.csv next to the folder. Works on Drive (in Colab)
or a downloaded copy.
"""
import csv, glob, json, os, sys

def sig_effects(d):
    """Yield (task, mean, lo, hi, n) for deltas whose CI excludes 0."""
    prim = d.get("primary") or {}
    deltas = prim.get("deltas") if isinstance(prim, dict) else None
    if not isinstance(deltas, dict):
        return
    for task, v in deltas.items():
        if not isinstance(v, dict):
            continue
        m, lo, hi = v.get("mean"), v.get("ci_lo"), v.get("ci_hi")
        if m is None or lo is None or hi is None:
            continue
        if lo > 0 or hi < 0:                       # CI excludes 0
            yield task, m, lo, hi, v.get("n")

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    root = sys.argv[1].rstrip("/\\")
    rows = []
    for rf in sorted(glob.glob(f"{root}/**/results*.json", recursive=True)):
        try:
            d = json.load(open(rf, encoding="utf-8"))
        except Exception as e:
            print("skip", rf, e); continue
        if isinstance(d, list):
            continue
        model = d.get("model_id") or d.get("model") or os.path.basename(os.path.dirname(os.path.dirname(rf)))
        exp = d.get("experiment") or os.path.basename(os.path.dirname(rf))
        gs = d.get("gates_summary") or d.get("gates") or {}
        gates = f"{gs.get('passed','?')}/{gs.get('total','?')}" if isinstance(gs, dict) else "?"
        effs = list(sig_effects(d))
        eff_str = "; ".join(f"{t} {m:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}" for t, m, lo, hi, n in effs) or "-"
        rows.append(dict(model=str(model).split("/")[-1], exp=exp, complete=d.get("complete"),
                         headline=(d.get("headline") or "")[:48], gates=gates, effects=eff_str))
    if not rows:
        print("no results*.json under", root); return 1
    rows.sort(key=lambda r: (r["model"], r["exp"]))
    w = max(len(r["model"]) for r in rows)
    print(f"{'model':<{w}}  {'exp':<6} {'complete':<8} {'gates':<6} {'headline':<48} effects (CI excl 0)")
    for r in rows:
        print(f"{r['model']:<{w}}  {r['exp']:<6} {str(r['complete']):<8} {r['gates']:<6} {r['headline']:<48} {r['effects']}")
    out = os.path.join(os.path.dirname(root) or ".", "summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["model", "exp", "complete", "gates", "headline", "effects"])
        wr.writeheader(); wr.writerows(rows)
    n_valid = sum(1 for r in rows if r["complete"] is True)
    print(f"\n{len(rows)} runs, {n_valid} complete. wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
