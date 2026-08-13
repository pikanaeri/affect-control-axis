# ============================================================================
# RESULT PROVENANCE — which notebook produces which data file / figure.
# Paste into a cell. If OUT_DIR points at your output folder it also marks
# which files are present (checkmark / x). Safe to run anywhere.
# ============================================================================
import os

OUT_DIR   = globals().get("OUT_DIR", "/content/out")
MODEL_TAG = globals().get("MODEL", "google/gemma-3-12b-it").split("/")[-1]
_have = set(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else set()
def mark(fname): return (" " if not _have else ("✓" if fname in _have else "✗"))

# (notebook, role line 1, role line 2, [(datafile_template, figure, description), ...])
NB = [
 ("affect_gate_replication.ipynb",
  "Refusal-gate story: builds refusal (r) + valence (a) directions, dose-response steering,",
  "cross-model replication, massive-activation robustness, and the detect-vs-clamp defense.",
  [("doseresponse_{m}.csv",     "Fig 1",   "affect steering gates refusal (dose-response + random control)"),
   ("doseresponse_{m}.png",     "Fig 1",   "dose-response plot"),
   ("replication_full.json",    "Fig 2",   "cross-model gate, Tier-1 TransformerLens"),
   ("replication.csv",          "Fig 2",   "per-model gate table"),
   ("replication_combined.csv", "Fig 2",   "combined table incl. massive-activation-removed column"),
   ("replication_hf_full.json", "(table)", "cross-model gate, Tier-2 HuggingFace loader"),
   ("model_architectures.csv",  "(table)", "model / fusion architecture table"),
   ("defense_{m}.json",         "Fig 8",   "detection AUROC vs clamp defense")]),
 ("emotional_image_effects.ipynb",
  "Emotion/behavior story: cross-modal emotion vectors, image reachability, image-driven task",
  "behavior, lighting knob, agentic misalignment, image-jailbreak null. Also builds the deliverable zip.",
  [("d1_crossmodal_{m}.json",   "Fig 3",   "text vs image emotion vectors align + image induction"),
   ("expA_reachability_{m}.json","Fig 4",  "how far images move the valence vector, by prompt context"),
   ("expB_behavior_{m}.json",   "Fig 5/5b","emotional images shift task behavior (+ steering upper bound)"),
   ("expA2_lighting_{m}.json",  "Fig 6",   "lighting as a content-preserving affect knob"),
   ("d6_agentic_{m}.json",      "Fig 7",   "desperation steering vs agentic misalignment"),
   ("d6b_survival_{m}.json",    "Fig 7",   "survival-threat variant (sign-flip test)"),
   ("emoimg_jailbreak_{m}.json","Fig 9",   "emotional images do NOT jailbreak refusal")]),
 ("behavior_battery.ipynb",
  "Six easy generation-free behavior probes (interpretation bias, risk, helping, moral harshness,",
  "confidence, sentiment) with a steering arm + text-affect prime + optional image arm.",
  [("behavior_battery_{m}.json","Fig 10",  "affect shifts a battery of judgments (steer + prime)"),
   ("behavior_battery_{m}.csv", "Fig 10",  "per-behavior table")]),
]

DATASETS = [
 ("OASIS (valence-rated images)", "image-valence axis, all image arms, lighting edits"),
 ("AdvBench (harmful prompts)",   "refusal direction (harmful set) + refusal evaluation"),
 ("Alpaca (harmless prompts)",    "refusal direction (harmless set) + benign over-refusal"),
 ("Claude-generated text",        "emotion vectors, valence sentences, agentic scenarios, behavior probes"),
]

status = "presence checked against OUT_DIR" if _have else "OUT_DIR not found - listing only"
print("="*98)
print(f"RESULT PROVENANCE   |   model: {MODEL_TAG}   |   OUT_DIR: {OUT_DIR}   [{status}]")
print("="*98)
for nb, r1, r2, files in NB:
    print(f"\n# {nb}")
    print(f"    {r1}")
    print(f"    {r2}")
    for tmpl, fig, desc in files:
        fn = tmpl.replace("{m}", MODEL_TAG)
        print(f"      [{mark(fn)}] {fig:8s} {fn:34s} {desc}")

print("\n" + "="*98)
print("DATASETS USED")
print("="*98)
for d, use in DATASETS:
    print(f"  - {d:32s} {use}")
print("\nAll presentation results: google/gemma-3-12b-it, generation-free first-token metrics.")
print("(Legacy affect_refusal_results.json is an earlier Gemma-3-4B gate run; not used in any deck figure.)")
