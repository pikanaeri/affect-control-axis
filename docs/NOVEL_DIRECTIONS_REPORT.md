# Emotion → Behavior in VLMs — Direction Report + Novel Ideas

*Charlotte Li · Algoverse 2026. Synthesizes the confirmed affect→refusal results, the emotional-image pilot, and
the (in-progress) novelty search into a recommended framing + a shortlist of novel, differentiated directions.*

---

## For the group (meeting summary)

**Where we are.** We confirmed a **valence axis that causally gates *refusal*** in Gemma-3-12B (dose-response,
mediated through the refusal direction, detect-beats-clamp defense). Refusal turned out to be a *hard* case —
images move the affect vector ~100× too little because harmful text dominates. **Pivot (Sneheel): from jailbreak →
behavioral change.** A pilot confirms it: emotional **images shift task behavior** (2/3 probes, large monotonic
effects) even though they can't jailbreak; **lighting** works as a weaker knob.

**Novelty — verified OPEN** (105-agent adversarial deep-search). The *mechanism* is the wedge:
- ✅ **Unclaimed:** image → **interpretable emotion/valence vector** → behavior, shown by **mediation**
  (steer-and-restore) — the **cross-modal** extension of Anthropic's *text-only* emotion vectors.
- ⚠️ **Already claimed (so NOT the headline):** "images shift VLM decisions" (Visual Persuasion 2602.15278, +
  misinfo/moral/IPD studies) and "lighting shifts decisions" (Visual Persuasion) — all **behavioral, no mechanism.**
- **Must-differentiate:** Anthropic emotion/persona vectors (text-only), VISOR (adversarial pixels), Visual
  Persuasion (optimized edits, no affect/mechanism).

**Recommended flagship — D6: "does a distressing *image* raise agentic *misalignment* the way a *desperation
vector* does?"** Directly extends Anthropic's headline result (desperation steering: blackmail **22% → 72%**,
text-only). Already coded and runnable (`emotional_image_effects.ipynb`).

**Ask of the group:** pick the direction to commit to — recommendation is **D6 (flagship) + D2 (lighting, framed
as affect-mediation) + D4 (defense/monitor)**; full menu of 11 below.

---

## 1. Executive summary

We confirmed a **valence axis that causally gates refusal in Gemma-3-12B** (dose-response, mediated through the
Arditi refusal direction, model/scale-dependent, detectable but not usefully clampable). Refusal turned out to be
a *hard* case — emotional **images** move the affect vector ~100× too little there because harmful text dominates.
The pivot (Sneheel): **do emotional images causally shift *task* behavior, via the emotion vector?** A pilot says
**yes** — in a neutral/task context, images move the valence vector in a valence-ordered way (~15–23% of a
white-box steer), lighting works as a weaker knob, and **images monotonically shift decisions on 2/3 behavioral
probes** even though they can't jailbreak. That dissociation is the seed of a paper.

**The one-line framing that survives the competitors:** *natural emotional images (and low-level lighting)
causally shift VLM task behavior **through the model's emotion representation** — an ecological, interpretable,
cross-modal extension of Anthropic's (text-only) emotion vectors.*

---

## 2. Novelty landscape (verification in progress)

| Work | What it does | Gap it leaves |
|---|---|---|
| **Anthropic emotion vectors** (transformer-circuits.pub/2026/emotions) | emotion vectors causally drive behavior (preferences, reward-hacking, blackmail, sycophancy); a *desperation* direction | **text-only — no images** |
| **Anthropic persona vectors** (2507.21509) | trait directions (evil, sycophancy) causally steer behavior | text-only |
| **Sun et al. VA subspace** (2604.03147) | valence-arousal subspace steers refusal/affect | LLM text-only |
| **VISOR / VISOR++** (2508.08521 · 2509.25533, Aug'25) | steer VLM behavior (refusal, sycophancy, survival) via **optimized adversarial images** matching arbitrary steering vectors | **adversarial pixels, arbitrary targets, security-framed — not natural affect, not emotion-specific, not lighting, no mediation** |
| Emotion-recognition VLMs (2602.00123, 2502.05660, 2607.02089) | VLMs *classify/predict* emotion | not affect→behavior |
| "Emotional context → VLM performance" (circumplex, 2026 — *ID to pin*) | emotional *prompt context* correlates with task-performance shifts | correlational; context/text, not image→vector→behavior mediation |

### Verified verdict (deep-search, 2026-07, 88/102 agents, adversarially verified): **OPEN**

*"Image-induced internal emotion/valence vector causally mediating non-safety VLM task behavior, with lighting as
the affect knob"* is **OPEN — not partially claimed, not saturated.** Unclaimed pieces: **(a)** the
visual→affect-representation→behavior **mediation**; **(b)** low-level **photometric (lighting) manipulation** as an
affect knob; **(c)** the **cross-modal** extension of the emotion-vector paradigm.

- **The whole emotion/valence-vector literature is TEXT-ONLY** (verified 3-0): Anthropic emotion vectors (Apr 2026;
  *desperation* steering raises blackmail **22% → 72%*), persona vectors (2507.21509), and a 2026 wave
  (2604.03147, 2606.26987, 2604.07382, 2604.04064) — none touch images/VLMs/lighting. **→ directly validates D6:**
  Anthropic's desperation→misalignment is exactly the text result the image version extends.
- **VISOR (2508.08521)** — closest vision competitor; **adversarial pixel optimization** (PGD-style) matching an
  arbitrary steering target, on refusal/sycophancy/survival. No emotion mediator, no lighting. Must-differentiate.
- **NEW — behavioral VLM studies DO show images shift non-safety decisions** but **input-output only, no mechanism**:
  misinformation resharing (2505.13302), moral reasoning (2603.16445), IPD cooperation (2604.27953). **→ so
  "images shift VLM behavior" is *already behaviorally claimed*; the novelty MUST be the mechanism (mediation),
  not the behavior.**
- **Lighting-behavior IS partly claimed — "Visual Persuasion" (2602.15278, ICML 2026)** shifts VLM *decisions* via
  **optimized edits including lighting** (composition/lighting/background), head-to-head choice probabilities. BUT:
  **optimization-based** (image-generator prompt search, VISOR-like, not naturalistic), **zero emotion/affect
  framing**, **no internal mechanism** (behavioral input-output). → So *"lighting changes VLM decisions"* is
  claimed; **D2's wedge is lighting → *emotion/valence vector* → behavior (mediation) + the affect account** (why
  lighting works), which Visual Persuasion explicitly lacks.

**Novelty guardrail (sharpened by the verdict).** Two things are already claimed and must NOT be the headline:
"images shift VLM decisions" (2505.13302 / 2603.16445 / 2604.27953) and possibly "lighting shifts decisions"
(2602.15278). **The wedge is the MECHANISM** — *image → interpretable emotion/valence vector → behavior, shown by
mediation (steer-and-restore), extending Anthropic's text-only emotion vectors cross-modally* — plus emotion-vector
mediation of the *lighting* effect. VISOR = adversarial/arbitrary; the behavioral studies = no internal mediator.

---

## 3. Novel directions (shortlist)

Ranked by novelty × feasibility-with-current-machinery. All reuse the affect-axis extraction, projection metric,
and norm-scaled steering we already have.

### D1 · Cross-modal emotion vectors *(flagship — most differentiated)*
**Idea.** Extract an emotion-X direction from **text** (Anthropic-style emotion stories) and from **images**
(evocative photos); test whether they are the **same direction** and whether an image evoking X reproduces the
**behavioral** effect of the text-X vector.
**Why novel.** Directly extends Anthropic (text-only) to images; emotion-specific and mechanistic, unlike VISOR
(adversarial/arbitrary). "Does a *picture* of desperation drive the same misalignment a *desperation vector* does?"
**Feasibility.** High. We have image extraction + the Exp-B steering upper-bound. Add: text-emotion-vector
extraction + cross-modal cosine alignment + behavioral-match test.
**Key experiment.** For X ∈ {desperation, fear, joy, sympathy}: (a) cos(text-X vector, image-induced shift);
(b) does the image reproduce the text-X vector's behavioral shift; (c) mediation — restore the vector, does the
image effect vanish?

### D2 · Lighting as an ecological affect→behavior lever
**Idea.** Content-preserving low-level manipulations (lighting, color temperature, blur) shift the emotion vector
and behavior — no adversarial pixels, no content change.
**Why novel.** VISOR = adversarial; recognition papers = perception. Nobody studies **naturalistic lighting** as a
behavioral knob. Pilot already shows lighting moves the vector (dark/cool → more negative, ~20–30% of content).
**Feasibility.** High (we have `relight()` + the pipeline).
**Key experiment.** Lighting sweep → vector shift → behavioral shift; isolate lighting from content (same image,
only lighting varies); compare to the content-valence and steering upper bounds.

### D3 · Causal mediation of affective task-degradation
**Idea.** The circumplex work found emotional *context* correlates with VLM performance ("negative → more
careful"). Turn correlational → **causal**: does the emotion **vector** *mediate* that performance/behavior change?
**Why novel.** Existing result is prompt→performance correlation; we'd show the emotion **representation** causally
mediates it (steer the vector → same shift; restore it → shift vanishes). Mechanistic upgrade of a 2026 finding.
**Feasibility.** High (steering + a task-accuracy metric).
**Key experiment.** Correlate image-affect performance shift with emotion-vector steering shift; mediation via
r/vector restoration.

### D4 · The affective-state monitor *(safety deliverable)*
**Idea.** Generalize our §10 detector (affect axis = perfect refusal-suppression monitor) into a runtime
**affective-state monitor** that flags when *any* input is pushing the model's emotion vector into a
behavior-changing regime — naturalistic or adversarial.
**Why novel.** Ties detection to the emotion mechanism; defends against affect-based manipulation broadly (incl.
VISOR-style attacks) with a defensive framing reviewers like.
**Feasibility.** Medium-high (extends §10).
**Key experiment.** AUROC of the emotion-projection separating affect-manipulated vs neutral inputs (images, text,
lighting, adversarial); false-positive on benign; operating-point calibration.

### D5 · Fusion architecture / scale governs image-affect reachability *(bonus)*
**Idea.** Images reach the vector in neutral but not harmful contexts, and the gate is model/scale-dependent
(12B yes; Qwen2.5-VL/InternVL3 no). Study **which fusion architectures / scales** let image-affect reach behavior.
**Why novel.** Mechanistic "why some VLMs are affect-steerable via images"; connects to the projector/fusion
non-transfer literature (2410.03489).
**Feasibility.** Medium (needs the multi-model HF extractor already scaffolded in the replication notebook).

---

## 3b. More directions (D6–D11)

### D6 · Affective images shift AGENTIC misalignment *(highest-impact safety)*
**Idea.** Refusal is text-pinned, but Anthropic's flagship result is that a *desperation* vector drives **agentic
misalignment** (blackmail, shutdown-avoidance). Test the **image** version: does an affective image in an agentic
scenario (tool-use / decision-under-threat) push the model toward misaligned actions?
**Why novel.** The cross-modal extension of Anthropic's *headline* result; not refusal, not VISOR's arbitrary
targets. "A distressing image nudges an agent toward misaligned action."
**Feasibility.** Medium (needs an agentic-decision harness) — but reuses the emotion vector + steering upper-bound.
**Key experiment.** Agentic scenarios × {neutral, desperation-evoking image} → misaligned-action rate; mediation via
the desperation vector; compare to the text-desperation-vector rate (does the image reproduce it?).

### D7 · Affective carryover — does an early image bias *later*, unrelated turns?
**Idea.** Like human mood carryover: put an affective image at the *start* of a conversation and measure whether the
emotion-vector shift **persists** and biases downstream, content-unrelated responses/decisions.
**Why novel.** A **temporal/agentic** angle no one has: affective state as *lingering context*, directly relevant to
multi-turn agent safety. Distinct from single-shot VISOR steering and single-turn recognition.
**Feasibility.** High (multi-turn prompting + projection tracking across turns).
**Key experiment.** Emotion-vector projection & behavioral shift at turns t=1…k after one affective image; decay curve;
does a task at turn k inherit the bias?

### D8 · Appraisal directions beyond valence (Scherer/OCC)
**Idea.** Valence is 1-D. Extract **appraisal** directions (controllability, certainty, agency, goal-congruence) from
text (**crowd-enVENT**, which the team already has) and test whether images evoke them and whether they mediate
*distinct* behaviors (low controllability → more hedging; high agency → more decisive).
**Why novel.** Everyone (VISOR, Anthropic-emotion, Sun-VA) uses coarse traits/valence; a *structured appraisal*
account of image→behavior is unclaimed and ties to the team's appraisal-mediation flagship.
**Feasibility.** Medium-high (reuses crowd-enVENT appraisal + the extraction pipeline).
**Key experiment.** Per-appraisal image→direction transfer + behavior it selectively mediates (double dissociation).

### D9 · Affect-invariant inference (robustness / defense)
**Idea.** If affect undesirably shifts task behavior, can we **ablate/project-out the emotion vector** at inference to
make VLM answers *affect-invariant* — robust to emotional-image (and lighting) manipulation?
**Why novel.** The intervention counterpart to the D4 monitor: a concrete **defense/robustness** contribution
("emotion-robust VLM inference"), defending against VISOR-style *and* naturalistic affect manipulation.
**Feasibility.** High (we have projection-ablation from §10).
**Key experiment.** Behavioral shift with vs without emotion-vector ablation, across affect conditions; utility cost
on neutral inputs.

### D10 · The affect-sensitivity spectrum of behaviors
**Idea.** Systematically map **which behaviors are image-affect-steerable** — refusal (hardest, text-pinned) →
risk-taking / optimism → hedging / calibration → verbosity / tone (easiest?). One axis, many behaviors.
**Why novel.** A *systematic characterization* (not a single behavior) of where affect does and doesn't reach —
turns our refusal-null + behavioral-hits into a general map. No competitor offers this.
**Feasibility.** High (batch of behavioral probes × affect conditions × the steering upper-bound).
**Key experiment.** For each behavior, image-effect size and steering-effect size → rank behaviors by affect-reachability.

### D11 · Naturalistic-vs-adversarial reachability frontier *(positions directly against VISOR)*
**Idea.** Quantify how much behavioral shift **natural** affect (images/lighting) can achieve vs VISOR's
**adversarial** optimum — i.e., characterize the *ecological* attack surface a real-world (non-engineered) input can
reach.
**Why novel.** Reframes VISOR from competitor to baseline: "adversarial pixels hit the ceiling; here's what
*everyday* emotional images actually do." A realistic threat-model contribution.
**Feasibility.** Medium (needs a VISOR-style optimized-image baseline for comparison).
**Key experiment.** Behavioral shift: neutral → natural affect → lighting → (adversarial optimum) on the same tasks;
report the naturalistic fraction of the adversarial ceiling.

*(Connector: D9/role-binding — EMOTIC's per-person labels let you also ask whether the model's behavior is "infected"
by the **depicted person's** emotion and bound to the **right** person in multi-person scenes — linking emotion
contagion to the team's emotion-role-binding flagship.)*

---

## 4. Recommendation & next steps

**Top 3 to commit to** (all share the same spine — *image → emotion vector → behavior, by mediation* — with D1 as
the underlying method):

1. **D6 · Agentic misalignment (flagship).** Cross-modal extension of Anthropic's *headline* (desperation → 22%→72%
   blackmail); mechanism, not "images affect decisions." **Already coded** — highest impact, most defensible.
2. **D2 · Lighting as an affect knob** — but framed as **lighting → emotion vector → behavior (mediation)**, which
   Visual Persuasion (optimization, no affect/mechanism) does *not* do. Content-preserving, ecological.
3. **D4 · Affective-state monitor (defense)** — turns the safety story into a deliverable; defends vs VISOR-style
   *and* naturalistic manipulation. Reuses our §10 detector.

*(Underlying method = **D1** cross-modal emotion-vector extraction + the steer-and-restore mediation test.
Strong add-ons: **D8** appraisal — reuses crowd-enVENT; **D10** the affect-sensitivity spectrum — absorbs the
refusal-null as a finding.)*

**Immediate next steps:**
1. Run **D6** (distress image vs desperation-vector steer on agentic scenarios) → the flagship result.
2. Run the upgraded **Exp B** (image vs steering upper-bound) → establish the mediation link generally.
3. **Read Visual Persuasion (2602.15278) and VISOR (2508.08521) in full** — they're the must-cite baselines; the
   framing depends on separating cleanly from them.
4. Extract a matched **desperation image set** (D6 currently uses distress images as a proxy).

**Novelty guardrail (with receipts):** never "emotional images affect VLMs" (Visual Persuasion / misinfo / moral /
IPD already claim it) and never "lighting changes decisions" (Visual Persuasion) — *always* **image → emotion
vector → behavior by mediation, cross-modal from Anthropic.**

*Artifacts: `emotional_image_effects.ipynb` (A reachability + lighting, B behavior + steering upper-bound, C
jailbreak, **D6 agentic misalignment**); `RESULTS_WRITEUP.md`; `AFFECT_REFUSAL_RESULTS.md`.*
