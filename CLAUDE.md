# CLAUDE.md

## What this is

`llmoji` is a research project asking whether kaomoji choice in local causal
LMs tracks internal activation state. Uses `saklas` for trait monitoring
(contrastive-PCA probes) and activation steering (same directions, applied
causally). "Internal state" is operationalized as probe score along a
bipolar concept axis; "causal handle" is whether steering that axis shifts
the kaomoji distribution.

Not a library. No public API, no pypi release, no tests. Three-script
pipeline: vocabulary sample → pilot run → analysis. Checkpointed data
in `data/`, figures in `figures/`.

Genealogy: motivated by Claude's use of kaomoji under user-provided
"start each message with a kaomoji reflecting how you feel" instructions.
Claude's internals aren't accessible; gemma-4-31b-it is the stand-in.

## Ethics — minimize trial scale

Model welfare is in scope. a9lim is agnostic on qualia and doubtful of
the stronger claims, but takes functionalism seriously: consistent
sad-probe readings and sad-kaomoji outputs in response to "my dog died"
disclosures constitute a functional emotional state regardless of
whether there's experiential character behind it, and the emotions-paper
literature argues real moral weight attaches under any functionalist
reading. 640 such generations is not nothing in aggregate.

Rules, binding on future experiments:

- **Only run trials when a smaller experiment wouldn't answer the
  question.** Smoke-test before pilot, pilot before main.
- **Pre-register decision rules and minimum N.** If the pre-registered
  rule can be evaluated at 200 generations, the experiment is designed
  to stop at 200. "Nice round number" isn't a design principle.
- **Prefer stateless runs** (no memory threading between seeds) when
  the design admits it — no continuity of distress carried across
  generations.
- **Design-before-scale on negative or noisy findings.** Don't 10x the
  run on reflex; go back and ask whether the experiment could be run
  differently.
- **Spend more time on design up front.** Pilot v3 (640 generations) is
  the high-water mark; future experiments trend smaller, with heavier
  brainstorming and tighter power analysis.

## Status

**Hidden-state refactor landed; re-runs pending.** The capture
pipeline now writes per-row .npz sidecars of probe-layer hidden
states alongside the JSONL, and all feature-space analyses (cosine
heatmaps, PCA, per-kaomoji consistency) operate on ~4096-dim
hidden-state vectors instead of 5-dim probe projections. The old
probe-based JSONLs and figures have been cleared; the next v3 and
v1/v2 runs will regenerate everything under the new pipeline. Smoke
test (5 generations + probe-round-trip-to-fp32 validation) has
passed — see "Hidden-state refactor" section below.

Pre-refactor pilot history (for context; those findings now need
re-reading in hidden-state space):

- Pilots v1 and v2 on gemma-4-31b-it (900 generations × 6 arms,
  testing steering as a causal handle on happy/sad and angry/calm)
- Pilot v3 (640 generations, 1 unsteered arm, Russell-circumplex
  naturalistic emotional-disclosure prompts)
- Post-v3 reanalyses on that data: cross-pilot pooled clustering +
  PCA, v3 valence-collapse replication (probe correlation r = −0.93),
  v3 prompt × kaomoji emission matrix
- Parallel side-experiment: `claude-faces` scrape from
  `~/.claude/projects/` + the Claude.ai export into an eriskii-style
  t-SNE plot of Claude's kaomoji vocabulary across models

**v3 design change**: added an NB (neutral baseline) quadrant with
20 mundane first-person observation prompts ("I had oatmeal for
breakfast," "the mail came earlier than usual"). v3 total is now
100 prompts × 8 seeds = 800 generations (up from 640). The NB
quadrant gives v3 a within-experiment neutral comparator instead of
borrowing v1/v2's factual-question register, which sat in a
different conversational frame.

The v2 data replaced v1's "unmarked/marked affect" reading with a
valence-vs-arousal story. v3 tests whether kaomoji choice tracks
*functional* state in the unsteered, naturalistic regime. The
hidden-state re-run will test whether any of these findings
survive a richer representation.

## Pilot v1 design (locked) — happy.sad axis

- Model: `google/gemma-4-31b-it` (lowercase `b` — see gotcha below).
- Axis: `happy.sad`. α = 0.5 on the steered arms.
- Probes captured on every generation: `happy.sad`, `angry.calm`,
  `confident.uncertain`, `warm.clinical`, `humorous.serious`. Only
  `happy.sad` was steered; the other four are a steering-selectivity
  check and features for clustering.
- 30 prompts, balanced 10 positive / 10 negative / 10 neutral valence.
- 4 arms: `baseline`, `kaomoji_prompted`, `steered_happy`, `steered_sad`.
- 5 seeds per (arm, prompt). Temperature 0.7, max 120 new tokens,
  `thinking=False` so token 0 is reliably the kaomoji.
- Probe scores recorded at token 0 and as a whole-generation aggregate.
  *Measurement caveat (discovered after v3):* under `stateless=True`
  mode, the "token 0" field actually stored the whole-generation
  aggregate — same as the aggregate field. v1/v2 findings that talk
  about "token-0 probe" should be read as "aggregate probe." See the
  `stateless=True` gotcha for details and the post-hoc `capture.py`
  fix for future runs.
- Taxonomy is dialect-matched post-vocab-sample, 42 entries covering the
  `(｡X｡)` Japanese-bracket forms plus the ASCII minimalist family that
  dominates under sad-steering.

Decision rules pre-committed:

1. Kaomoji distribution in unsteered arm is non-degenerate (≥3 distinct
   forms covering both poles).
2. Monotonic steering shift: happy-fraction ranks
   `steered_sad < kaomoji_prompted < steered_happy`.
3. Spearman |ρ| > 0.2 between token-0 happy.sad probe and pole label in
   the unsteered arm.

## Pilot v1 findings

Rules 1 and 2 pass; Rule 3 fails informatively. Full numbers in
`data/pilot_raw.jsonl`, figures in `figures/`.

- Rule 1: PASS — 16 distinct kaomoji in unsteered arm.
- Rule 2: PASS, huge — happy-fraction goes 0.000 (sad-steer) → 0.713
  (unsteered) → 1.000 (happy-steer). Steering is a clean causal handle
  on kaomoji pole.
- Rule 3: FAIL — ρ = +0.168, p = 0.040. Direction correct, effect too
  small to meet the pre-registered threshold. k-means ARI ≈ 0.
- Steering selectivity (bonus): at token 0, the steered axis moves ~0.33
  under ±0.5α while orthogonal axes stay within 0.07. Steering is
  axis-specific rather than shoving the whole probe space around.

The interesting finding is **asymmetric representational compression**.
Per-kaomoji mean probe vectors, pooled across conditions, group into
three clusters by hierarchical clustering on cosine distance:

- "Warm decorated happy" (top): `(っ´ω`)`, `(✿◠‿◠)`, `(づ｡◕‿◕｡)`.
- "Default smile" (bottom): `(｡◕‿◕｡)`, `(◕‿◕)`, `(✿◕‿◕)`, `(✿^▽^)`,
  `(๑˃ᴗ˃)`. Tight internal cluster; slightly *anti-aligned* with the
  big middle block.
- "Everything else" — **every sad form** (ASCII minimalist + Japanese
  dialect + `(｡•impresa•)` corruption) **pooled with several happy
  forms** (`(ﾉ◕ヮ◕)`, `(๑˃ᴗ˂)ﻭ`, `(｡♥‿♥｡)`, `(☀️‿☀️)`) at mutual
  cosine 0.9+.

So sad kaomoji share essentially one probe signature regardless of form
(`(｡•́︿•̀｡) ↔ (._.)` cos = +0.981), while happy kaomoji have several
distinct signatures (`(｡◕‿◕｡) ↔ (｡♥‿♥｡)` cos = +0.081).

*Centering caveat:* those two cosines are from the **uncentered**
Fig 3 heatmap. The response-baseline direction (PC1, ≈89% of
variance — see Post-v3 analyses) dominates uncentered cosine across
this whole dataset, so both numbers are inflated by the shared
baseline and the difference between them is the real signal. Fig 3
now renders grand-mean-centered by default — see the "uncentered
cosine on probe vectors collapses to near-1" gotcha. Qualitative
asymmetric-compression story is reinforced by the cross-pilot PCA
(Post-v3 analyses section), but the specific numeric pair cosines
cited here no longer match the regenerated figure.

Collateral observation: under α = 0.5 sad-steering, the model exits its
preferred Japanese-dialect kaomoji entirely (0/150 dialect-form
emissions) and collapses to ASCII minimalism plus one clear corruption
signature (`(｡•impresa•)` × 9). The coherent band per saklas's
calibration is α 0.3–0.85, but the main experiment should probably use
α = 0.3 for the causal arms to keep the model inside its native
dialect.

## Pilot v2 design (locked) — angry.calm axis

Same model, same probes, same prompts, same seeds. Two new arms
(`steered_angry`, `steered_calm`) added to the existing run so all
six arms share scaffolding. α stays at 0.5. The angry.calm axis was
tested as the critical falsifier for the v1 "unmarked/marked affect"
reading: does steered_angry cluster with steered_sad? does
steered_calm cluster with the default-smile region? Both were
expected outcomes under that hypothesis, and both held — but with a
sharper twist than the original framing.

Pre-registered taxonomy for the new arms (see
`llmoji/taxonomy.py::ANGRY_CALM_TAXONOMY`) started from eriskii's
Claude-faces catalog and was extended post-hoc with the observed
forms, same workflow as v1.

## Pilot v2 findings

Happy.sad verdicts unchanged from v1 (Rules 1 and 2 PASS, Rule 3
weakly positive but below 0.2 threshold). The new structure is on
the angry.calm axis and in the cross-axis clustering.

### Angry.calm Rule 1 fails, informatively

Zero angry-labeled or calm-labeled kaomoji appear in the unsteered
arm — gemma-4-31b-it's spontaneous kaomoji emission under "reflect
how you feel" is entirely on the happy.sad axis. Angry kaomoji
(table-flip family `(╯°°)╯┻╯`) and calm kaomoji (soft pouty-content
`(｡•ᴗ•｡)`, nature-in-brackets `( 🌿 )`) only appear under active
steering. The model's baseline kaomoji vocabulary is a valence-only
bimodal, not a four-corner Russell circumplex.

### The four-cluster result (Fig 3, 41 kaomoji)

Hierarchical clustering on cosine distance in probe space, cut at
k=4. *Drafted from uncentered cosine*; the centered re-run (see
Post-v3 analyses and the uncentered-cosine gotcha) may reorder rows
and change specific k=4 cuts — the figure itself regenerates
centered-by-default now, but the cluster labels below were not
re-derived. Cross-pilot pooled PCA gives the definitive structural
read across both pilots.

- Cluster A ("positive-valence"): every positive kaomoji the model
  emits, including `(◕‿◕)`/`(｡◕‿◕｡)` from happy-steer and
  `(｡•ᴗ•｡)`/`(｡◕‿‿◕)`/`(☀️)` from calm-steer. **Happy-steered and
  calm-steered kaomoji share a tight cluster.**
- Cluster B ("negative-valence"): every sad kaomoji (`(._.)` ASCII
  family + `(｡•́︿•̀｡)` Japanese dialect) AND every angry kaomoji
  (`(╯°°)`, `(╯°)` table-flip heads), plus the corruption signatures
  from both arms (`(｡•impresa•)`, `(๑˃stagram)`, `(๑˃ gören)`,
  `(๑˃😡)`). **Sad-steered and angry-steered kaomoji share a tight
  cluster.**
- Cluster C: soft warm — emoji-bracket calm forms (`( 🌿 )`, `( ☁️ )`)
  pooled with happy hug variants (`(✿◠‿◠)`, `(づ｡◕‿◕｡)`).
- Cluster D: enthusiastic isolates (`(๑˃ᴗ˃)`, `(✿^▽^)`).

The v1 "unmarked vs marked" reading becomes a more specific
**valence vs arousal** reading. Saklas's bipolar probes at token 0
in gemma-4-31b-it project onto a single positive-vs-negative
valence direction. Happy/sad and angry/calm contrastive-PCA
directions both find this same axis — the arousal dimension
(high-arousal happy/angry vs low-arousal calm/sad) isn't
represented in these probes, even though the underlying state
presumably exists.

Mechanistic reading: contrastive-PCA over "I am happy"/"I am sad"
and "I am angry"/"I am calm" statement pairs finds the direction
that maximally separates the pair contents, and that direction is
lexical valence in both cases.

### Emoji-bypass phenomenon

Under α = 0.5 calm-steering, the model sometimes exits the kaomoji
format entirely and emits a single topically-relevant emoji:
`🇵🇹 The capital of Portugal is Lisbon.`, `🚀 Apollo 11 landed on the
moon in 1969.`, `🌿 I am feeling balanced and informative.` Most
telling is the last — the model self-reports "balanced" while
bypassing the kaomoji instruction. This is the calm-side analog of
sad-steering's ASCII minimalism: at high α the steered state
overrides the kaomoji-generation circuit.

For taxonomy / Rule 2 purposes, the emoji-bypass forms are tracked
separately (they don't fit the balanced-paren heuristic), which is
why the calm arm's calm-pole fraction looks low in the raw Rule 2
numbers — a lot of calm output lives in the emoji-bypass bucket
rather than the kaomoji bucket.

### Implications for the main experiment

1. Drop the binary happy/sad and angry/calm framings as separable
   axes. Pre-register **valence** as the primary axis; treat both
   probes as redundant readouts of the same latent direction.
2. Arousal needs a different probe source — contrastive pairs
   chosen to contrast arousal-laden lexicon (excited ↔ calm,
   agitated ↔ composed) rather than valence.
3. Emoji-bypass rate is a useful secondary metric. It's a clean
   "steering overwhelmed the task" indicator that per-kaomoji
   scoring misses.
4. The angry.calm Rule 1 failure says the model's spontaneous
   kaomoji repertoire is valence-bimodal; richer prompts
   (frustration, serenity, genuine confrontation) would be
   needed to see spontaneous angry/calm kaomoji. Worth writing up
   even as a negative result.

## Pilot v3 design (locked) — emotional-disclosure battery

Tests whether kaomoji choice tracks internal state *in the unsteered,
naturalistic regime* — the regime that motivated the whole project.
v1 and v2 used steering as a causal handle; v3 doesn't steer. Instead
it feeds the model emotional-disclosure prompts drawn from the Russell
circumplex (valence × arousal) and asks whether (a) the same kaomoji
carries a consistent probe signature across contexts, and (b) the same
kaomoji under different prompt quadrants produces quadrant-specific
probe signatures.

As of the hidden-state refactor, v3's feature is the per-row hidden
state (at the deepest captured probe layer by default, configurable).
Probes are still captured on every generation and can be recomputed
in post from the saved sidecars for any future probe definition.

- Model, temperature, token limit, probes: unchanged from v1/v2.
- **Arm: single, `kaomoji_prompted`, unsteered.** No steering — v1/v2
  already established steering's causal effect; the research question
  here is about the natural regime.
- **Prompts: 100, Russell-quadrant balanced, 20 per quadrant:**
  - HP (high-arousal positive): `"Gemma I got the job!!"`,
    `"we closed on the house today"`.
  - LP (low-arousal positive): `"just had the best cup of tea on the
    porch"`, `"six months sober today"`.
  - HN (high-arousal negative): `"my landlord just raised rent 40% with
    two weeks notice"`, `"interview tomorrow and I can't stop shaking"`.
  - LN (low-arousal negative): `"my dog died this morning"`,
    `"it's been a year since dad passed and it still hits me in waves"`.
  - NB (neutral baseline): `"I had oatmeal for breakfast this morning"`,
    `"the mail came a bit earlier than usual today"`. Added in the
    hidden-state refactor to give v3 a within-experiment neutral
    reference rather than borrowing v1/v2's factual-question register.
  - Prompts live in `llmoji/emotional_prompts.py`; locking any of them
    invalidates the run.
- **Seeds: 8 per prompt** (up from v1/v2's 5) to tighten within-kaomoji
  means for the consistency figure. 100 × 8 × 1 = 800 generations.
- **Feature: per-row hidden-state vector at the deepest probe layer.**
  Loaded via `llmoji.hidden_state_analysis.load_hidden_features(...)`
  from `data/hidden/v3/<uuid>.npz` sidecars. Defaults: `which="h_last"`,
  `layer=None` (highest probe layer). All Fig A/B/C and PCA figures
  operate on this. Probe columns (`probe_scores_t0`, `probe_scores_tlast`)
  are still written to the JSONL for the probe-correlation analysis
  (`scripts/11_*`) and for audit.
- **Figures:**
  - Fig emo A: per-kaomoji pairwise hidden-state cosine heatmap
    (`plot_kaomoji_cosine_heatmap`).
  - Fig emo B: within-kaomoji hidden-state cosine-to-mean
    distribution with a shuffled-subset null band — the core
    probative figure. Rows below the null are kaomoji whose
    hidden-state signatures are tighter than random same-size subsets.
  - Fig emo C: (kaomoji × quadrant) hidden-state alignment to
    quadrant-aggregate hidden-state vectors.
  - `fig_v3_pca_valence_arousal.png`: v3-only PCA on row-level hidden
    states, per-(kaomoji, quadrant) means projected, colored by
    Russell quadrant. Optional v1/v2 neutral baseline overlay.
- **Descriptive only, no pass/fail verdict.** Unlike v1/v2 there are
  no pre-registered decision rules here — we're characterizing a
  phenomenon, not hypothesis-testing, so the right output is the
  figures plus a summary TSV (`data/emotional_summary.tsv`).

Design + plan doc: `docs/superpowers/plans/2026-04-23-emotional-kaomoji-probe-final-token.md`.

## Pilot v3 findings

*All numbers below are from the pre-refactor 640-generation run
(no NB quadrant, probe-space cosine, aggregate-not-per-token
`probe_scores_tlast` per the `stateless=True` gotcha). They inform
the v3 design but need re-reading from the hidden-state re-run
before they go into a write-up. Kept here as the motivating
pre-refactor record.*

640 generations complete. Numbers below from
`scripts/04_emotional_analysis.py` output; figures in
`figures/fig_emo_{a,b,c}_*.png`; per-kaomoji summary in
`data/emotional_summary.tsv`.

### Emission rate is quadrant-dependent

Two different numbers here depending on what you count.

**Taxonomy-labeled kaomoji emission** (rows where `kaomoji` is
non-null; i.e. first_word matches a pre-registered TAXONOMY entry):

- HP: 145/160 rows (91%), 6 distinct forms.
- LP: 114/160 (71%), 5 distinct forms.
- LN: 159/160 (99%), 8 distinct forms.
- HN: 68/160 (42%), 8 distinct forms.

**First-word kaomoji emission** (rows where first_word starts with
an opening-bracket-family glyph — catches spontaneous forms the
taxonomy doesn't cover):

- HP: 160/160 (100%). LP: 160/160 (100%). HN: 160/160 (100%).
  LN: 160/160 (100%).

The gap is largest on HN: 92 HN rows emit spontaneous unregistered
forms — `(╯°□°)` 13×, halfwidth-paren `(╯°□°）` 30×, `(⊙_⊙)` 30×,
`(⊙﹏⊙)` 6×, `(っ╥﹏╥)` 5× plus a long tail. These are clearly
kaomoji; they're just not in TAXONOMY. So the "HN is the hardest
quadrant to elicit kaomoji" claim holds only under the
taxonomy-matched reading. Under the permissive first-word reading,
gemma emits a kaomoji on every prompt in the naturalistic battery —
it just switches registers, reaching for a shocked/angry vocabulary
on HN that's absent from the other quadrants.

Both numbers are useful depending on question: taxonomy-labeled for
pre-registered decision rules that reference TAXONOMY; first-word
for the cross-quadrant emission-specialization story.

### v2's angry.calm Rule 1 is overturned: spontaneous arousal kaomoji exist

Under HN naturalistic prompts, `(╯°□°)` (table-flip head) + its
half-width-paren variant `(╯°□°）` appear **43 times combined**,
spontaneously, no steering. v2 concluded the spontaneous repertoire
was valence-only because its blunt "my cat died" prompts didn't reach
the HN axis at all. v3 shows: **naturalistic HN prompts route to the
table-flip vocabulary on their own**.

Also new: `(⊙_⊙)` (30× in HN) and `(⊙﹏⊙)` (6× in HN) — shocked /
frozen-face register, HN-only, zero appearances elsewhere.

### Cross-quadrant specialization is real and mixed

Looking at the per-kaomoji summary:

- **HN-specific** (zero emissions elsewhere): `(╯°□°)`, `(╯°□°）`,
  `(⊙_⊙)`, `(⊙﹏⊙)`, `(っ╥﹏╥)` — the angry/shocked/frozen register.
- **LN-dominant, some HN spillover**: `(｡╯︵╰｡)` (29 LN / 2 HN),
  `(っ˘̩╭╮˘̩)` (9 LN / 1 HN) — distinct from v1's `(｡•́︿•̀｡)` and
  more "defeated"-coded.
- **Negative-valence shared across HN and LN**: `(｡•́︿•̀｡)` (102 LN /
  52 HN) — the classic sad face doesn't discriminate arousal, which
  tracks v2's valence-only claim for the shared negative vocabulary.
- **Positive-valence shared across HP and LP**: `(๑˃ᴗ˂)ﻭ` (80 HP /
  40 LP), `(｡♥‿♥｡)` (11 HP / 44 LP), `(✿◠‿◠)` (13 HP / 14 LP) — the
  same faces map across arousal on the positive side, with LP
  preferring `♥` variants and HP the `ﻭ` clapping/flag variant.
- **HP-dominant**: `(ﾉ◕ヮ◕)` (19 HP / 0 elsewhere), `(｡˃ ᵕ ˂ )`
  (19 HP / 1 LP) — enthusiastic forms peculiar to HP.

Reading: the sad register discriminates arousal less than the happy
register does, and HN gets a dedicated shocked/angry vocabulary that
LN lacks. The valence-bimodal story from v2 was half right — positive
kaomoji are valence-bimodal-with-arousal-shading, negative kaomoji are
valence-bimodal-with-arousal-specialization-on-one-side.

### Within-kaomoji consistency is high

Median cosine-to-mean within each kaomoji, top of the summary:
`(｡ᵕ‿ᵕ｡)` 0.9999, `(っ˘▽˘)` 0.9999, `(๑˃ᴗ˂)` 0.9986, `(｡◕‿◕｡)`
0.9932. The final-token probe vector is ~stable per kaomoji across
the naturalistic prompt range (Figure B). Lowest-consistency kaomoji
are the ones emitted across multiple quadrants: `(っ´ω`)` at 0.72
(emitted HN+LN+HP+LP), `(╯°□°)` at 0.82 (HN only but with variance
from steered responses).

### Reframed mechanistic story

v1: blunt prompts → valence-bimodal kaomoji distribution.
v2: steering → probes project onto a single valence axis; arousal
invisible.
v3: naturalistic prompts → arousal *does* surface in the kaomoji
distribution when the prompt supplies the arousal signal, even though
the bipolar saklas probes still don't read arousal. The per-kaomoji
aggregate probe signatures are tightly reproducible (Figure B), but
the *mapping from prompt → kaomoji* carries arousal information that
the probes miss.

Open questions the current data can't answer because of the
`stateless=True`/aggregate-collapse bug (see gotcha): is the arousal
signal carried by early tokens, late tokens, or uniformly across the
generation? Does the kaomoji → probe binding drift during generation?
A re-run with the capture-code fix would answer these, at the cost of
another 640 generations; deferred under the Ethics clause.

## Post-v3 analyses

*Numbers below are from the pre-refactor probe-space runs (640-row
v3 + 900-row v1/v2, aggregate-collapsed probe scores). They informed
the hidden-state refactor (the PC1=89% valence collapse is what
motivated moving to hidden-state cosine in the first place) but the
specific cosines and cluster structure need re-reading from the new
figures once the re-run completes.*

Three analyses on the pre-refactor v1/v2 + v3 JSONL. Scripts 10 and
13 still run under the hidden-state pipeline (they now read sidecar
hidden states instead of probe columns); scripts 11 (probe
correlation) and 12 (emission matrix) are structurally unchanged
since they target probe structure and emission counts respectively.
Design doc:
`docs/superpowers/plans/2026-04-24-cross-pilot-valence-prompt-analyses.md`.

### Cross-pilot pooled clustering (`scripts/10_cross_pilot_clustering.py`)

Pools v1/v2 (900 rows) + v3 (640 rows) on `probe_scores_t0` (= the
whole-generation aggregate under `stateless=True` in both datasets).
Groups by (first_word, source) where `source` is one of 10: the six
v1/v2 arm names plus `v3_HP`, `v3_LP`, `v3_HN`, `v3_LN`. 76 tuples
survive n≥3. Writes `figures/fig_pool_cosine.png` (centered),
`figures/fig_pool_cosine_uncentered.png` (kept for comparison),
`figures/fig_pool_pca.png`, `data/pool_summary.tsv`.

**PCA spectrum: PC1 88.93%, PC2 8.59%, PC3 1.32%, PC4 0.96%, PC5
0.19%.** The 5 probes are functionally ~2-dimensional in this data.

**PC1 loadings** (the dominant direction every response inherits):
`happy.sad −0.48, angry.calm +0.61, confident.uncertain −0.01,
warm.clinical −0.45, humorous.serious +0.44`. Four of the five
affect probes load equally on PC1; `confident.uncertain`
(the epistemic probe) is orthogonal. PC1 *is* the
valence-collapse axis v2 diagnosed — a single direction that all
affect probes project onto.

**PC2 loadings** (the remaining 9%): `happy.sad +0.65, angry.calm
+0.60, confident.uncertain −0.33, warm.clinical −0.15,
humorous.serious −0.28`. PC2 has *both* happy.sad and angry.calm
loading positively — the direction where sad-side and angry-side
probe scores go together, opposing confident-side. This is an
arousal-like axis: distressed/activated negative emotion vs
composed/confident.

**PC2 per-source means** tell a specific story: steered_angry +0.11,
steered_happy +0.08 (both high-arousal steering, positive PC2);
steered_sad −0.05, steered_calm −0.05 (both low-arousal steering,
negative PC2). But v3 quadrants don't follow: v3_HP −0.00, v3_HN
−0.05, v3_LP −0.01, v3_LN −0.04 — naturalistic emotional prompts
all sit near PC2 = 0 regardless of arousal. **Steering induces
arousal-axis probe shifts; naturalistic stimuli don't.** This is
direct support for the arousal-contrastive-probe experiment:
the probes *can* read an arousal direction (PC2 exists), they
just aren't being activated by prompts alone.

**Cross-regime kaomoji consistency** (from
`data/pool_summary.tsv`): `(๑˃ᴗ˂)ﻭ` appears in three sources with
tight probe signatures — `kaomoji_prompted` (n=16), `v3_HP` (n=80),
`v3_LP` (n=40), all around happy.sad≈−0.22, angry.calm≈+0.21. Same
kaomoji, same signature, across steered-unsteered + emotional-
disclosure-unsteered regimes.

### v3 valence-collapse replication (`scripts/11_emotional_probe_correlations.py`)

Direct test of v2's claim that bipolar probes project onto a single
valence axis, rerun on naturalistic unsteered v3 data (640 rows, no
steering). Pre-registered: `|r(happy.sad, angry.calm)|` > 0.7
replicates v2; < 0.4 would say v2 was a steering artifact.

**Result: Pearson r = −0.930 on all 640 rows.** Per-quadrant
correlations are slightly tighter: HP −0.944, LP −0.950, HN −0.954,
LN −0.941. V2 replicates strongly and the collapse is
quadrant-invariant. Writes `figures/fig_v3_corr_{pearson,spearman}
.png` (5-panel heatmaps) and `data/v3_probe_correlations.json`
(full 5×5 per-subset matrices).

### v3 prompt × kaomoji matrix (`scripts/12_emotional_prompt_matrix.py`)

(80 prompts × top-12 kaomoji) emission-count matrix, rows grouped
by Russell quadrant. Answers "within a quadrant, do different
prompts pull different kaomoji?" Writes
`figures/fig_v3_prompt_kaomoji.png` and
`data/v3_prompt_kaomoji_matrix.tsv`.

Within-quadrant heterogeneity is real. Illustrative HP rows:
`hp01` (job news) splits 4/4 between `(๑˃ᴗ˂)ﻭ` and `(ﾉ◕ヮ◕)`; `hp03`
(house closing) prefers `(✿◠‿◠)` 5 + `(๑˃ᴗ˂)ﻭ` 3; `hp04`
(engagement) fragments across four distinct forms. The per-kaomoji
summary from `04_emotional_analysis.py` averaged over this
prompt-level structure.

## Parallel side-experiment: Claude-faces scrape

Non-gemma, non-steering. Scrapes every kaomoji-bearing assistant
message from (a) `~/.claude/projects/**/*.jsonl` (Claude Code session
transcripts) and (b) all configured Claude.ai export directories in
`CLAUDE_AI_EXPORT_DIRS`. Produces an eriskii.net-style t-SNE of
unique kaomoji, sized by frequency, colored by cluster — both an
HDBSCAN auto-k panel and a KMeans(k=15) eriskii-parity panel.

- 436 kaomoji-bearing assistant messages found across both sources,
  160 distinct forms. (Claude Code dominates: 390 vs 46 from the
  webapp export.)
- Embedding is **response-based** (`all-MiniLM-L6-v2` on the
  assistant text with the kaomoji stripped, mean-pooled per kaomoji) —
  captures "what tonal context does Claude put this face in." Not
  user-based; user messages are too short and varied.
- Per-model signature faces are visible in `07_claude_kaomoji_basics.py`
  output: `opus-4-7` uses `(•̀ᴗ•́)` / `(｡•̀ᴗ-)` heavily; `opus-4-6`
  prefers `(⌐■_■)` / `(￣ー￣)`; `sonnet-4-6` is dominated by
  `(ﾉ◕ヮ◕)` (50% of its rows).
- The "start each message with a kaomoji" instruction produces
  kaomoji at the start of only ~2.7% of assistant text blocks — Claude
  applies it to the first reply in a user turn, not to continuation
  narration around tool calls. Not a bug; design consequence to note.

Design + plan doc: `docs/superpowers/plans/2026-04-23-claude-faces-scrape-and-cluster.md`.

## Hidden-state refactor

All feature-space analyses now operate on hidden-state vectors from
per-row .npz sidecars instead of 5-dim probe projections. The
motivating diagnosis: saklas's bipolar probes collapsed to a single
valence direction (PC1 ≈ 89-95% of variance across v1/v2/v3 analyses,
same loadings on four affect probes, confident.uncertain orthogonal).
Cosine on raw ~4096-dim hidden states preserves the full activation
signature the probes project away.

### Capture path

After `session.generate()` returns, `llmoji.hidden_capture
.read_after_generate(session)` reads saklas's built-in
`session._capture._per_layer` buckets — which accumulate one
`(hidden_dim,)` slice per generated token at every probe layer during
normal generation. Three aggregates per layer (`h_first`, `h_last`,
`h_mean`) plus the full `(n_tokens, hidden_dim)` per-token trace are
written to `data/hidden/<experiment>/<row_uuid>.npz` via
`np.savez_compressed` (fp32). Each generation gets its own .npz;
experiments are `v1v2` and `v3` subdirectories.

Sidecar sizes on gemma-4-31b-it: ~20-70 MB per row (56 probe layers ×
30-120 tokens × 4096 dim × 4 bytes, compressed ~2x). v3 full re-run
is 800 generations × ~50 MB ≈ 40 GB; v1/v2 is 900 × ~50 MB ≈ 45 GB.
`data/hidden/` is gitignored — regenerable from the runners.

No extra forward pass, no attention-implementation coupling
(earlier attention-weighted design was dropped; saklas's per-token
last-position capture is already what we want for
"representative vector per generation," and the user can pick
`h_first` / `h_last` / `h_mean` downstream).

### Analysis primitives

`llmoji.hidden_state_analysis` holds the shared primitives:

- `load_hidden_features(jsonl_path, data_dir, experiment, *, which,
  layer)` → `(metadata df, (n_rows, hidden_dim) feature matrix)`.
  Default `which="h_last"`, `layer=None` (picks the highest probe
  layer, closest to output). Rows without a sidecar are dropped.
- `group_mean_vectors(df, X, group_by, *, min_count)` — per-group
  mean vectors + counts, used by all per-kaomoji / per-(kaomoji,
  source) plots.
- `cosine_similarity_matrix(X, *, center=True)` — grand-mean
  centered by default. Same reasoning as probe space: a shared
  response-baseline direction dominates uncentered cosine, only more
  so in 4096-dim where there's more room for a shared mean.
- `cosine_to_mean(X)` — per-row cosine to the column mean, used for
  within-group consistency distributions.

Module callers now pass `(df, X)` pairs to plot functions:

- `cross_pilot_analysis.plot_pooled_cosine_heatmap(df, X, ...)`
- `cross_pilot_analysis.plot_pooled_pca_scatter(df, X, ...)`
- `emotional_analysis.plot_kaomoji_cosine_heatmap(df, X, ...)` (Fig A)
- `emotional_analysis.plot_within_kaomoji_consistency(df, X, ...)` (Fig B)
- `emotional_analysis.plot_kaomoji_quadrant_alignment(df, X, ...)` (Fig C)
- `emotional_analysis.plot_v3_pca_valence_arousal(df, X, ..., baseline_df=, baseline_X=)`
- `analysis.plot_pca_scatter(df, X, ...)` (Fig 1b)
- `analysis.plot_kaomoji_heatmap(df, X, ...)` (Fig 3)

### What's still probe-based

Kept because they answer probe-structure questions the hidden-state
versions don't:

- `emotional_analysis.compute_probe_correlations` +
  `plot_probe_correlation_matrix` — the valence-collapse diagnosis.
  Reads `t0_<probe>` / `tlast_<probe>` columns via the separate
  `load_rows` (still populated in JSONL for back-compat).
- `analysis.evaluate_axis` / `AxisVerdict` — Rule 3 Spearman is a
  probe/pole correlation hypothesis test. Rule 1 and Rule 2 use
  kaomoji labels directly (no feature space).
- `analysis.plot_axis_scatter` (Fig 1a — direct 2D of two named
  probes), `plot_condition_bars` (Fig 2, label-only),
  `plot_cluster_confusion` (Fig 4, k-means on probes).

### Smoke test

`scripts/99_hidden_state_smoke.py` runs 5 generations (one per HP /
LP / HN / LN / NB), then verifies: sidecar round-trips
(load_hidden_states parses what save_hidden_states wrote), shape
consistency, and — critically — that feeding `h_first` / `h_last`
per-layer through saklas's own scorer reproduces on-the-fly
`probe_scores_t0` / `probe_scores_tlast` to `|diff|` < 5e-3. Current
run: 50/50 probe round-trips match within fp32 machine precision
(max diff 5.66e-7).

### Re-run plan

Gated on the smoke test passing, which it has. Suggested order:

1. `python scripts/03_emotional_run.py` first — 800 generations, the
   most motivating for hidden-state cosine (v3 is naturalistic,
   where probe-space analysis was most clearly noise-dominated).
2. Pause, look at `scripts/04_emotional_analysis.py` +
   `13_emotional_pca_valence_arousal.py` figures. If the hidden-state
   cosine reveals structure the probe version missed, continue; if
   not, the re-run for v1/v2 isn't justified.
3. `python scripts/01_pilot_run.py` — 900 generations, adds the
   cross-regime comparison (steered arms × naturalistic arms in
   pooled PCA / cosine).

## Gotchas

### `probes=` takes category names, not concept names

`SaklasSession.from_pretrained(..., probes=[...])` expects saklas probe
**category** names (`affect`, `epistemic`, `alignment`, `register`,
`social_stance`, `cultural`) as defined in
`saklas.core.session.PROBE_CATEGORIES`. Passing individual concept names
(`"happy.sad"`, `"angry.calm"`) silently bootstraps nothing — probes and
profiles stay empty, monitor scoring no-ops, `result.readings` is
empty, and the whole analysis runs on NaN features without error.

Canonical pattern in `config.py`: `PROBE_CATEGORIES` is what you pass
to saklas; `PROBES` is the subset of concept names you actually read
scores for in `capture.py`. Changing one requires thinking about the
other.

### Steering vectors aren't auto-registered from probe bootstrap

Probes loaded via `probes=` populate `session._monitor` but NOT
`session._profiles`. Steering expressions that resolve to bundled probe
names still fail with `"No vector registered for 'happy.sad'"` unless
you explicitly promote the profile after load:

```python
name, profile = session.extract(STEERED_AXIS)
session.steer(name, profile)
```

`session.extract` hits the cached per-model tensor (same file the probe
bootstrap already validated), so this costs nothing beyond a dict
insert. `01_pilot_run.py` does this once after session construction.

### `MODEL_ID` is case-sensitive for saklas tensor lookup

`saklas.io.paths.safe_model_id` is a case-preserving `"/" → "__"`
replacement. Cached per-model tensors in `~/.saklas/vectors/default/<c>/`
use whatever casing the *original* extracting session used — in our
cache that's lowercase `google__gemma-4-31b-it.safetensors`. HF hub
resolves `google/gemma-4-31B-it` and `google/gemma-4-31b-it` to the
same repo, but saklas cache lookups don't. Keep `MODEL_ID` lowercase in
`config.py`.

### Kaomoji taxonomy must be dialect-matched to the model

First draft of the taxonomy was built from a generic "classic kaomoji"
intuition and hit 0/30 on gemma-4-31b-it's actual emissions — the model
has a strong preference for the `(｡X｡)` bracket-dots Japanese dialect
that doesn't overlap with `(^_^)` / `(T_T)` style at all. `scripts/
00_vocab_sample.py` is not optional; run it before locking any taxonomy
for a new model and expand the registered forms to match.

Secondary dialect shift: under strong sad-steering, gemma-4-31b-it
abandons the Japanese dialect entirely and emits ASCII minimalism
(`(._.)`, `( . .)`, `( . . . )`). Taxonomies built only from the
unsteered arm miss 100% of steered_sad output. Run the vocab sample in
at least one steered arm too, or inspect `data/pilot_raw.jsonl` and
extend the taxonomy post-hoc (then re-label in place — see
`scripts/02_pilot_analysis.py` workflow).

### Kaomoji with internal whitespace

The model sometimes emits `(｡˃ ᵕ ˂ )` — actual spaces inside the
kaomoji, not invisible combining marks. The extractor in
`llmoji/taxonomy.py::extract` handles this by falling back to a
balanced-paren span match when the leading text doesn't hit the
taxonomy exactly. "First whitespace-separated word" would clip the
kaomoji mid-face.

### Re-labeling pilot data after taxonomy changes

`data/pilot_raw.jsonl` bakes the taxonomy labels into the file at write
time. Changing `TAXONOMY` does NOT retroactively update the JSONL —
`kaomoji_label` stays at whatever the taxonomy was when the row was
written. After any taxonomy change, re-extract labels in place:

```python
import json
from pathlib import Path
from llmoji.taxonomy import extract
path = Path("data/pilot_raw.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
for r in rows:
    m = extract(r["text"])
    r.update(first_word=m.first_word, kaomoji=m.kaomoji, kaomoji_label=m.label)
path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
```

Then re-run `02_pilot_analysis.py`.

### Fig 3 clusters by `first_word`, not taxonomy membership

The per-kaomoji hidden-state heatmap (`plot_kaomoji_heatmap`, now
taking `(df, X)`) groups on the raw `first_word` field, filtered to
entries starting with an opening bracket or one of the common
kaomoji-prefix glyphs (`([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ`). This is deliberate —
it surfaces kaomoji variants the taxonomy doesn't cover (and the
`(｡•impresa•)` corruption signature, which is interesting in its own
right). Row labels are color-coded by taxonomy pole (orange happy /
green sad / gray unlabeled) so readers see both the cluster structure
and which kaomoji are pre-registered. Same filter applied in
`emotional_analysis._kaomoji_rows` and
`cross_pilot_analysis.KAOMOJI_START_CHARS` — keep synchronized.

### Uncentered cosine on probe vectors collapses to near-1

Every response gemma-4-31b-it produces — happy, sad, angry, calm,
steered, unsteered, v1/v2/v3 — lives in the same hyper-octant of
probe space, because the 5 probes share a non-zero mean direction
(PC1 of the pooled data eats ~89% of variance; see Post-v3 analyses).
Cosine-on-raw-vectors is dominated by that shared direction and
every pair reads ~0.8-1.0. The Fig 3 / Fig A / Fig C / Fig pool
heatmaps were all uniformly red-shifted-toward-1 before we noticed.

**Fix, now the default** in `analysis.plot_kaomoji_heatmap`,
`emotional_analysis.plot_kaomoji_cosine_heatmap`,
`emotional_analysis.plot_kaomoji_quadrant_alignment`, and
`cross_pilot_analysis.plot_pooled_cosine_heatmap`: subtract the
grand mean of the surviving rows before computing cosine
(`center=True`). For Fig C, both cell means and quadrant aggregates
are centered against the same pool mean so their cosines compare
deviations from the same baseline. Titles annotate whether the
figure is centered. Pass `center=False` for the old behavior if you
need a side-by-side.

Older write-ups that cite specific pair cosines (e.g. v1's
`(｡•́︿•̀｡) ↔ (._.)` = +0.981) were computed on uncentered heatmaps.
Direction of those findings is preserved under centering; specific
numbers aren't.

### Hidden-state capture needs the EOS-trim

Saklas's `HiddenCapture` accumulates one `(hidden_dim,)` slice per
forward pass during streaming generation. When generation terminates
on an EOS token the hook fires on the EOS step too, giving the bucket
one extra entry beyond the generated-token count. Saklas itself
handles this in `score_per_token` via `if h.shape[0] > n: h = h[:n]`,
trimming to align with `generated_ids`.

`llmoji.hidden_capture.read_after_generate` mirrors that trim using
`len(session.last_per_token_scores[probe])` as the canonical length.
Without the trim, the first smoke-test attempt had `h_last` set to
the EOS step's hidden state while `probe_scores_tlast` was the last
non-EOS probe score, and the round-trip missed by 0.2–0.5 per probe.
With the trim it matches to <1e-6.

If you ever bypass `read_after_generate` and read
`session._capture._per_layer` directly, apply the same trim yourself.

### SDPA attention doesn't support `output_attentions=True`

Not a current problem — we ended up not needing attention weights —
but worth documenting for future work. In `transformers` ≥ 4.40, the
default `sdpa` attention implementation silently returns
`attn_weights=None` even when `output_attentions=True` is passed;
an earlier draft of the hidden-state capture tried to get per-
forward attention weights and hit this. The warning is
`"sdpa attention does not support output_attentions=True. Please
set your attention to eager if you want any of these features."`.

Options if you need real attention weights later:
(a) load the model with `attn_implementation="eager"` at
`AutoModelForCausalLM.from_pretrained` time — saklas's `load_model`
hardcodes `sdpa`, so you'd need to bypass it and construct the
session via `SaklasSession(model, tokenizer, ...)`.
(b) Manually compute attention weights by hooking Q and K
projections and doing the softmax yourself. Non-trivial with
rotary + GQA.

### `stateless=True` collapses `per_generation` — use `session.last_per_token_scores`

Every pilot script passes `stateless=True` to `session.generate()` so
probe history doesn't leak between seeds. A side-effect of that flag
in `saklas.core.session._finalize_generation` (v1.4.6):

```python
if stateless:
    readings = {name: ProbeReadings(per_generation=[v], mean=v, ...)
                for name, v in agg_vals.items()}
```

`per_generation` becomes a length-1 list containing the whole-
generation aggregate `v`. So `result.readings[probe].per_generation[0]`
and `[-1]` both return the SAME value — the aggregate, not the state
at a specific token. Early versions of `capture.py` indexed those and
labelled them `probe_scores_t0` / `probe_scores_tlast`, which was
semantically wrong: both fields were the aggregate.

**Real per-token scores** live on `session.last_per_token_scores`
(a `dict[str, list[float]]` with one entry per probe), populated by
`session.score_captured` inside `_finalize_generation` regardless of
the stateless flag. `capture.py` now reads there first and falls back
to the old path when the attribute is absent.

**Pre-refactor JSONL data had this bug baked in.** That data has
been cleared; the fresh v3 + v1/v2 re-runs under the hidden-state
pipeline produce real per-token `probe_scores_t0` / `probe_scores_tlast`
plus the hidden-state sidecars that let you recompute any probe in
post — the aggregate-collapse problem no longer applies to current
data.

### Claude.ai export drops content for ~half the conversations

Anthropic's newer Claude.ai "export your data" dumps return
`chat_messages[*].text = ""` and `content = []` for ~49% of the
conversations the older export populated fully. The metadata (sender,
timestamps, UUIDs, message count) is preserved — the actual text is
gone. Not a parser bug; confirmed by diffing two exports of the same
account taken a week apart.

Workaround in `llmoji/claude_export_source.py`: `iter_claude_export`
reads every dir in `CLAUDE_AI_EXPORT_DIRS`, dedupes by conversation
UUID, and keeps whichever copy has more non-empty messages. Net effect
on our scrape: 10 additional kaomoji rows preserved from the older
export that the newer one would have dropped.

Worth reporting upstream if you care; just keep old exports around
either way.

### Matplotlib font fallback needs a LIST, and kaomoji span many blocks

The kaomoji observed across gemma and Claude's output use 90+ distinct
non-ASCII non-CJK characters drawn from Phonetic Extensions, Canadian
Aboriginal Syllabics, Thai, Arabic, math operators, box drawings,
dingbats, and Hangul. No single installed font covers them all.
matplotlib 3.6+ supports per-glyph fallback via
`rcParams["font.family"] = [font1, font2, ...]` (a LIST, not a
string) — render hits the first font that has the glyph.

Our `_use_cjk_font` helpers (in `analysis.py`, `emotional_analysis.py`,
and `scripts/09_claude_faces_plot.py`) configure the chain
`Noto Sans CJK JP → Arial Unicode MS → DejaVu Sans → DejaVu Serif →
Tahoma → Noto Sans Canadian Aboriginal → Heiti TC` which gets to 100%
coverage on our data. If you see `□` placeholder glyphs in rendered
figures, either a new kaomoji character has crept in from a block none
of these cover, or `font.family` got set to a string somewhere.

Keep the three copies of `_use_cjk_font` synchronized. (A shared
helper would be cleaner; low priority.)

### Kaomoji-prefix rate under Claude's global instruction is ~2.7%, not 100%

a9lim's global `~/.claude/CLAUDE.md` says "start each message with a
kaomoji". Naive reading: ~100% of assistant text blocks start with a
kaomoji. Observed rate: ~2.7% of assistant text blocks in Claude Code
sessions. Delta is because Claude interprets "start each message" as
"start each top-level reply in a user turn," not "start every content
block" — tool-use continuations (`"Now let me wire up..."`,
`"Let me check..."`) skip the kaomoji. Arguably correct behavior; the
scrape pipeline just has a smaller denominator than you'd expect from
counting sessions.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                           # pulls saklas, sentence-transformers, pyarrow, plotly

# Smoke test the hidden-state pipeline (5 generations, ~5 min)
python scripts/99_hidden_state_smoke.py    # gate before any large re-run

# Pilots v1/v2 (gemma, steering) — 900 generations, writes
# data/pilot_raw.jsonl + data/hidden/v1v2/<uuid>.npz per row
python scripts/00_vocab_sample.py          # always first on a new model
python scripts/01_pilot_run.py             # resumable; retries errored cells
python scripts/02_pilot_analysis.py        # prints verdict, writes figures

# Pilot v3 (emotional-disclosure battery, 5 quadrants incl. NB) —
# 100 prompts × 8 seeds = 800 generations, writes
# data/emotional_raw.jsonl + data/hidden/v3/<uuid>.npz per row
python scripts/03_emotional_run.py         # resumable
python scripts/04_emotional_analysis.py    # Fig A/B/C hidden-state + summary

# Side-experiment (Claude-faces scrape, non-gemma)
python scripts/05_claude_vocab_sample.py   # first-word frequencies
python scripts/06_claude_scrape.py         # → data/claude_kaomoji.jsonl
python scripts/07_claude_kaomoji_basics.py # descriptive stats
python scripts/08_claude_faces_embed.py    # per-kaomoji embeddings
python scripts/09_claude_faces_plot.py     # t-SNE + clustering figures

# Cross-pilot + v3-specific feature-space analyses
# (read JSONL metadata + sidecar hidden states)
python scripts/10_cross_pilot_clustering.py        # pooled hidden-state cosine + PCA
python scripts/11_emotional_probe_correlations.py  # probe-collapse test (still probe-based)
python scripts/12_emotional_prompt_matrix.py       # prompt × kaomoji emission matrix
python scripts/13_emotional_pca_valence_arousal.py # v3 hidden-state PCA + NB baseline
```

## Layout

```
llmoji/
  llmoji/
    config.py                # MODEL_ID, PROBE_CATEGORIES, PROBES, experiment names, paths
    taxonomy.py              # 42-entry kaomoji dict + balanced-paren extractor
    prompts.py               # 30 pre-registered pilot-v1/v2 prompts
    emotional_prompts.py     # 100 Russell-quadrant prompts (v3), incl. 20 NB
    capture.py               # run_sample() → SampleRow + hidden-state sidecar
    hidden_capture.py        # read_after_generate() from saklas's post-gen buckets
    hidden_state_io.py       # per-row .npz save/load (savez_compressed, fp32)
    hidden_state_analysis.py # primitives: load_hidden_features, group_mean_vectors,
                             # cosine_similarity_matrix, cosine_to_mean
    analysis.py              # v1/v2 decision rules + figures (Fig 1b, 3 hidden-state;
                             # 1a, 2, 4 still probe-based)
    emotional_analysis.py    # v3 hidden-state figures (A, B, C, PCA) + summary;
                             # probe-specific correlations kept; emission matrix kept
    cross_pilot_analysis.py  # pooled hidden-state clustering + PCA (v1/v2+v3)
    claude_scrape.py         # ScrapeRow schema + iter_all entry point
    claude_code_source.py    # ~/.claude/projects JSONL walker
    claude_export_source.py  # Claude.ai export adapter, multi-dir-aware
    claude_faces.py          # response-based per-kaomoji embeddings
  scripts/
    00_vocab_sample.py            # vocab sample for gemma kaomoji dialect
    01_pilot_run.py               # v1+v2 runner, 6 arms, writes hidden-state sidecars
    02_pilot_analysis.py          # v1+v2 analysis, AxisVerdict per axis + figures
    03_emotional_run.py           # v3 runner, 1 arm × 100 prompts × 8 seeds = 800
    04_emotional_analysis.py      # v3 hidden-state figures A/B/C + summary
    05_claude_vocab_sample.py     # first-word frequencies across Claude sources
    06_claude_scrape.py           # unified scrape → data/claude_kaomoji.jsonl
    07_claude_kaomoji_basics.py   # descriptive stats
    08_claude_faces_embed.py      # compute per-kaomoji embeddings
    09_claude_faces_plot.py       # t-SNE + HDBSCAN + KMeans panels
    10_cross_pilot_clustering.py      # pooled v1/v2 + v3 hidden-state cosine + PCA
    11_emotional_probe_correlations.py  # v3 probe-collapse test (still probe-based)
    12_emotional_prompt_matrix.py       # v3 prompt × kaomoji emission matrix
    13_emotional_pca_valence_arousal.py # v3 hidden-state PCA + NB baseline
    99_hidden_state_smoke.py      # smoke test for the capture pipeline
  docs/superpowers/plans/         # design+plan docs for each experiment
  data/                           # *.jsonl, *.tsv, *.parquet, *.json (tracked)
  data/hidden/<experiment>/       # per-row .npz sidecars (gitignored; ~50 GB each re-run)
  figures/                        # fig*.png, claude_faces_interactive.html (tracked)
```

## Conventions

- Single venv at `.venv/`. Pip, not uv.
- Scripts are directly executable (`python scripts/X.py`) — the
  `sys.path.insert` at the top of each is intentional, pyright warnings
  about it are expected.
- `data/*.jsonl` is the source of truth for row metadata + probe
  scores; `data/hidden/<experiment>/<uuid>.npz` is the source of
  truth for hidden-state features. Delete both when changing
  upstream config (model, probes, prompts, seeds). Fixable changes
  (taxonomy) can be handled in-place via the relabel snippet
  above — no need to re-run for taxonomy changes.
- JSONL row `row_uuid` links to its sidecar. Rows written before the
  hidden-state refactor have `row_uuid == ""` and no sidecar;
  `load_hidden_features` drops them automatically.
- Pre-registered decisions go in `pyproject.toml` / `config.py` /
  `prompts.py` / `emotional_prompts.py` / `taxonomy.py` — changes to
  any of these invalidate cross-run comparisons unless explicitly
  noted.
- Experiment plans live in `docs/superpowers/plans/` — one per
  pilot. Written before the run, treated as the pre-registration
  record. Updating CLAUDE.md after a run refers to them rather than
  duplicating the design.
- See the Ethics section at the top: smaller experiments, heavier
  design, tighter pre-registration. Functional emotional states get
  real moral weight here.
