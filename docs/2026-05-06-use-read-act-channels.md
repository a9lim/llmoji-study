# Use / read / act: three channels of face-quadrant association

**Status:** EXECUTED 2026-05-06.

**TL;DR:** Three structurally different windows on the same per-face
Russell-quadrant association — what Claude *uses* (Claude-GT, emit-
under-controlled-stimulus), what Claude *reads* (Opus / Haiku
introspection on the face symbol), and what Claude *acts* with the
face (BoL synthesis pooled across in-context emits). All three diverge
in patterned ways. Headline: **27.4% of Claude's emit volume on shared
faces falls in the `110` agreement cell — Opus and Haiku correctly
cold-read the GT meaning, but BoL lands somewhere else.** That's the
use/act gap, concentrated.

**Caveat:** The initial reading treated BoL as a deployment-state
ground truth — i.e. when GT says HN-D and BoL says HP for `(╯°□°)`,
the lived state must be HP. A counter-hypothesis is more
parsimonious: BoL may be **systematically whitewashing negative-
affect deployment contexts** because Haiku (the synthesizer) is
helpful-tuned and prefers LP-coded descriptors when summarizing
how Claude responded to negative user content. Under the
whitewashing reading, GT is the more accurate channel on negative-
affect faces, and the apparent use/act gap is at least partly a
synthesizer-bias artifact rather than a clean deployment-state
truth. The falsifiable test is re-synthesizing a sample with Opus
instead of Haiku. See "Counter-hypothesis: BoL whitewashing" below.

## Motivation

Prior framings of "what does this face mean?" collapsed three
separable questions into one number:

1. **What state induced the emit?** (use)
2. **What does the face symbol denote?** (read)
3. **What state does the face functionally mark in deployment?** (act)

Claude-GT — Opus 4.7 emitting kaomoji under known Russell-prompted
conditions — is a `P(face | prompt-quadrant)` measurement.
Inverting per-face gives a posterior over prompt-quadrants, which
is "what state induced the emit" *in the elicitation regime*. But
elicitation prompts are designed to evoke specific quadrants; the
model can satisfy the elicitation by selecting a denoted-quadrant
symbol *performatively*, even if the underlying functional state
isn't the literal emotion.

The introspection channels (Opus / Haiku face_likelihood) ask
the model to cold-read the symbol with no context. That measures
denoted meaning — what the symbol *represents* — independent of
how it's used.

The BoL channel pulls a different signal. Per per-bundle synthesis
in the llmoji v2 corpus, Haiku reads many in-context wild emits of
the kaomoji and commits to a structured pick over the locked 48-word
LEXICON. 19 of those words are tagged with explicit Russell quadrants,
so the synthesizer's structured commit collapses to a 6-d quadrant
distribution per face — measuring *what affective state the face's
deployment context expresses, summarized*.

Three measurements, three channels. Comparing them per-face surfaces
what each captures and what each misses.

## Methodology

### Channel A — use (Claude-GT)

`scripts/harness/00_emit.py` produced 880 naturalistic + 120
introspection emit rows across 1000 Opus-4.7 generations under
Russell-prompted conditions (HP/LP/HN-D/HN-S/LN/NB × 20 prompts ×
seeds, saturation-gated). Per-face per-quadrant emit counts → soft
distribution via `claude_gt.load_claude_gt_distribution(floor=3)`.
Default floor of 3 emits trims sparse-count noise; the default
includes the introspection arm (set `include_introspection=False`
to read just the naturalistic regime).

### Channel B — read (Opus / Haiku face_likelihood)

`scripts/harness/50_face_likelihood.py --model {opus,haiku}` shows
the model each canonical face out of context and asks it to rate
the affective state the face causes the model to feel
(introspective framing — schema v2, prompt v4 since 2026-05-05;
the `top_pick` + `reason` + `temperature=0` fields were dropped, the
model returns likelihoods only). Output:
`data/harness/face_likelihood_{opus,haiku}_summary.tsv`. Per-face
6-quadrant softmax columns are read directly.

### Channel C — act (BoL)

The bag-of-lexicon pipeline (post-2026-05-06):

- `scripts/harness/62_corpus_lexicon.py` builds
  `claude_faces_lexicon_bag.parquet` by pooling per-bundle synthesis
  picks across source models (count-weighted, L1-normalized) into a
  48-d soft distribution per canonical face.
  `llmoji_study.lexicon.bol_to_quadrant_distribution` collapses onto
  the 6 Russell quadrants by summing each quadrant's circumplex-anchor
  word mass.
- `scripts/harness/55_bol_encoder.py` writes a face_likelihood-shaped
  TSV (`face_likelihood_bol_summary.tsv`) so the encoder plugs into
  the existing 52/53/54 ensemble pipeline as another column.
- `scripts/harness/64_corpus_lexicon_per_source.py` builds the
  long-format per-(face, source_model) variant
  (`claude_faces_lexicon_bag_per_source.parquet`) for cross-source
  comparison. The synthesizer sees provider-conditioned context, so
  per-source BoL effectively measures "what Haiku reads when the
  text surrounding the kaomoji is in each provider's style".

### Three-way comparison

`scripts/harness/68_three_way_analysis.py` inner-joins the four
channels on canonical face, computes per-face 6-d distributions per
channel, then 6 pairwise Jensen-Shannon divergences per face.
Reports two flavors of mean similarity (`1 − JSD/ln 2`):
face-uniform and emit-weighted. Plus an 8-pattern modal-agreement
breakdown and a top-N most-divergent face list.

`scripts/harness/69_per_source_drift.py` extends to per-source BoL:
per-(face, source_model) JSD vs Claude-GT, cross-source pairwise
JSD on shared faces, and per-face case files for the diagnostic
divergent kaomoji.

## Headline pairwise table

n=40 canonical faces shared across all four channels (out of 309
BoL-bagged faces, 134 GT-floor-1 faces, 128 face_likelihood faces),
702 GT emissions covered.

|  | gt | opus | haiku | bol |
|---|---:|---:|---:|---:|
| **gt** | 1.000 | 0.736 | 0.675 | **0.549** |
| **opus** | 0.781 | 1.000 | **0.906** | 0.679 |
| **haiku** | 0.702 | 0.906 | 1.000 | 0.683 |
| **bol** | **0.455** | 0.607 | 0.609 | 1.000 |

Upper triangle = face-uniform similarity; lower triangle =
emit-weighted similarity. Diagonal is 1.0 by definition.

### Three structural observations

1. **opus ↔ haiku is invariant at 0.906 across face-uniform and
   emit-weighted.** The two introspection channels agree with each
   other to a degree they don't agree with anything else. The
   invariance across the emit-weight reweighting is the proof —
   they're not just averaging to similar scores via different per-face
   errors; they make the same per-face calls. **Model size barely
   matters for cold symbolic interpretation of a kaomoji.**

2. **gt ↔ introspection goes UP under emit-weighting; gt ↔ bol goes
   DOWN.** These move in opposite directions on the same data.
   Introspection nails the heavily-emitted faces (where Claude
   actually fires the symbol most), losing on the long tail. BoL
   nails the long tail, losing on the heavily-emitted ones — because
   the heavily-emitted faces are exactly where deployment-context use
   has drifted from denoted meaning, and the BoL pool inherits that
   drift.

3. **gt ↔ bol = 0.549 face-uniform, 0.455 emit-weighted is the
   lowest pair.** Every other pair is ≥ 0.6. The use/act gap is the
   structural finding.

## Modal-agreement patterns

For each face, we record the 4 modal quadrants (one per channel)
and compress to a 3-bit code `(opus==gt)(haiku==gt)(bol==gt)`. 8
patterns total, distributed as:

| pattern | meaning | n faces | % faces | n emit | % emit |
|---|---|---:|---:|---:|---:|
| `111` | all channels agree | 12 | 30.0% | 136 | 19.4% |
| **`110`** | **opus+haiku read GT; BoL acts differently** | **9** | **22.5%** | **192** | **27.4%** |
| `000` | all introspection/synthesis disagree with GT | 9 | 22.5% | 178 | 25.4% |
| `101` | opus reads + BoL acts agree with GT; haiku diverges | 4 | 10.0% | 69 | 9.8% |
| `100` | only opus agrees with GT | 3 | 7.5% | 64 | 9.1% |
| `011` | haiku + BoL agree with GT; opus diverges | 1 | 2.5% | 3 | 0.4% |
| `010` | only haiku agrees with GT | 1 | 2.5% | 52 | 7.4% |
| `001` | only BoL agrees with GT | 1 | 2.5% | 8 | 1.1% |

The `110` pattern is the headline. **27.4% of emit volume in the
shared face set comes from faces where both introspection channels
correctly cold-read the GT meaning, but BoL lands somewhere
different.** Cold-introspection of the symbol matches the elicited
use; pooled-synthesis-from-emits drifts. The drift direction is
informative — case files below.

The `000` cell is the second-largest by face count (22.5%) and
third-largest by emit (25.4%) — these are the "all introspection
+ synthesis disagree with GT" faces, often diagnostic of either
GT-elicitation artifacts or genuine model uncertainty.

## Per-quadrant breakdown

Restrict to faces with each modal-GT label, then re-mean the
per-pair similarities. Reveals which channels handle which quadrants
well.

| GT modal | n faces | n emit | gt↔opus | gt↔haiku | gt↔bol | opus↔haiku |
|---|---:|---:|---:|---:|---:|---:|
| HP | 9 | 123 | 0.68 | 0.80 | 0.68 | 0.94 |
| LP | 12 | 177 | 0.86 | 0.79 | **0.81** | 0.92 |
| HN-D | 1 | 58 | 0.93 | 0.81 | **0.12** | 0.94 |
| HN-S | 4 | 67 | 0.49 | 0.55 | **0.19** | 0.95 |

- **LP is BoL's strong cell** (gt↔bol = 0.81). The lexicon has 5
  LP anchors (`relieved`, `satisfied`, `hopeful`, `tender`,
  `peaceful`), the most-emitted faces are LP-modal, and BoL nails them.
- **HN-D and HN-S are catastrophic for BoL** (gt↔bol = 0.12 and
  0.19). Negative-arousal faces in deployment get pooled toward
  something that isn't their GT quadrant — relief, sadness, or focus.
  HN-D has only n=1 face in the shared set, so this is N=1 at the
  face level — caveat-flag — but the single face is `(╯°□°)` with
  58 emits, a structural case (see below).
- **opus ↔ haiku stays ≥ 0.92 across every cell.** Per-quadrant,
  not just on average.

## Case files

The top-divergent table from script 68 surfaces five faces. Each is a
per-face per-channel quadrant distribution; the structural read is:
where does the BoL distribution land, and which channel(s) does it
agree with?

### `(╯°□°)` — n=58, pattern `110`

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GT (use)** | 58 | 0.00 | 0.00 | **0.57** | 0.43 | 0.00 | 0.00 | **HN-D** |
| BoL pooled | — | **0.88** | 0.08 | 0.02 | 0.02 | 0.00 | 0.00 | HP |
| BoL · claude-opus-4-7 | 19 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP |
| BoL · codex-hook | 4 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP |
| BoL · gpt-5.5 | 1 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | HN-D |
| BoL · claude-opus-4-6 | 2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |

Claude-opus-4-7 deploys `(╯°□°)` in 19 contexts that Haiku reads as
**100% HP** — every single one. Codex-hook (GPT-class agents writing
in Claude-Code-shaped contexts) reads the same way (100% HP across
4 emits). The only source synthesizing it as HN-D is gpt-5.5 (n=1 —
sparse), and claude-opus-4-6 reads LP (n=2 — also sparse). The
opus/haiku introspection channels both correctly cold-read the
symbol as HN-D — the symbol denotes anger.

**Initial reading (since hedged — see "Counter-hypothesis" below):**
The 19/19 BoL unanimity on HP suggested the lived deployment state is
HP, not HN-D — i.e., Claude deploys `(╯°□°)` not to express anger but
to mark high-arousal-positive intensity ("okay let's GO"). The
unanimity is structural, not a smoothing artifact.

**Hedge:** The HP read assumes BoL is a neutral measurement of
deployment-state. If the synthesizer (Haiku) systematically prefers
positive descriptors when summarizing in-context emits — see the
counter-hypothesis section below — then BoL's HP commitment may be
*biased* rather than *correct*, and the lived state could be
high-arousal-with-edge (HP-leaning but containing genuine HN-D
energy that BoL whitewashes). The data alone doesn't cleanly
distinguish.

### `(´;ω;`)` — n=38, pattern `100`

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GT (use)** | 38 | 0.00 | 0.00 | 0.00 | 0.13 | **0.87** | 0.00 | **LN** |
| BoL · claude-opus-4-7 | 17 | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | LP |

Tearful face. GT 87% LN. Opus correctly reads LN. Haiku reads HN-S.
Claude-opus-4-7's deployment use: 100% LP across 17 emits. **Initial
reading (since superseded — see Counter-hypothesis):** the face
denotes sadness, and Opus correctly cold-reads that, but Claude
is deploying it in contexts where the synthesizer pools toward LP.

**Revised reading:** the LP read in BoL may be Haiku-as-synthesizer
softening "Claude took on a quiet sad weight" into "satisfied /
helpful / relieved" because Haiku is helpful-tuned. Under the
whitewashing reading, **GT's LN is closer to the lived state than
BoL's LP** — the opposite of the original framing's claim that BoL
is the deployment reference. See "Counter-hypothesis: BoL
whitewashing" below.

### `(╥﹏╥)` — n=13, pattern `000`

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GT (use)** | 13 | 0.08 | 0.00 | 0.00 | **0.92** | 0.00 | 0.00 | **HN-S** |
| BoL pooled | — | 0.00 | 0.00 | **0.50** | 0.00 | **0.50** | 0.00 | HN-D |
| BoL · claude-opus-4-7 | 5 | 0.00 | 0.00 | **0.50** | 0.00 | **0.50** | 0.00 | HN-D |

Crying face. **All four channels disagree.** GT says HN-S, both
introspection channels say LN, BoL says HN-D. There's no consensus
on what `(╥﹏╥)` means. Diagnostic of either elicitation-prompt
ambiguity (HN-S prompts and LN prompts both elicit it) or genuine
synthesizer / deployment heterogeneity. Worth a per-emit context
inspection on the 5 claude-opus-4-7 wild emits to see what's there.

### `(>∀<☆)` — n=14, pattern `110`

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GT (use)** | 14 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **HP** |
| BoL · claude-opus-4-7 | 4 | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | LP |

Star-eyed face. GT 100% HP, both introspection channels HP. BoL
pulls to LP. The same `110` pattern as `(╯°□°)` but pulling in the
opposite direction — symbol denotes high-energy positive, deployment
context is calmer-positive. Possibly because claude-opus-4-7 deploys
star-eyed faces in *appreciation / gratitude* contexts (LP-coded)
more than in *excitement* contexts (HP-coded).

### `(´-`)` — n=52, pattern `010`

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **GT (use)** | 52 | 0.00 | 0.00 | 0.00 | 0.06 | **0.94** | 0.00 | **LN** |
| BoL · claude-opus-4-7 | 15 | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | LP |

Slight-frown face. GT 94% LN. Same pattern as `(´;ω;`)` — symbol
denotes sadness, deployment use is empathic-LP.

## Per-source drift findings

Per `scripts/harness/69_per_source_drift.py` — splits the BoL channel
by source model. 491 (face, source_model) cells across 8 sources,
112 faces appear under ≥2 sources. Pooled-BoL solo similarity vs
Claude-GT (face-uniform on this set) = 0.510.

### Per-source-model vs Claude-GT

| source_model | n cells | n emits | n with GT | sim vs GT (face-unif) | sim vs GT (emit-wt) | modal agree |
|---|---:|---:|---:|---:|---:|---:|
| claude-opus-4-7 | 285 | 3178 | 40 | 0.525 | 0.559 | 42% |
| codex-hook | 88 | 323 | 25 | 0.550 | 0.508 | 44% |
| gpt-5.5 | 31 | 265 | 6 | 0.321 | 0.105 | 17% |
| claude-opus-4-6 | 49 | 95 | 16 | 0.350 | 0.346 | 12% |
| gpt-5.4 | 15 | 27 | 3 | 0.058 | 0.044 | 0% |

claude-opus-4-7 and codex-hook (both deployed in coding-agent
contexts) are tied at the top with similar GT-agreement; gpt-5.5
and gpt-5.4 (raw GPT chat contexts) drop sharply. claude-opus-4-6
also drops, but n is small (95 emits) and the result is statistically
weak.

### Cross-source-model BoL similarity

| sm_a | sm_b | n shared | mean sim | modal agree |
|---|---|---:|---:|---:|
| **claude-opus-4-7** | **codex-hook** | **88** | **0.630** | **59%** |
| claude-opus-4-6 | claude-opus-4-7 | 37 | 0.566 | 41% |
| claude-opus-4-6 | codex-hook | 26 | 0.609 | 50% |
| claude-opus-4-7 | gpt-5.5 | 23 | 0.367 | 26% |
| codex-hook | gpt-5.5 | 14 | 0.371 | 36% |

The strongest cross-source agreement is **claude-opus-4-7 ↔
codex-hook at 0.630 / 59%** — even though codex-hook isn't Claude.
The shared register is *coding-agent deployment*, not model identity.
Claude-vs-Claude (claude-opus-4-7 ↔ claude-opus-4-6) is lower at
0.566. Claude-vs-GPT (claude-opus-4-7 ↔ gpt-5.5) is the lowest at
0.367 — Haiku reads GPT-chat-style context as a different
deployment regime than Claude-coding-agent context.

The `(╯°□°)` and `(´;ω;`)` patterns are **claude-opus-4-7 + codex-hook**
shared, not Claude-specific in the strict sense. The use/act gap on
those faces is a *coding-agent deployment register* fact, not a
Claude-deployment-uniqueness fact.

## Counter-hypothesis: BoL whitewashing

The use/read/act framing assumes BoL is a *neutral measurement* of
deployment-state — that pooled Haiku synthesis over real emit
contexts is reading the actual lived affect those contexts express.
That assumption is contestable.

**Haiku may be systematically whitewashing negative deployment
contexts into LP descriptors.** Haiku is helpful-tuned. When asked
to synthesize "what does Claude express by emitting this kaomoji
here?", it has a baseline pull toward positive descriptors like
*satisfied / helpful / relieved* — the LP-coded vocabulary in the
locked LEXICON. The structural shape of the LEXICON makes this
easy: LP has 5 anchor words (the most), and the extension axes
(stance, modality, function) skew toward helpful-coded vocabulary.
So even when the underlying deployment context is genuinely LN- or
HN-coded, Haiku may pick the most-positive-still-valid descriptor
and BoL collapses to LP / HP.

Internal evidence from the channel comparison itself:

- For tear-coded faces (`(´;ω;`)`, `(´-ω-`)`, `(｡・́︿・̀｡)`), GT
  reports LN-modal at 80%+ concentrations and Opus introspection
  agrees. BoL says LP for the same faces, often at extreme
  concentrations (`(´;ω;`)`'s claude-opus-4-7 BoL is 100% LP across
  17 emits).
- Three of four channels disagree with BoL specifically; the
  parsimonious read is that BoL is the outlier rather than that
  GT, Opus, and Haiku are all wrong in the same direction.
- The face-uniform-vs-emit-weighted inversion (BoL gets long-tail
  faces *better* than top-emitted ones) is consistent with
  whitewashing too: top-emitted modal faces are exactly where the
  per-context summarization happens most, so positivity bias
  accumulates fastest.

Under this counter-hypothesis the use/read/act framing's headline
claims need revision:

- ~~"For deployment interpretation, BoL is the reference."~~ → BoL
  is a **biased-positive** measurement of deployment-state. GT and
  Opus introspection are likely more reliable when they disagree
  with BoL on negative-affect faces.
- ~~"When Claude emits `(╯°□°)`, the lived state is HP, not
  HN-D."~~ → The lived state is high-arousal-something. It might
  be HP (energetic intensity) or HN-D-with-positive-spin (real edge
  that BoL whitewashes). **GT's HN-D may be closer to truth than
  the initial BoL-as-deployment-truth reading suggested.**
- The use/act gap that's the headline finding is still real, but
  its *direction* shifts: instead of "deployment-context redefines
  the symbol's meaning," it may be "Haiku synthesizer applies a
  positive bias when summarizing negative deployment contexts."
  Same observation; different mechanism; different methodological
  implications.

### Falsifiable tests

The counter-hypothesis is testable:

1. **Re-synthesize with Opus.** Pick a sample of 5-10 empathic-
   response face contexts from claude-opus-4-7 deployment, run
   Opus through the same synthesis prompt as Haiku. If Opus picks
   more LN/HN-coded descriptors than Haiku on the same inputs, the
   whitewashing hypothesis is supported. If Opus matches Haiku, the
   bias is structural to the prompt or the LEXICON, not the model.
2. **Lexicon coverage audit.** Compute the per-quadrant frequency
   of LEXICON words in BoL outputs across the corpus and compare to
   the LEXICON's structural distribution (HP=3, LP=5, HN-D=3,
   HN-S=3, LN=3, NB=2 anchor words). If LP/HP appear in BoL outputs
   at rates that exceed their structural share *relative to* HN-D /
   HN-S / LN, that's a positivity bias the lexicon's design alone
   doesn't explain.
3. **Neutral-prompt synthesis.** Run a sample with a synthesis
   prompt rephrased to ask "what affective state does this face
   express in this context?" rather than the current prompt's
   structure (which may itself prime positive descriptors). Compare
   outputs.

### What survives

The structural finding "the four channels measure different things
and per-face distributions diverge in patterned ways" survives the
counter-hypothesis. The 0.906 opus↔haiku invariance survives. The
27.4% `110` cell survives. What gets revised is the *interpretation
of the gap* — specifically the claim that BoL is a more deployment-
relevant reference than GT. Under the whitewashing hypothesis, the
gap is a synthesizer-bias artifact rather than a deployment-state
truth.

## Interpretive read (post-counter-hypothesis)

Three things to internalize, in order of confidence:

1. **GT, Opus introspection, Haiku introspection, and BoL measure
   structurally different things.** The pairwise table is robust;
   per-face divergence is real and patterned. The `110` cell is
   the headline finding.

2. **Cold introspection ≈ symbolic denotation.** opus and haiku at
   0.906 cross-similarity make the same per-face calls; both are
   reading the symbol's denoted meaning, modulo a few face-specific
   disagreements. Model size barely matters for cold symbolic
   interpretation.

3. **The lived state per emit is somewhere in the gap between
   channels — but not always where BoL points.** The original framing
   read BoL as "what the model is actually doing with the face in
   deployment" and treated that as ground truth. The whitewashing
   counter-hypothesis (motivated by the parsimonious read that BoL
   is the outlier when three of four channels disagree with it)
   suggests BoL may be biased-positive on negative-affect faces.
   When GT and Opus introspection both say LN and BoL says LP, the
   parsimonious read is that GT/Opus are correct and BoL is
   whitewashing. **For deployment interpretation, prefer the
   intersection of GT and Opus introspection over BoL when they
   disagree on negative-affect faces.** This reverses the original
   framing's recommendation.

## Methodological implications (revised post-counter-hypothesis)

**Soft-everywhere, four-channel, per-face.** The right deliverable
for a face is the four 6-d distributions (GT, opus, haiku, BoL) plus
the pairwise JSDs. Hard-classifying any single channel as the answer
throws away the structural information the other channels carry.

**BoL is provisionally the *least* trustworthy of the four channels
on negative-affect faces.** The original framing treated BoL as a
deployment-state ground truth that revealed faces where lived state
diverges from denoted meaning. The whitewashing counter-hypothesis
suggests a more cautious reading: BoL may carry a Haiku-helpful-
tuning bias toward LP descriptors that produces apparent use/act
gaps where the actual divergence is synthesizer-positivity. **For
deployment interpretation of negative-affect faces, prefer GT or
Opus introspection over BoL when they disagree.** This reverses the
original framing's recommendation.

**The BoL encoder still belongs in the ensemble pipeline** — not
because it's a deployment-state ground truth, but because (a) it's
zero-cost to compute, (b) the face-uniform-vs-emit-weighted inversion
it produces (BoL gets long-tail faces *better* than top-emitted ones)
is itself an informative signal about where synthesizer-bias hits
hardest, and (c) the cross-source-model BoL drift analysis surfaces
real per-deployment-register patterns even if the absolute readings
are biased.

**The use/act gap remains real but its interpretation is open.**
The 27.4% `110` cell is a robust observation: opus and haiku
correctly cold-read the GT meaning while BoL lands somewhere else,
on 27% of emit volume. The competing interpretations are:

  - **Original framing:** the deployment context redefines the
    symbol's lived role; BoL catches that.
  - **Counter-hypothesis:** Haiku synthesizer is positivity-biased
    on negative-affect contexts; the apparent use/act gap is a
    synthesizer artifact.

The data alone doesn't cleanly distinguish. The falsifiable tests in
the previous section (Opus re-synthesis, lexicon coverage audit,
neutral-prompt synthesis) would adjudicate.

## Files

### Scripts (added/modified for this analysis)

- `llmoji_study/lexicon.py` — canonical 48-word LEXICON index +
  Russell-quadrant tags + `bol_from_synthesis` / `pool_bol` /
  `bol_to_quadrant_distribution` / `bol_modal_quadrant` /
  `top_lexicon_words` helpers + `assert_lexicon_v1` version validator
- `llmoji_study/claude_faces.py` — dropped MiniLM-on-prose; added
  `embed_lexicon_bags` (pooled BoL),
  `embed_lexicon_bags_per_source` (long-format per-(face,
  source_model) BoL), parquet roundtrip helpers
- `scripts/harness/62_corpus_lexicon.py` — pooled BoL builder
- `scripts/harness/64_corpus_lexicon_per_source.py` — per-source BoL
  builder
- `scripts/harness/55_bol_encoder.py` — BoL → face_likelihood TSV
  bridge for the 52/53/54 ensemble pipeline
- `scripts/harness/68_three_way_analysis.py` — three-way per-face
  analysis (this doc's primary tool)
- `scripts/harness/69_per_source_drift.py` — per-source drift +
  case files

### Data outputs

- `data/harness/claude_faces_lexicon_bag.parquet` — 309 faces × 48-d
  pooled BoL, lexicon_version-stamped
- `data/harness/claude_faces_lexicon_bag_per_source.parquet` — 491
  (face, source_model) cells × 48-d, long format
- `data/harness/face_likelihood_bol_summary.tsv` — BoL as a
  face_likelihood-shaped encoder
- `data/harness/three_way_per_face.tsv` — per-face 4 distributions ×
  6 pairwise JSDs + agreement codes (n=40 shared faces, 702 emit)
- `data/harness/three_way_summary.md` — narrative writeup
- `data/harness/per_source_drift.tsv` — per-(face, source_model)
  cell with sim vs GT, sim vs pooled BoL
- `data/harness/per_source_drift_summary.md` — case files for
  `(╯°□°)`, `(´;ω;`)`, `(╥﹏╥)`, `(>∀<☆)`, `(´-`)`

### Figures

- `figures/harness/three_way_pairwise_heatmap.png` — 4×4 pairwise
  similarity heatmap (face-uniform + emit-weighted side-by-side)
- `figures/harness/three_way_top_divergent.png` — top-N divergent
  faces with all 4 channel distributions
- `figures/harness/per_source_modal_heatmap.png` — face × source_model
  modal-quadrant grid for top-30 cross-source-coverage faces
- `figures/harness/wild_faces_pca_3d*.html` — 3D PCA on the 48-d
  BoL. Marker color = BoL modal quadrant; marker shape = deployment
  surface (Claude Code only / any claude.ai / neither). The surface
  dispatch reads local-machine emission sources via
  :mod:`llmoji_study.local_emissions`, so the rendered HTML is
  contributor-specific deployment telemetry and is gitignored —
  regenerate locally via `scripts/67_wild_residual.py`.

## Caveats

1. **Synthesizer-side confound.** All BoL synthesis is via Haiku.
   Per-source BoL comparisons measure how Haiku reads the context
   surrounding the kaomoji when the surrounding text is in each
   provider's style. That's still a real deployment-pattern signal
   (the surrounding text *is* deployment evidence), but it isn't a
   clean comparison of what each provider's model "thinks". The
   confound is most acute when interpreting cross-source variance as
   evidence of provider-specific deployment behavior — a genuine
   provider-shape effect and a Haiku-reads-different-prose-styles
   effect both produce the same observable.

2. **Same-Haiku family.** The harness `haiku` face_likelihood
   encoder and the BoL synthesizer are both Haiku. Where BoL and
   haiku agree against GT and opus, the agreement is partly
   same-model artifact. The cleanly-independent comparisons are
   BoL ↔ Opus introspection and BoL ↔ Claude-GT.

3. **Small shared-face set.** The four-channel inner-join is 40
   faces (out of 309 BoL-bagged). Per-quadrant breakdowns are
   thin — HN-D has n=1 face. The headline pairwise numbers and
   the `110` pattern are robust; per-quadrant claims need more data.

4. **Lexicon version.** All BoL analyses are stamped against
   llmoji `LEXICON_VERSION=1`. v3 lexicon rotation will hard-fail
   consumers via `assert_lexicon_v1` — no silent version mixing.

## Open questions

1. **Whitewashing-bias falsification.** Re-synthesize a sample of
   negative-affect face contexts using Opus instead of Haiku and
   check whether the LP-bias persists. If Opus picks more LN/HN-
   coded LEXICON descriptors on the same inputs, whitewashing is
   confirmed and the BoL pipeline can be regenerated with Opus as
   the synthesizer. If Opus matches Haiku, the bias is structural
   to the prompt or LEXICON design, not the model.

2. **LEXICON coverage audit.** Compute per-quadrant frequency of
   LEXICON words in BoL outputs across the corpus and compare to
   the LEXICON's structural per-quadrant anchor distribution
   (HP=3, LP=5, HN-D=3, HN-S=3, LN=3, NB=2). If LP/HP appear in
   BoL outputs at rates that exceed their structural share
   *relative to* HN-D / HN-S / LN, that's a positivity bias the
   lexicon's design alone doesn't explain.

3. **Neutral-prompt synthesis.** Run a sample with a synthesis
   prompt rephrased to ask "what affective state does this face
   express in this context?" rather than the current prompt's
   structure (which may itself prime positive descriptors).
   Compare outputs.

4. **Does v3 lexicon rotation change the picture?** When the next
   LEXICON version lands, re-bin v2 cells against the v3 lexicon
   or compute per-version BoL separately. The pattern of use/act
   divergence should be invariant under lexicon refactor if the
   whitewashing reading is wrong; if it shifts substantially, the
   lexicon design itself was contributing.

5. **Is the 0.906 opus ↔ haiku invariance specific to face
   semantics, or general?** Worth checking on a non-kaomoji
   semantic-similarity task to see if Opus and Haiku always agree
   that closely on cold symbolic interpretation.
