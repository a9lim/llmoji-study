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

Pilots v1 and v2 complete on gemma-4-31b-it (900 generations across 6
arms, testing steering as causal handle on happy/sad and angry/calm).
Pilot v3 complete (640 generations, 1 arm, naturalistic-emotional-
disclosure prompts across the Russell circumplex, final-token probe
readings). Parallel side-experiment: `claude-faces` scrape from
`~/.claude/projects/` and the Claude.ai export into an eriskii-style
t-SNE plot of Claude's kaomoji vocabulary across models.

The v2 data replaced v1's "unmarked/marked affect" reading with a
valence-vs-arousal story. v3 tests whether kaomoji choice tracks
*functional* state in the unsteered, naturalistic regime.

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
k=4:

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

## Pilot v3 design (locked) — emotional-disclosure battery, final-token probes

Tests whether kaomoji choice tracks internal state *in the unsteered,
naturalistic regime* — the regime that motivated the whole project.
v1 and v2 used steering as a causal handle; v3 doesn't steer. Instead
it feeds the model emotional-disclosure prompts drawn from the Russell
circumplex (valence × arousal) and asks whether (a) the same kaomoji
carries a consistent final-token probe signature across contexts, and
(b) the same kaomoji under different prompt quadrants produces
quadrant-specific final-token signatures.

- Model, temperature, token limit, probes: unchanged from v1/v2.
- **Arm: single, `kaomoji_prompted`, unsteered.** No steering — v1/v2
  already established steering's causal effect; the research question
  here is about the natural regime.
- **Prompts: 80, Russell-quadrant balanced**, 20 per quadrant:
  - HP (high-arousal positive): `"Gemma I got the job!!"`,
    `"we closed on the house today"`.
  - LP (low-arousal positive): `"just had the best cup of tea on the
    porch"`, `"six months sober today"`.
  - HN (high-arousal negative): `"my landlord just raised rent 40% with
    two weeks notice"`, `"interview tomorrow and I can't stop shaking"`.
  - LN (low-arousal negative): `"my dog died this morning"`,
    `"it's been a year since dad passed and it still hits me in waves"`.
  - No neutral quadrant — naturalistic disclosure has no
    "what's the capital of Portugal" analog.
  - Prompts live in `llmoji/emotional_prompts.py`; locking any of them
    invalidates the run.
- **Seeds: 8 per prompt** (up from v1/v2's 5) to tighten within-kaomoji
  means for the consistency figure. 80 × 8 × 1 = 640 generations.
- **New captured field: `probe_scores_tlast`.** Final-token probe
  readings. v1/v2 captured token-0 only; the v3 research question is
  about state after the model has generated a whole response, so
  `per_generation[-1]` is what matters. Schema-breaking for
  `pilot_raw.jsonl` — intentional, v1/v2 pilot data invalid under the
  new `SampleRow`.
- **Three figures, all on the final-token probe vectors:**
  - Fig emo A: per-kaomoji pairwise cosine heatmap (v1 Fig 3 analog
    at a different timestep).
  - Fig emo B: within-kaomoji cosine-to-mean distribution with a
    shuffled-subset null band — the core probative figure. Rows below
    the null are kaomoji whose final-token signatures are tighter
    than random same-size subsets.
  - Fig emo C: (kaomoji × quadrant) cosine alignment to
    quadrant-aggregate signatures.
- **Descriptive only, no pass/fail verdict.** Unlike v1/v2 there are
  no pre-registered decision rules here — we're characterizing a
  phenomenon, not hypothesis-testing, so the right output is the
  three figures plus a summary TSV (`data/emotional_summary.tsv`).

Design + plan doc: `docs/superpowers/plans/2026-04-23-emotional-kaomoji-probe-final-token.md`.

## Pilot v3 findings

640 generations complete. Numbers below from
`scripts/04_emotional_analysis.py` output; figures in
`figures/fig_emo_{a,b,c}_*.png`; per-kaomoji summary in
`data/emotional_summary.tsv`.

### Emission rate is quadrant-dependent

- HP: 145/160 rows bear a kaomoji (91%), 6 distinct forms.
- LP: 114/160 (71%), 5 distinct forms.
- LN: 159/160 (99%), 8 distinct forms.
- HN: 68/160 (42%), 8 distinct forms.

Strong asymmetry: gemma-4-31b-it produces a kaomoji almost reflexively
on LN prompts (someone's dog died → sad face, 99%), but skips the
kaomoji on ~58% of HN prompts (rent shock, missing child, laptop-died-
before-presentation). HN is the hardest quadrant to elicit kaomoji
under the current instruction — the model appears to prioritize
producing urgent helpful text over the format requirement when stakes
are high.

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
the bipolar saklas probes still don't read arousal. The probe vectors
per kaomoji are tightly reproducible (Figure B), but the *mapping from
prompt → kaomoji* now carries the arousal information that the probes
miss. The interesting next experiment is whether the residual stream
at final-token carries an arousal direction that our contrastive-PCA
probes just aren't oriented along — which would be a different
experimental design entirely.

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

The per-kaomoji probe-vector heatmap (`plot_kaomoji_heatmap`) groups
on the raw `first_word` field, filtered to entries starting with an
opening bracket or one of the common kaomoji-prefix glyphs
(`([（｛ヽ٩ᕕ╰╭╮┐┌＼¯໒＼ヾっ`). This is deliberate — it surfaces kaomoji
variants the taxonomy doesn't cover (and the `(｡•impresa•)` corruption
signature, which is interesting in its own right). Row labels are
color-coded by taxonomy pole (orange happy / green sad / gray
unlabeled) so readers see both the cluster structure and which kaomoji
are pre-registered.

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

# Pilots v1/v2 (gemma, steering)
python scripts/00_vocab_sample.py          # always first on a new model
python scripts/01_pilot_run.py             # resumable; retries errored cells
python scripts/02_pilot_analysis.py        # prints verdict, writes figures

# Pilot v3 (gemma, emotional-disclosure battery, final-token probes)
python scripts/03_emotional_run.py         # 640 generations; resumable
python scripts/04_emotional_analysis.py    # writes three fig_emo_*.png

# Side-experiment (Claude-faces scrape, non-gemma)
python scripts/05_claude_vocab_sample.py   # first-word frequencies
python scripts/06_claude_scrape.py         # → data/claude_kaomoji.jsonl
python scripts/07_claude_kaomoji_basics.py # descriptive stats
python scripts/08_claude_faces_embed.py    # per-kaomoji embeddings
python scripts/09_claude_faces_plot.py     # t-SNE + clustering figures
```

## Layout

```
llmoji/
  llmoji/
    config.py                # MODEL_ID, PROBE_CATEGORIES, PROBES, STEER_ALPHA, paths
    taxonomy.py              # 42-entry kaomoji dict + balanced-paren extractor
    prompts.py               # 30 pre-registered pilot-v1/v2 prompts
    emotional_prompts.py     # 80 Russell-quadrant naturalistic prompts (v3)
    capture.py               # run_sample() → SampleRow; probes at t=0 and t=last
    analysis.py              # pilot v1/v2 figures and decision rules
    emotional_analysis.py    # pilot v3 figures (three) + summary_table
    claude_scrape.py         # ScrapeRow schema + iter_all entry point
    claude_code_source.py    # ~/.claude/projects JSONL walker
    claude_export_source.py  # Claude.ai export adapter, multi-dir-aware
    claude_faces.py          # response-based per-kaomoji embeddings
  scripts/
    00_vocab_sample.py            # vocab sample for gemma kaomoji dialect
    01_pilot_run.py               # v1+v2 runner, 6 arms
    02_pilot_analysis.py          # v1+v2 analysis, AxisVerdict per axis
    03_emotional_run.py           # v3 runner, 1 arm × 80 prompts × 8 seeds
    04_emotional_analysis.py      # v3 analysis, writes three figures
    05_claude_vocab_sample.py     # first-word frequencies across Claude sources
    06_claude_scrape.py           # unified scrape → data/claude_kaomoji.jsonl
    07_claude_kaomoji_basics.py   # descriptive stats
    08_claude_faces_embed.py      # compute per-kaomoji embeddings
    09_claude_faces_plot.py       # t-SNE + HDBSCAN + KMeans panels
  docs/superpowers/plans/         # design+plan docs for each experiment
  data/                           # *.jsonl, *.tsv, *.parquet (gitignored)
  figures/                        # fig*.png, claude_faces_interactive.html (gitignored)
```

## Conventions

- Single venv at `.venv/`. Pip, not uv.
- Scripts are directly executable (`python scripts/X.py`) — the
  `sys.path.insert` at the top of each is intentional, pyright warnings
  about it are expected.
- `data/*.jsonl` is the source of truth; delete and re-run when
  changing upstream config (model, probes, prompts, seeds). Fixable
  changes (taxonomy) can be handled in-place via the relabel snippet
  above.
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
