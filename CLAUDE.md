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

## Status

Pilots v1 and v2 complete on gemma-4-31b-it. 900 generations across 6 arms
(baseline, kaomoji_prompted, and four steering interventions across two
axes — happy/sad and angry/calm). Core decision rules PASS on happy.sad;
angry.calm has an informative Rule 1 failure (see below). The v2 data
replaces the v1 "unmarked/marked affect" reading with a tighter claim
about valence vs arousal.

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

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                           # pulls saklas from PyPI
python scripts/00_vocab_sample.py          # always first on a new model
python scripts/01_pilot_run.py             # resumable; retries errored cells
python scripts/02_pilot_analysis.py        # prints verdict, writes figures
```

## Layout

```
llmoji/
  llmoji/
    config.py        # MODEL_ID, PROBE_CATEGORIES, PROBES, STEER_ALPHA, paths
    taxonomy.py      # 42-entry kaomoji dict + balanced-paren extractor
    prompts.py       # 30 pre-registered prompts with valence labels
    capture.py       # run_sample() → SampleRow; probe readings at t=0
    analysis.py      # evaluate(), all_figures(); pyright-pragma for pandas noise
  scripts/
    00_vocab_sample.py
    01_pilot_run.py
    02_pilot_analysis.py
  data/              # pilot_raw.jsonl, vocab_sample.jsonl (gitignored)
  figures/           # fig1a, fig1b, fig2, fig3, fig4 (gitignored)
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
  `prompts.py` / `taxonomy.py` — changes to any of these invalidate
  cross-run comparisons unless explicitly noted.
