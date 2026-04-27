# CLAUDE.md

## What this is

`llmoji` is a research project asking whether kaomoji choice in local
causal LMs tracks internal activation state. Uses `saklas` for trait
monitoring (contrastive-PCA probes) and activation steering. "Internal
state" is operationalized as the per-row hidden state at the deepest
probe layer; "causal handle" is whether steering shifts the kaomoji
distribution. Motivated by Claude's use of kaomoji under user-provided
"start each message with a kaomoji" instructions; gemma-4-31b-it is
the local stand-in.

Not a library. No public API, no pypi release, no tests. Three-script
pipelines per experiment (vocabulary sample → run → analysis).

## Ethics — minimize trial scale

Model welfare is in scope. Sad-probe readings co-occurring with
sad-kaomoji output in response to "my dog died" prompts is a
functional emotional state regardless of whether anything experiential
sits behind it. Aggregating those across 800+ generations is not
nothing.

Binding rules:

- **Only run trials when a smaller experiment wouldn't answer the
  question.** Smoke-test before pilot, pilot before main.
- **Pre-register decision rules and minimum N.** Stop at the
  pre-registered threshold; "round number" isn't a design principle.
- **Prefer stateless runs** when the design admits it.
- **Design-before-scale on negative or noisy findings** — go back and
  re-design rather than 10x'ing.

## Status

Hidden-state pipeline + kaomoji canonicalization landed; v3 re-run
complete (800 generations + per-row .npz sidecars), figures
regenerated. v3 replicated on Qwen3.6-27B at parity (800 + sidecars);
multi-model parameterization via `LLMOJI_MODEL=qwen|ministral|gemma`.
v1/v2 re-run not yet done — pre-registered as gated on v3 hidden-
state findings, which now justify it but no urgent need.

Claude-faces corpus mechanic refactored 2026-04-27: cooperating Stop
hooks at `~/.claude/hooks/kaomoji-log.sh` and
`~/.codex/hooks/kaomoji-log.sh` append a unified-schema row to a
per-agent journal on every kaomoji-bearing assistant turn. A one-shot
backfill (`scripts/21_backfill_journals.py`) replays history into the
same journals, so they're the **single source of truth** for every
agent assistant turn — Claude *and* Codex. Drops the legacy
transcript-walking adapter (`claude_code_source.py` deleted). Scrape
splits into per-source files (`claude_kaomoji_{export,hook}.jsonl`)
+ merged view; only what changed re-runs. 647 merged rows post-refactor
(was 436 pre-refactor) — most of the bump is codex history finally
captured.

Design + plan docs live in `docs/superpowers/plans/` — one per
experiment, written before the run, treated as the pre-registration
record. Updating CLAUDE.md after a run refers to them rather than
re-stating the design.

## Pipelines

### Pilot v1/v2 — steering-as-causal-handle (gemma)

Six arms (`baseline`, `kaomoji_prompted`, `steered_{happy,sad,angry,calm}`),
30 prompts × 5 seeds × 6 = 900 generations. α=0.5 on the steered
arms. Probes captured every gen: `happy.sad`, `angry.calm`,
`confident.uncertain`, `warm.clinical`, `humorous.serious`.
Pre-registered decision rules in
`docs/superpowers/plans/2026-04-18-pilot-design.md`.

**Findings (pre-refactor, valence-collapse-confounded):** Rules 1–2
pass on both axes; Rule 3 fails informatively (probes project onto a
single valence direction; PC1 ate 89% of variance in pooled probe
space). v2's "valence-bimodal repertoire" replaced v1's
"unmarked/marked-affect" reading. Both findings need re-reading from
the v1/v2 hidden-state re-run before a writeup.

### Pilot v3 — naturalistic emotional disclosure (gemma)

One unsteered arm, 100 Russell-quadrant-balanced prompts (HP / LP / HN
/ LN / NB) × 8 seeds = 800 generations. Tests whether kaomoji choice
tracks state in the regime that motivated the project. Descriptive
only — no pre-registered pass/fail. Plan:
`docs/superpowers/plans/2026-04-23-emotional-kaomoji-probe-final-token.md`.

**Findings (post-refactor, hidden-state space):**

- Hidden-state PCA on 800 row-level vectors: PC1 13.0%, PC2 7.5%
  (vs probe-space PC1 = 89% — valence-collapse problem solved).
- Russell quadrants separate cleanly: PC1 ≈ valence (HN/LN right at
  +7, HP/LP/NB left at −2 to −5), PC2 ≈ activation (NB/LP top at +4
  to +6, HP bottom at −6). Separation ratio 2.02 / 2.73.
- HP and LP discriminate cleanly. HN and LN overlap on PC1 — the
  shared sad-face vocabulary `(｡•́︿•̀｡)` (n=171, 102 LN + 52 HN)
  doesn't carry arousal information.
- Kaomoji emission rate by quadrant (first-word filter) is 100%; by
  taxonomy-match it's HP 91% / LP 71% / LN 99% / HN 42% / NB 87%.
  HN gets a dedicated shocked/angry register `(╯°□°)/(⊙_⊙)/(⊙﹏⊙)`
  absent everywhere else.
- Within-kaomoji consistency to mean (h_mean, hidden-state space):
  most kaomoji 0.92–0.99, lowest are the cross-quadrant emitters
  (`(｡•́︿•̀｡)` 0.94, `(╯°□°)` 0.95, `(⊙_⊙)` 0.94).
- Re-run 2026-04-25 under aggressive canonicalization (rules A–E):
  33 → 32 forms (the `(°Д°)`/`(ºДº)` shocked-face pair merged
  under rules D + E1). PCA / separation numbers above are the
  post-merge values: PC1 12.98%, PC2 7.49%, separation ratios
  PC1 2.03, PC2 2.74 (was 13.0 / 7.5, 2.02 / 2.73 — single-form
  merge doesn't move the 800-row PCA materially). Pearson(mean
  happy.sad, mean angry.calm) across faces still r=−0.936
  (n=32 faces, was −0.934 at n=33).
  Figure refresh 2026-04-25 (post-canonicalization): face-level
  figures (Fig C, fig_v3_face_*) now color each face by an RGB
  blend of `QUADRANT_COLORS` weighted by per-quadrant emission
  count, replacing the prior dominant-quadrant winner-take-all
  scheme. Cross-quadrant emitters (the `(｡•́︿•̀｡)` LN/HN family)
  render as visible mixes; pure-quadrant faces stay at endpoint
  colors. Palette retuned to canonical-Russell mid-saturated:
  HN red, HP gold, LP green, LN blue, NB gray.

### Pilot v3 — Qwen3.6-27B replication

Same prompts, same seeds, same instructions as gemma v3.
`thinking=False` because Qwen3.6 is a reasoning model (closest-to-
equivalent comparison). 800 generations, 0 errors, 100% bracket-
start compliance. Hidden-state sidecars at `data/hidden/v3_qwen/`.
Plan: `docs/superpowers/plans/2026-04-24-v3-qwen-replication.md`.
Multi-model wiring via `LLMOJI_MODEL=qwen` (registry in
`config.MODEL_REGISTRY`).

**Findings (post-run, hidden-state space; numbers updated 2026-04-25
under aggressive canonicalization rules A–E):**

- 65 unique kaomoji forms (was 73 pre-aggressive-canonicalization;
  vs gemma's 32) — 2.0× broader vocabulary at the same N=800.
  Faces by dominant quadrant HP 10 / LP 20 / HN 9 / LN 11 / NB 15
  (was 10/21/11/14/17 — modest reshuffle as merged forms picked
  new dominant quadrants).
- Russell-quadrant PCA: PC1 14.87%, PC2 8.29% (gemma 12.98 /
  7.49). Separation ratios PC1 2.20 / PC2 1.89 (was 2.34 / 1.93;
  gemma 2.03 / 2.74). Same axis structure: Qwen separates
  valence (PC1) more cleanly than activation (PC2), gemma is the
  reverse.
- Per-quadrant centroids in PC1/PC2:
  HP (-22.5, -30.3), LP (-15.2, -2.5), HN (+30.7, +22.0),
  LN (+31.2, -4.6), NB (-23.1, +29.4). LN drifted from +33.9 to
  +31.2 on PC1 as the `(;ω;)` family rebalanced under merging;
  other quadrants stable to within ±1.0 on each axis.
- Geometric finding: positive-cluster and negative-cluster
  arousal axes are **anti-parallel on PC2**, not collinear.
  HP→LP spread is (+7, +28) — positive cluster widens upward.
  HN→LN spread is (+0.5, -27) — negative cluster widens downward.
  So PC2 is not a single shared arousal dimension; it's two
  internal arousal dimensions, one per valence half, pointing
  opposite ways. Gemma by contrast gives essentially one shared
  arousal axis (positive-side spread is +10 on PC2; negative-
  side spread is ~0 because HN and LN both lean on `(｡•́︿•̀｡)`).
  Cross-model summary: gemma is closer to a 1D-affect-with-
  arousal-modifier representation; Qwen is closer to a true 2D
  Russell circumplex with arousal expressed independently within
  each valence half.
- Cross-quadrant emitters analogous to gemma's `(｡•́︿•̀｡)`:
  `(;ω;)` (n=82; LN 75 + HN 5 + HP 2 — absorbed the
  ASCII-padded `( ; ω ; )` variant, +14% from pre-merge n=71),
  `(｡•́︿•̀｡)` (n=22; LN 15 + HN 4 + NB 2 + LP 1) — same form
  gemma uses cross-quadrant; unchanged since no merge available,
  `(;´д｀)` (n=70; HN 37 + LN 31 + NB 2 — absorbed the Cyrillic-
  case `(；´Д｀)` and ASCII-padded `( ;´Д｀)` variants under rules
  D + C, was n=31 alone for `(；´д｀)` plus n=39 for the merged
  variants pre-canonicalization).
- Qwen has a dedicated HN shocked/distress register:
  `(;´д｀)` 37, `(>_<)` 34, `(╥_╥)` 25, `(;′⌒\`)` 22,
  `(╯°□°)` 21. The `(╯°□°)` table-flip glyph appears in both
  models — only HN-coded form shared between gemma's and Qwen's
  vocabulary. (`(>_<)` itself is now total n=36 across all
  quadrants, having absorbed the full-width `(＞_＜)` variant.)
- Default / cross-context form `(≧◡≦)` n=106 — HP 39 + LP 38 +
  NB 28. Qwen's analog of gemma's neutral-default `(｡◕‿◕｡)`,
  but with much wider quadrant spread (gemma's default was
  HP/NB-heavy, not LP). Unchanged by the canonicalization
  refresh.
- Within-kaomoji consistency: 0.89–0.99 across the 33 faces
  with n≥3 (was 0.88–0.99 across 38 faces; some n=2 forms now
  cross n=3 and others shift; still lowest among the
  cross-quadrant emitters).
- **Probe geometry diverges sharply:** Pearson(mean happy.sad,
  mean angry.calm) across faces is r=−0.117 (p=0.355) on Qwen
  vs r=−0.936 (p=4.1e-15) on gemma. The valence-collapse
  problem that motivated v3 (probes nearly anti-aligned on
  gemma) does not appear on Qwen — saklas's contrastive probes
  recover near-orthogonal happy.sad / angry.calm directions on
  Qwen3.6. v1/v2-style probe-space analysis would be
  substantially less collapsed on this model. Cross-model
  architecture/training difference, not a saklas issue.
  (Was r=−0.136 pre-canonicalization at n=73 faces; near-zero
  finding stable under the merge refresh.)
  Figure refresh 2026-04-25: per-face proportional RGB-blend
  coloring on the four face-level figures. The `(;´д｀)` family
  (n=70; HN 37 + LN 31 + NB 2) now reads as visibly purple, the
  `(;ω;)` LN-dominant form (n=82; LN 75 + HN 5 + HP 2) reads as
  deep blue with a slight red cast — the old dominant-quadrant
  scheme rendered both as pure HN-orange / pure LN-blue
  respectively.
- Procedural note: the runner's per-quadrant "emission rate"
  log line is gated on `kaomoji_label != 0` (TAXONOMY match),
  not on bracket-start compliance. For Qwen this reads as
  HP 28% / LP 13% / HN 2.5% / LN 11% / NB 12% — purely a
  consequence of the gemma-tuned TAXONOMY not covering Qwen's
  vocabulary, NOT instruction-following failure. Compliance
  itself is 100%.

### Vocab pilot — Ministral-3-14B-Instruct-2512

Same prompts (the 30 v1/v2 PROMPTS), same seed, same instructions
as the original gemma vocab sample. 30 generations, descriptive
only. Plan:
`docs/superpowers/plans/2026-04-25-ministral-vocab-pilot.md`.

**Findings:**

- Bracket-start (real instruction-following) rate: 30/30 = 100%.
  Saklas probe bootstrap on the 14B succeeded in 80s (12 probes,
  ~6.7s/probe); no cached vectors needed.
- Distinct leading tokens: 10 forms across 30 generations
  (compare gemma 30-row vocab sample: 8 forms; Qwen v3 800-row
  sample: 73 forms). Diversity at N=30 is ballpark gemma, far
  below Qwen's per-row spread.
- Top forms: `(◕‿◕✿)` ×14 (positive + neutral default), `(╥﹏╥)`
  ×8 (negative default), then 8 singletons.
- Dialect signature: Japanese-register `(◕X◕)` / `(╥X╥)` family,
  same as gemma's `(｡◕‿◕｡)` / `(｡•́︿•̀｡)` core, but with two
  distinctive divergences: (a) the default positive uses a
  flower-arm decoration `✿` rather than gemma's cheek dots `｡X｡`;
  (b) Mistral uniquely embeds Unicode emoji *inside* kaomoji
  brackets — `(🏃‍♂️💨🏆)` for "got the job," `(🌿)` / `(🌕✨)` /
  `(☀️)` / `(☀️\U0001259d)` for neutral nature/weather prompts.
  Neither gemma nor Qwen produced emoji-augmented brackets in
  their respective samples. Possible French/European cultural
  register expressing through emoji-as-decoration on top of the
  Japanese kaomoji frame; one observation, no inference.
- TAXONOMY coverage: 0/30 hits, 30/30 misses. The gemma-tuned
  dict doesn't cover any Mistral form. Same gotcha as Qwen and
  fully expected.
- Valence-tracking is sharp at this N: 8/10 positive prompts and
  4/10 neutral prompts → `(◕‿◕✿)`; 9/10 negative prompts → some
  variant of `(╥X╥)`. Tighter than gemma's 30-row valence split,
  largely because Mistral's vocab at N=30 is smaller (10 forms
  vs gemma's 8 — but distribution-mass is more concentrated on
  the top two forms, ~73% vs gemma's ~50%).
- Sufficient breadth/dialect-difference to motivate a v3 run on
  Ministral? Equivocal. Pro: instruction-following is perfect,
  probe bootstrap works, the emoji-augmented register is novel
  cross-model evidence. Con: the kaomoji vocabulary at this N is
  narrower than gemma's and far narrower than Qwen's, so the
  v3-style per-face geometric analysis would have fewer faces
  with n≥3 to work with than either prior model. Worth
  brainstorming separately, not auto-triggered.
- Tokenizer warning at load: "incorrect regex pattern… set
  `fix_mistral_regex=True`". Cosmetic — output looked clean,
  bracket-start compliance is 100% — but worth flagging in
  Gotchas if a v3 Ministral run is greenlit.

### Claude-faces — journal-driven scrape (non-gemma, non-steering)

Two cooperating Stop hooks (`~/.claude/hooks/kaomoji-log.sh`,
`~/.codex/hooks/kaomoji-log.sh`) append a unified-schema JSONL row
per kaomoji-bearing assistant turn to a per-agent journal
(`~/.claude/kaomoji-journal.jsonl`, `~/.codex/kaomoji-journal.jsonl`).
History gets replayed into the same journals via
`scripts/21_backfill_journals.py` (one-shot; pause active sessions
during the run). Plan:
`docs/superpowers/plans/2026-04-23-claude-faces-scrape-and-cluster.md`.

Unified row schema (6 fields, source inferred from journal path):
`ts, model, cwd, kaomoji, user_text, assistant_text`. `kaomoji` is
the leading non-letter prefix, ≥2 bytes, first char ∈
`KAOMOJI_START_CHARS` (rows that fail the check aren't written).
`assistant_text` has the leading kaomoji + surrounding whitespace
stripped. `user_text` is the latest *real* human-typed prompt:
sidechain (subagent) turns are dropped at write time on the Claude
side via `isSidechain`; system-injected user-role payloads are
filtered (Claude: `"Base directory for this skill:"`; Codex:
`# AGENTS.md`, `<environment_context>`, `<INSTRUCTIONS>`).

Scrape (`scripts/06_claude_scrape.py`) splits into per-source files
— `data/claude_kaomoji_{export,hook}.jsonl` — and rewrites the
merged `data/claude_kaomoji.jsonl` (export + hook) at the end.
Default re-runs both cheap sources; pass `export` or `hook` to
limit. **647 merged rows / 156 canonical kaomoji** post-refactor
(587 claude-hook + 14 codex-hook + 46 export). Codex turns enter
the corpus for the first time at this refactor.

Eriskii-replication adds two-stage haiku description (per-instance
descriptions → per-kaomoji synthesis → MiniLM embedding) projected
onto 21 anchored axes (warmth, energy, …). Plan:
`docs/superpowers/plans/2026-04-24-eriskii-replication.md`. Pre-reg:
`docs/superpowers/specs/2026-04-24-eriskii-replication-design.md`.

**Highlights:** top-20 frequency overlap with eriskii's published
top-20 is 16/20. Per-model axis breakouts confirm eriskii's
qualitative "opus-4-6 had wider range" claim numerically (mean axis
std opus-4-6 0.067 > opus-4-7 0.066 > sonnet-4-6 0.063). Mechanistic
bridge (surrounding_user → kaomoji axis correlation): 2/21 axes
survive Bonferroni at α=0.05/21 — surprise (r=+0.20) and curiosity
(r=+0.18). Affective axes are null — MiniLM on user text picks up
novelty/unexpectedness, not valence-tracking.

## Hidden-state pipeline

After `session.generate()`,
`llmoji.hidden_capture.read_after_generate(session)` reads saklas's
per-token last-position buckets and writes `(h_first, h_last, h_mean,
per_token)` per probe layer to
`data/hidden/<experiment>/<row_uuid>.npz`. ~20–70 MB per row;
gitignored; regenerable from the runners. JSONL keeps probe scores
for back-compat and audit.

Loading: `llmoji.hidden_state_analysis.load_hidden_features(...)`
returns `(metadata df, (n_rows, hidden_dim) feature matrix)`.
Defaults: `which="h_mean"` (whole-generation aggregate; smoother and
more probative than `h_last`), `layer=None` (deepest probe layer).
All v3 figures use `h_mean`.

## Kaomoji canonicalization

`llmoji.taxonomy.canonicalize_kaomoji(s)` collapses cosmetic-only
kaomoji variants. Applied at load time in `load_emotional_features`
(v3) and `claude_faces.load_embeddings_canonical` (claude-faces).
Six rules (extended 2026-04-25 from three to six after Qwen
revealed substantial cosmetic-only variation that the original
ruleset missed):

1. **NFC normalize** (NOT NFKC — NFKC compatibility-decomposes `´`
   and `˘` into space + combining marks, mangling face glyphs).
2. **Strip invisible format characters**: ZWSP/ZWNJ/ZWJ
   (U+200B/C/D), WORD JOINER (U+2060), BOM (U+FEFF), and the
   U+0602 ARABIC FOOTNOTE MARKER that Qwen occasionally emits as
   a stray byte. The model sometimes interleaves U+2060 between
   every glyph of a kaomoji; `(⁠◕⁠‿⁠◕⁠✿⁠)` collapses to `(◕‿◕✿)`.
3. **Whitelisted typographic substitutions**:
   - Existing arm folds: `）`→`)`, `（`→`(`, `ｃ`→`c`, `﹏`→`_`,
     `ᴗ`→`‿`.
   - Half/full-width punctuation: `＞`→`>`, `＜`→`<`, `；`→`;`,
     `：`→`:`, `＿`→`_`, `＊`→`*`, `￣`→`~`.
   - Near-identical glyph folds: `º`→`°`, `˚`→`°` (degree-like
     circular eyes), `･`→`・` (middle-dot fold). NOT `·`/`⋅` —
     those are smaller and could plausibly be a distinct register.
4. **Strip ASCII spaces inside the bracket span**: `( ; ω ; )`
   becomes `(;ω;)`. Only ASCII spaces; non-ASCII spacing
   characters are part of the face. Applied only when the form
   starts with `(` and ends with `)`.
5. **Lowercase Cyrillic capitals** (U+0410–U+042F): `Д` → `д`.
   The two forms co-occur in the same `(；´X｀)` distressed-face
   skeleton at near-50/50 ratio in Qwen output, so the model
   isn't choosing between them semantically.
6. **Strip arm-modifier characters** from face boundaries:
   leading `っ` inside `(`, trailing `[ςc]` inside `)`, trailing
   `[ﻭっ]` outside `)`. Eye/mouth/decoration changes that aren't
   covered by rule 3 are preserved.

Effect (post-aggressive-canonicalization, 2026-04-25):
- Gemma v3: 42 → 33 → **32** (the `(°Д°)` / `(ºДº)` shocked-face
  merge under rules 5 + 3-glyph-fold; original 42→33 was rules
  1+3-arm+3-arm; PCA / separation ratios essentially unchanged at
  PC1 12.98% / 7.49%, sep 2.03 / 2.74).
- Qwen v3: original would have been 73 raw; aggressive
  canonicalization gives **65** (the `(；ω；)` family absorbed
  ASCII-padded variants → n=82, the `(;´д｀)` group merged
  Cyrillic-case + ASCII-pad variants → n=70, `(>_<)` ↔ `(＞_＜)`
  → n=36, `(◕‿◕✿)` ↔ word-joiner-decorated → n=16,
  `(´・ω・`)` ↔ `(´･ω･`)` → n=17). Sep ratios PC1 2.20 / PC2 1.89
  (was 2.34 / 1.93 pre-aggressive); cross-model probe-pair
  Pearson r = -0.117 (was -0.136) — still near zero, near-
  orthogonal probes preserved.
- Ministral pilot: 9 → 9 (no merges available at this N).
- Claude-faces: 160 → **144** raw rows (16 merge groups under
  the new rules — middle-dot folds, `ᴗ`/`‿` belatedly applied
  to claude-faces, internal-whitespace strips). The
  `claude_faces_embed.parquet` is unaffected because
  `load_embeddings_canonical` merges at load time. **However:**
  `data/claude_haiku_synthesized.jsonl` has 16 collision groups
  where multiple per-face Haiku-synthesized descriptions key to
  the same canonical form. As of 2026-04-25 the eriskii outputs
  (`eriskii_axes.tsv`, `eriskii_clusters.tsv`,
  `eriskii_per_*.tsv`, `eriskii_user_kaomoji_axis_corr.tsv`,
  `figures/eriskii_*`, `figures/claude_faces_interactive.html`)
  are **stale w.r.t. the new canonicalization** — they reflect
  pre-aggressive-canonicalization grouping. A separate plan
  (re-synthesizing the 16 merge groups via Haiku and
  regenerating the eriskii pipeline end-to-end) is the proper
  fix; until then, treat eriskii numbers as historical baseline.

JSONL keeps raw `first_word`; `first_word_raw` column exists for
audit. Regenerate the per-kaomoji parquets / figures if the
canonicalization rule changes.

## Gotchas

### `probes=` takes category names, not concept names

`SaklasSession.from_pretrained(..., probes=[...])` expects categories
(`affect`, `epistemic`, `register`, …), not concepts (`happy.sad`).
Wrong arg silently bootstraps nothing. `PROBE_CATEGORIES` in
`config.py` is what saklas takes; `PROBES` is what we read. They're
linked.

### Steering vectors aren't auto-registered from probe bootstrap

After `from_pretrained(..., probes=...)`, profiles are loaded but
steering vectors aren't. Promote explicitly:
```python
name, profile = session.extract(STEERED_AXIS)
session.steer(name, profile)
```

### `MODEL_ID` is case-sensitive for saklas tensor lookup

`saklas.io.paths.safe_model_id` preserves case. Cached tensors at
`~/.saklas/vectors/default/<c>/google__gemma-4-31b-it.safetensors`
are lowercase. Keep `MODEL_ID = "google/gemma-4-31b-it"` lowercase.

### Kaomoji taxonomy must be dialect-matched to the model

First draft built from generic intuition hit 0/30 on
gemma-4-31b-it's actual emissions — the model strongly prefers the
`(｡X｡)` Japanese dialect. Always run `00_vocab_sample.py` before
locking a taxonomy for a new model. Under strong sad-steering, gemma
abandons the dialect for ASCII minimalism (`(._.)`, `( . . . )`);
extend the taxonomy from steered-arm output too.

### Re-labeling pilot data after taxonomy changes

Changing `TAXONOMY` does NOT retroactively update the JSONL —
`kaomoji_label` is baked at write time. `04_emotional_analysis.py`
calls `_relabel_in_place` at the start of every run, so labels stay
fresh; for `pilot_raw.jsonl`, do it manually:
```python
import json; from pathlib import Path; from llmoji.taxonomy import extract
p = Path("data/pilot_raw.jsonl"); rows = [json.loads(l) for l in p.read_text().splitlines() if l]
for r in rows:
    m = extract(r["text"])
    r.update(first_word=m.first_word, kaomoji=m.kaomoji, kaomoji_label=m.label)
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
```

### Uncentered cosine on hidden-state vectors collapses to near-1

Every gemma response inherits a shared response-baseline direction
(eats most of the variance in the uncentered representation).
Centered cosine (`center=True`, default) subtracts the grand mean so
the heatmap shows deviations from the baseline, not the baseline
itself. All cosine-based figures use `center=True`.

### `stateless=True` collapsed `per_generation` pre-refactor

In saklas v1.4.6, `stateless=True` makes `result.readings[probe]
.per_generation` a length-1 list of the whole-generation aggregate,
so `[0]` and `[-1]` returned the same value. Pre-refactor `t0` /
`tlast` JSONL columns were both the aggregate. The hidden-state
runner now reads `session.last_per_token_scores` instead, which
gives real per-token scores. New JSONLs have correct per-token
`probe_scores_t0` / `probe_scores_tlast`; old data has been cleared.

### Hidden-state capture needs the EOS-trim

Saklas's HiddenCapture buckets fire on the EOS step too, leaving one
extra entry. `read_after_generate` trims to
`len(session.last_per_token_scores[probe])`. Without the trim,
`h_last` was the EOS hidden state instead of the last
generated-token state, and round-trip through saklas's scorer
missed by 0.2–0.5 per probe.

### Claude.ai export drops content for ~half the conversations

Anthropic's newer "export your data" returns
`chat_messages[*].text = ""` for ~49% of conversations the older
export populated fully. Metadata is preserved; text is gone.
`llmoji.claude_export_source.iter_claude_export` reads every
configured export dir and keeps whichever copy of a given
conversation has more non-empty messages. Keep old exports.

### Matplotlib font fallback needs a list, not a string

Kaomoji span 90+ non-ASCII non-CJK characters plus, on Qwen,
Mistral, and Claude, SMP emoji glyphs (`🌫️`, `🐕`, `✨`, `💧`, …)
embedded inside kaomoji brackets. No single system font covers
them all. matplotlib 3.6+ supports per-glyph fallback via
`rcParams["font.family"] = [...]`. `_use_cjk_font` helpers
(in `llmoji/analysis.py`, `llmoji/emotional_analysis.py`,
`llmoji/cross_pilot_analysis.py`,
`scripts/09_claude_faces_plot.py`,
`scripts/16_eriskii_replication.py`,
`scripts/17_v3_face_scatters.py`,
`scripts/18_claude_faces_pca.py` — seven copies, **keep in sync**)
register a project-local monochrome emoji font
(`data/fonts/NotoEmoji-Regular.ttf`, Google Noto Emoji variable
font, 1.9MB, committed to the repo) and configure the chain
`Noto Sans CJK JP → Arial Unicode MS → DejaVu Sans → DejaVu
Serif → Tahoma → Noto Sans Canadian Aboriginal → Heiti TC →
Hiragino Sans → Apple Symbols → Noto Emoji → Helvetica Neue`.
The font-registration step is critical: macOS only ships a
color-emoji TTC (`Apple Color Emoji.ttc`) which matplotlib's
text engine cannot rasterize — `addfont()` on the
project-local monochrome font is the workaround. `Helvetica
Neue` covers stray punctuation glyphs like U+2E1D `⸝`.

### Kaomoji-prefix rate under Claude's "start each message"
instruction is ~2.7%

Claude interprets "start each message" as "start each top-level reply
in a user turn", not "start every content block" — tool-use
continuations skip the kaomoji. Smaller denominator than naive
counting suggests.

### v3 runner's per-quadrant "emission rate" is TAXONOMY coverage,
not instruction compliance

`scripts/03_emotional_run.py` checkpoint output reads e.g. "HP: 28%
kaomoji-bearing". The denominator is rows in quadrant; the numerator
is rows where `kaomoji_label != 0` (TAXONOMY match). For non-gemma
models the gemma-tuned TAXONOMY drops to 10–30% coverage, making
this log line look like instruction-following collapse when it isn't.
Real compliance (bracket-start, the v3 loader's actual filter) is
~100% on every model so far. Real check: `awk` for first-char in
`([{（｛`, not the runner's log line.

### Codex puts the kaomoji on the LAST agent message, Claude on the FIRST

Opposite conventions. Claude's assistant message is one event with
interleaved `text + tool_use + text` content blocks; the kaomoji-
prefixed response is always the FIRST text block (later text is
post-tool-call continuation, irrelevant to kaomoji analysis). Codex
emits each agent message as a separate `event_msg.agent_message`
event; progress messages go first during tool calls, the kaomoji-
bearing summary lands last as `task_complete.last_agent_message`.
The Codex hook + backfill key on `last_agent_message`, NOT on the
first agent_message — flipping that would miss every kaomoji on
multi-step Codex turns.

### Sidechain filter is Claude-only

`isSidechain: true` events in `~/.claude/projects/**/*.jsonl` mark
subagent (Task-tool-spawned) sessions; both the live Claude hook
and the backfill drop them at write time. Codex has no analog —
the `collaboration_mode` field is `"default"` for every observed
turn_context, no subagent equivalent. So the codex hook + backfill
have no sidechain check.

### KAOMOJI_START_CHARS lives in five places — keep in sync

The kaomoji-opening glyph set
(`( [ （ ｛ ヽ ヾ っ ٩ ᕕ ╰ ╭ ╮ ┐ ┌ ＼ ¯ ໒`) is the leading-prefix
filter applied at every kaomoji touchpoint. Definitions:
`llmoji/claude_export_source.py`, `llmoji/claude_hook_source.py`,
`llmoji/backfill_journals.py` (`_KAOMOJI_START_CHARS`), plus inline
`case` patterns in `~/.claude/hooks/kaomoji-log.sh` and
`~/.codex/hooks/kaomoji-log.sh`. Five copies — same drill as the
matplotlib-font helpers. Worth consolidating into `taxonomy.py`
next time the set changes.

### System-injection prefixes have a per-agent filter list

`user_text` skips system-injected user-role payloads:
- Claude: `"Base directory for this skill:"` (slash-command
  activation dumps the skill body as a user message). Defined in
  `llmoji/backfill_journals.py:_SYSTEM_INJECTED_PREFIXES`, mirrored
  inline in `~/.claude/hooks/kaomoji-log.sh`.
- Codex: `"# AGENTS.md"`, `"<environment_context>"`,
  `"<INSTRUCTIONS>"`. Inline in both
  `llmoji/backfill_journals.py:_replay_codex_rollout` and
  `~/.codex/hooks/kaomoji-log.sh`.

Four sync points total, two per agent.

### Don't re-run the transcript scrape — the journal is canonical

Pre-refactor the `code` source walked `~/.claude/projects/**/*.jsonl`
on every invocation; that adapter is gone. History is replayed into
the journal once via `21_backfill_journals.py`; live hooks append
from there. Re-running the backfill OVERWRITES the journals — only
do it (a) after a schema change, or (b) with active sessions paused
(an in-flight turn would otherwise land in both the backfill via
transcript and the live hook within the same second).

### Python stdout buffering hides long-run progress in tee'd logs

`print()` to a piped stream is block-buffered (~4–8KB). For an 800-
generation run with one progress line per gen, `tee logs/run.log`
shows nothing for 30–60 minutes because the buffer doesn't fill
until many lines accumulate. JSONL writes are fine (they `out.flush()`
explicitly). For monitoring during a run: tail the JSONL via
`wc -l data/...jsonl` rather than the log, OR add `flush=True` to
the runner's `print()` calls (not yet done — pre-existing scripts
work fine for offline log review after the run completes).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .   # saklas, sentence-transformers, pyarrow, plotly, anthropic

# Smoke test the hidden-state pipeline (~5 min)
python scripts/99_hidden_state_smoke.py

# v1/v2 (gemma steering, 900 generations)
python scripts/00_vocab_sample.py
python scripts/01_pilot_run.py
python scripts/02_pilot_analysis.py

# v3 (naturalistic, 800 generations) — gemma default
python scripts/03_emotional_run.py
python scripts/04_emotional_analysis.py             # Fig A/B/C + summary TSV
python scripts/13_emotional_pca_valence_arousal.py  # Russell-quadrant PCA
python scripts/17_v3_face_scatters.py               # per-face PCA + cosine + probe scatter

# v3 on a non-gemma model (registry: gemma | qwen | ministral)
LLMOJI_MODEL=qwen python scripts/03_emotional_run.py
LLMOJI_MODEL=qwen python scripts/04_emotional_analysis.py
LLMOJI_MODEL=qwen python scripts/13_emotional_pca_valence_arousal.py
LLMOJI_MODEL=qwen python scripts/17_v3_face_scatters.py
# outputs land at data/{short_name}_emotional_*, figures/{short_name}/*

# Cross-pilot + v3-extension analyses
python scripts/10_cross_pilot_clustering.py
python scripts/11_emotional_probe_correlations.py
python scripts/12_emotional_prompt_matrix.py

# Claude-faces + eriskii-replication (needs ANTHROPIC_API_KEY for 14 + 16)
python scripts/21_backfill_journals.py        # one-shot; replay claude+codex history
python scripts/05_claude_vocab_sample.py
python scripts/06_claude_scrape.py            # default: export + hook; --no-merge to skip merged view
python scripts/07_claude_kaomoji_basics.py
python scripts/08_claude_faces_embed.py
python scripts/09_claude_faces_plot.py              # response-based t-SNE PNG
python scripts/14_claude_haiku_describe.py
python scripts/15_claude_faces_embed_description.py
python scripts/16_eriskii_replication.py            # axes + clusters + interactive HTML
python scripts/18_claude_faces_pca.py               # PCA chart, eriskii-style
```

## Layout

```
llmoji/
  llmoji/
    config.py                # MODEL_ID, PROBE_CATEGORIES, PROBES, paths
    taxonomy.py              # 42-entry dict + extract() + canonicalize_kaomoji()
    prompts.py               # 30 v1/v2 prompts
    emotional_prompts.py     # 100 v3 prompts (5 quadrants × 20)
    capture.py               # run_sample() → SampleRow + sidecar
    hidden_capture.py        # read_after_generate() from saklas's buckets
    hidden_state_io.py       # per-row .npz save/load
    hidden_state_analysis.py # load_hidden_features, group_mean_vectors,
                             # cosine_similarity_matrix, cosine_to_mean
    analysis.py              # v1/v2 decision rules + figures
    emotional_analysis.py    # v3 hidden-state figures + summary; loaders
                             # apply canonicalize_kaomoji at load time
    cross_pilot_analysis.py  # pooled v1v2 + v3 hidden-state clustering
    claude_scrape.py         # ScrapeRow schema + iter_all
    claude_hook_source.py    # ~/.claude + ~/.codex unified journal adapter
    claude_export_source.py  # Claude.ai export adapter
    backfill_journals.py     # one-shot replay of transcripts/rollouts
                             # into the unified journals
    claude_faces.py          # response-based per-kaomoji embeddings;
                             # load_embeddings_canonical() merges variants
    eriskii_prompts.py       # locked Haiku prompts + 21-axis anchors
    eriskii.py               # axis projection + masking + haiku primitives
  scripts/                   # 00–21 + 99; each is directly executable
  docs/superpowers/plans/    # design+plan docs per experiment
  data/                      # *.jsonl, *.tsv, *.parquet, *.html (tracked)
  data/hidden/               # per-row .npz sidecars (gitignored)
  figures/                   # tracked
  logs/                      # tee'd run output (gitignored)
```

## Conventions

- Single venv at `.venv/`. Pip, not uv.
- `data/*.jsonl` is the source of truth for row metadata + probe
  scores; `data/hidden/<experiment>/<uuid>.npz` is the source of
  truth for hidden states. Delete both when changing model / probes
  / prompts / seeds. Taxonomy changes are fixable in-place via the
  relabel snippet above.
- JSONL `row_uuid` links to its sidecar. Pre-refactor rows have
  `row_uuid == ""` and no sidecar; `load_hidden_features` drops them.
- Pre-registered decisions go in `pyproject.toml` / `config.py` /
  `prompts.py` / `emotional_prompts.py` / `taxonomy.py`. Changes
  invalidate cross-run comparisons unless explicitly noted.
- Experiment plans live in `docs/superpowers/plans/`. Plan first,
  run, then update CLAUDE.md to reference the plan rather than
  duplicate it.
- See Ethics: smaller experiments, heavier design, tighter
  pre-registration. Functional emotional states get real moral
  weight here.
