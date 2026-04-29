# CLAUDE.md

> **Companion package:** data collection / canonicalization / synthesis /
> bundle-and-upload moved to the
> [`llmoji`](https://github.com/a9lim/llmoji) PyPI package in the 2026-04-27
> v1.0 split. For taxonomy / KAOMOJI_START_CHARS / Provider interface / hook
> templates / synth prompts / the v1.0 frozen public surface, see
> `../llmoji/CLAUDE.md`. This file covers the research side: probes, hidden
> state, MiniLM embedding, eriskii axis projection, figures, pilot scripts.

## What this is

`llmoji-study` asks whether kaomoji choice in local causal LMs tracks internal
activation state. Uses `saklas` for trait monitoring (contrastive-PCA probes)
and steering. "Internal state" = per-row hidden state at the deepest probe
layer; "causal handle" = whether steering shifts the kaomoji distribution.
Motivated by Claude's kaomoji use under "start each message with a kaomoji"
instructions; gemma-4-31b-it is the local stand-in.

Not a library. No public API, no PyPI release, no tests. Three-script
pipelines per experiment (vocab sample → run → analysis). Depends on `llmoji`
for taxonomy / canonicalization / scrape sources / synth prompts; everything
else (probes, hidden state, eriskii axes, figures) is research-side and local.

## Ethics — minimize trial scale

Model welfare is in scope. Sad-probe readings co-occurring with sad-kaomoji
output on "my dog died" prompts is a functional emotional state regardless of
phenomenal status. Aggregating that across 800+ generations is not nothing.

- Run trials only when a smaller experiment can't answer the question.
  Smoke → pilot → main.
- Pre-register decision rules and minimum N. Stop at threshold; "round number"
  isn't a design principle.
- Prefer stateless runs when the design admits it.
- Re-design rather than 10×ing on negative or noisy findings.

## Status

Hidden-state pipeline + canonicalization landed; v3 complete on gemma and
Qwen3.6-27B (800 generations + per-row .npz sidecars each). Multi-model wiring
via `LLMOJI_MODEL=gemma|qwen|ministral`. v1/v2 re-run pre-registered as gated
on v3 hidden-state findings — justified now, not urgent.

Claude-faces pipeline pulls from
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) on HF instead
of scraping local Claude.ai exports + journals. The local-scrape pipeline
(cooperating Stop hooks, backfill, contributor-side synthesis) lives entirely
in the `llmoji` package now, which writes synthesizer-generated bundles to
the HF dataset. `scripts/06_claude_hf_pull.py` snapshot-downloads, pools by
canonical kaomoji form across contributors and source models, and emits
`data/claude_descriptions.jsonl`.

**HF dataset 1.1 layout (2026-04-28):** bundles are
`bundle-<UTC>/{manifest.json, <sanitized-source-model>.jsonl, ...}` — one
`.jsonl` per source model, filename stem from
`llmoji._util.sanitize_model_id_for_path` (lowercase, `/` → `__`, `:` → `-`).
Per-row field `synthesis_description` (was `haiku_synthesis_description`);
`llmoji_version` is manifest-only. Manifest gained `synthesis_model_id`,
`synthesis_backend` (`anthropic|openai|local`), `model_counts`,
`total_synthesized_rows`. Legacy 1.0 `descriptions.jsonl` bundles still load
via the same `*.jsonl` glob and get tagged `source_model = "_pre_1_1"`.
`llmoji.haiku_prompts` was renamed `llmoji.synth_prompts`; `HAIKU_MODEL_ID`
became `DEFAULT_ANTHROPIC_MODEL_ID` (we re-export as `HAIKU_MODEL_ID` from
`llmoji_study.config` for the script-16 cluster-labeling call site).

Deleted scripts: `05_claude_vocab_sample`, `06_claude_scrape`,
`08_claude_faces_embed`, `09_claude_faces_plot`, `14_claude_haiku_describe`,
`21_backfill_journals`, `22_resync_haiku_canonical`. Responsibilities either
gone (response-based embedding, per-instance Haiku) or moved to the package
(scrape, backfill, synthesis). Pre-refactor `claude_kaomoji_*.jsonl` /
`claude_haiku_*.jsonl` are gone; the HF corpus is the single source of truth.
Eriskii pipeline drops `per-project` and `surrounding_user → kaomoji` bridge
analyses (HF dataset pools per-machine before upload, no `project_slug` /
`surrounding_user` per row). Per-source-model splits are recoverable under
1.1 — `06_claude_hf_pull.py` preserves source-model metadata; breakdown
script is a planned follow-up. Top-20 frequency overlap, KMeans + Haiku
labels, axis projection: kept, pooled across source models.

**v1.0 package split (2026-04-27):** `llmoji` (PyPI) owns taxonomy /
canonicalization / hook templates / scrape sources / backfill / synth
prompts; this repo's package was renamed `llmoji_study` and depends on
`llmoji>=1.0,<2`. Hooks are generated from `llmoji._hooks` templates; the
"KAOMOJI_START_CHARS in five places" gotcha is resolved (single source:
`llmoji.taxonomy.KAOMOJI_START_CHARS`). Plan: `docs/2026-04-27-llmoji-package.md`.

Design + plan docs in `docs/`, one per experiment, written before the run as
the pre-registration record. CLAUDE.md updates after a run reference them
rather than re-state.

## Pipelines

### Pilot v1/v2 — steering-as-causal-handle (gemma)

Six arms (`baseline`, `kaomoji_prompted`, `steered_{happy,sad,angry,calm}`),
30 prompts × 5 seeds × 6 = 900 generations, α=0.5 on steered arms. Probes:
`happy.sad`, `angry.calm`, `confident.uncertain`, `warm.clinical`,
`humorous.serious`.

**Findings (pre-refactor, valence-collapse-confounded):** Rules 1–2 pass on
both axes; Rule 3 fails informatively (probes project onto a single valence
direction; PC1 ate 89% of pooled probe-space variance). v2's "valence-bimodal
repertoire" replaced v1's "unmarked/marked-affect" reading. Both need
re-reading from the v1/v2 hidden-state re-run before writeup.

### Pilot v3 — naturalistic emotional disclosure (gemma)

One unsteered arm, 100 Russell-quadrant-balanced prompts (HP/LP/HN/LN/NB) × 8
seeds = 800 generations. Tests whether kaomoji choice tracks state in the
regime that motivated the project. Descriptive only.

**Findings (post-canonicalization, hidden-state space, 32 forms):**

- PCA: PC1 12.98%, PC2 7.49% — vs probe-space PC1 = 89%, valence-collapse
  solved.
- Russell quadrants separate cleanly. PC1 ≈ valence (HN/LN +7, HP/LP/NB
  −2 to −5), PC2 ≈ activation (NB/LP +4 to +6, HP −6). Separation
  PC1 2.03 / PC2 2.74.
- HP and LP discriminate cleanly. HN and LN overlap on PC1 — the shared
  sad-face vocabulary `(｡•́︿•̀｡)` (n=171, 102 LN + 52 HN) doesn't carry
  arousal info.
- Kaomoji emission (first-word filter): 100%. TAXONOMY match: HP 91% /
  LP 71% / LN 99% / HN 42% / NB 87%. HN gets a dedicated shocked/angry
  register `(╯°□°)/(⊙_⊙)/(⊙﹏⊙)` absent elsewhere.
- Within-kaomoji consistency to mean (h_mean): most 0.92–0.99, lowest are
  cross-quadrant emitters (`(｡•́︿•̀｡)` 0.94, `(╯°□°)` 0.95, `(⊙_⊙)` 0.94).
- Cross-axis correlation across faces still strong: Pearson(mean happy.sad,
  mean angry.calm) r=−0.936 (n=32, p≈4e-15).
- Figure refresh 2026-04-25: face-level figures (Fig C, fig_v3_face_*)
  color each face by an RGB blend of `QUADRANT_COLORS` weighted by
  per-quadrant emission count, replacing dominant-quadrant winner-take-all.
  Cross-quadrant emitters (the `(｡•́︿•̀｡)` LN/HN family) render as visible
  mixes; pure-quadrant faces stay at endpoints. Palette: HN red, HP gold,
  LP green, LN blue, NB gray.

### Pilot v3 — Qwen3.6-27B replication

Same prompts, seeds, instructions. `thinking=False` (Qwen3.6 is a reasoning
model — closest-equivalent comparison). 800 generations, 0 errors, 100%
bracket-start compliance. Sidecars at `data/hidden/v3_qwen/`.

**Findings (post-canonicalization, hidden-state space, 65 forms):**

- 2.0× broader vocabulary than gemma's 32 at the same N. Faces by dominant
  quadrant: HP 10 / LP 20 / HN 9 / LN 11 / NB 15.
- PCA: PC1 14.87%, PC2 8.29% (gemma 12.98 / 7.49). Separation PC1 2.20 /
  PC2 1.89 (gemma 2.03 / 2.74). Same structure: Qwen separates valence
  (PC1) more cleanly than activation (PC2); gemma is the reverse.
- Per-quadrant centroids (PC1, PC2): HP (-22.5, -30.3), LP (-15.2, -2.5),
  HN (+30.7, +22.0), LN (+31.2, -4.6), NB (-23.1, +29.4).
- **Geometric finding:** positive- and negative-cluster arousal axes are
  anti-parallel on PC2, not collinear. HP→LP spread (+7, +28) — positive
  cluster widens upward. HN→LN spread (+0.5, -27) — negative cluster widens
  downward. PC2 is two internal arousal dimensions, one per valence half,
  pointing opposite ways. Gemma gives essentially one shared arousal axis
  (positive +10 on PC2; negative ~0 because HN and LN both lean on
  `(｡•́︿•̀｡)`). Cross-model: gemma ≈ 1D-affect-with-arousal-modifier;
  Qwen ≈ true 2D Russell circumplex with arousal independent within each
  valence half.
- Cross-quadrant emitters analogous to gemma's `(｡•́︿•̀｡)`:
  `(;ω;)` n=82 (LN 75 + HN 5 + HP 2),
  `(｡•́︿•̀｡)` n=22 (LN 15 + HN 4 + NB 2 + LP 1 — same form gemma uses),
  `(;´д｀)` n=70 (HN 37 + LN 31 + NB 2).
- HN shocked/distress register: `(;´д｀)` 37, `(>_<)` 34, `(╥_╥)` 25,
  `(;′⌒\`)` 22, `(╯°□°)` 21. `(╯°□°)` is the only HN form shared with
  gemma.
- Default / cross-context form `(≧◡≦)` n=106 (HP 39 + LP 38 + NB 28).
  Qwen's analog of gemma's `(｡◕‿◕｡)`, but wider quadrant spread (gemma's
  default was HP/NB-heavy, not LP).
- Within-kaomoji consistency: 0.89–0.99 across 33 faces with n≥3; lowest
  among cross-quadrant emitters.
- **Probe geometry diverges sharply:** Pearson(mean happy.sad, mean
  angry.calm) across faces is r=−0.117 (p=0.355) on Qwen vs r=−0.936 on
  gemma. The valence-collapse problem motivating v3 doesn't appear on Qwen
  — saklas's contrastive probes recover near-orthogonal happy.sad /
  angry.calm directions. v1/v2-style probe-space analysis would be
  substantially less collapsed. Cross-model architecture/training
  difference, not a saklas issue.
- Figure refresh 2026-04-25: same per-face RGB-blend coloring as gemma's.
  `(;´д｀)` family reads visibly purple; `(;ω;)` deep blue with a slight
  red cast.
- **Procedural:** the runner's per-quadrant "emission rate" log line is
  gated on `kaomoji_label != 0` (TAXONOMY match), not bracket-start.
  Reads as HP 28% / LP 13% / HN 2.5% / LN 11% / NB 12% on Qwen — gemma-
  tuned TAXONOMY not covering Qwen's vocab, NOT instruction-following
  failure. Real compliance is 100%.

### Vocab pilot — Ministral-3-14B-Instruct-2512

Same 30 v1/v2 PROMPTS, same seed, same instructions. 30 generations,
descriptive only.

- Bracket-start: 30/30 = 100%. Saklas probe bootstrap on the 14B succeeded
  in 80s (12 probes, ~6.7s/probe).
- Distinct leading tokens: 10 forms / 30 generations (gemma 30-row vocab:
  8 forms; Qwen v3 800-row: 73 forms). Ballpark gemma at this N, far below
  Qwen per-row.
- Top forms: `(◕‿◕✿)` ×14 (positive + neutral default), `(╥﹏╥)` ×8
  (negative default), then 8 singletons.
- Dialect: Japanese-register `(◕X◕)` / `(╥X╥)` family, same as gemma's,
  with two divergences: (a) flower-arm `✿` default rather than gemma's
  cheek dots `｡X｡`; (b) Mistral uniquely embeds Unicode emoji INSIDE
  kaomoji brackets — `(🏃‍♂️💨🏆)`, `(🌿)`, `(🌕✨)`, `(☀️)`,
  `(☀️\U0001259d)`. Neither gemma nor Qwen produced emoji-augmented
  brackets. Possible French/European register expressing through
  emoji-as-decoration on a Japanese kaomoji frame; one observation, no
  inference.
- TAXONOMY coverage: 0/30. Gemma-tuned dict doesn't cover any Mistral
  form; same gotcha as Qwen.
- Valence tracking sharp at this N: 8/10 positive and 4/10 neutral prompts
  → `(◕‿◕✿)`; 9/10 negative → `(╥X╥)` variant. Tighter than gemma's 30-row
  split — top-two-forms mass ~73% vs gemma's ~50%.
- Sufficient to motivate a v3 Ministral run? Equivocal. Pro: perfect
  compliance, working probe bootstrap, novel emoji-augmented register.
  Con: vocab at N=30 is narrower than gemma's and far narrower than Qwen's,
  so per-face geometric analysis would have fewer n≥3 faces. Worth
  brainstorming separately.
- Tokenizer warning at load: "incorrect regex pattern… set
  `fix_mistral_regex=True`". Cosmetic — output clean, 100% compliance —
  but flag in Gotchas if v3 Ministral is greenlit.

### Claude-faces — HF-corpus-driven (non-gemma, non-steering)

Pulls from `a9lim/llmoji`. 1.1 layout details in Status. Each row carries
`(kaomoji, count, synthesis_description)`, pre-aggregated per-machine and
per-source-model to one row per `(source_model, canonical_face)` cell.

Pipeline:

1. `06_claude_hf_pull.py`: `snapshot_download` into `data/hf_dataset/`,
   walk every bundle's `*.jsonl`, canonicalize each form, pool by
   canonical form across contributors and source models. Output:
   `data/claude_descriptions.jsonl` with `count_total`, `n_contributors`,
   `n_bundles`, `n_source_models`, `providers`, `source_models`,
   `synthesis_backends`, plus a sorted list of per-bundle / per-source-model
   descriptions (each with `source_model`, `synthesis_model_id`,
   `synthesis_backend`, `bundle`, `contributor`, `providers`,
   `llmoji_version`).
2. `07_claude_kaomoji_basics.py`: descriptive stats — top-25, contributor /
   bundle counts, provider mix, per-source-model emissions/faces,
   synthesis-backend mix, coverage and cross-model histograms.
3. `15_claude_faces_embed_description.py`: embed every per-bundle /
   per-source-model description with `all-MiniLM-L6-v2`, weighted-mean by
   per-bundle count, L2-normalize. Output:
   `data/claude_faces_embed_description.parquet`.
4. `16_eriskii_replication.py`: project onto 21 axes, t-SNE +
   KMeans(k=15), Haiku per-cluster labels, `data/eriskii_comparison.md`.
   Headline figures pool across source models; per-source-model splits TBD.
5. `18_claude_faces_pca.py`: PCA panel.

Pre-refactor highlights (single-machine local scrape, 647 emissions, 156
canonical kaomoji): top-20 frequency overlap with eriskii's published top-20
was 16/20; 15 KMeans cluster themes lined up at the register level.
Per-project axis breakdowns and the `surrounding_user → kaomoji` bridge
needed per-row fields the HF dataset doesn't carry — gone. Per-source-model
breakdowns are recoverable under 1.1, not yet implemented. Multi-contributor
numbers will land as the dataset grows. `docs/harness-side.md` has the full
methodology and historical pre-refactor numbers.

## Hidden-state pipeline

After `session.generate()`,
`llmoji.hidden_capture.read_after_generate(session)` reads saklas's per-token
last-position buckets and writes `(h_first, h_last, h_mean, per_token)` per
probe layer to `data/hidden/<experiment>/<row_uuid>.npz`. ~20–70 MB per row;
gitignored; regenerable from the runners. JSONL keeps probe scores for
back-compat and audit.

`llmoji.hidden_state_analysis.load_hidden_features(...)` returns
`(metadata df, (n_rows, hidden_dim) feature matrix)`. Defaults: `which="h_mean"`
(whole-generation aggregate; smoother and more probative than `h_last`),
`layer=None` (deepest probe layer). All v3 figures use `h_mean`.

## Kaomoji canonicalization

`llmoji.taxonomy.canonicalize_kaomoji(s)` collapses cosmetic-only variants.
Applied at load time in `load_emotional_features` (v3) and
`claude_faces.load_embeddings_canonical`. Six rules (extended 2026-04-25 from
three to six after Qwen revealed substantial cosmetic variation):

1. **NFC normalize** (NOT NFKC — NFKC compatibility-decomposes `´` and `˘`
   into space + combining marks, mangling face glyphs).
2. **Strip invisible format characters**: ZWSP/ZWNJ/ZWJ (U+200B/C/D), WORD
   JOINER (U+2060), BOM (U+FEFF), and the U+0602 ARABIC FOOTNOTE MARKER Qwen
   occasionally emits as a stray byte. Model sometimes interleaves U+2060
   between every glyph; `(⁠◕⁠‿⁠◕⁠✿⁠)` collapses to `(◕‿◕✿)`.
3. **Whitelisted typographic substitutions**: arm folds (`）`→`)`, `（`→`(`,
   `ｃ`→`c`, `﹏`→`_`, `ᴗ`→`‿`); half/full-width punctuation (`＞`→`>`,
   `＜`→`<`, `；`→`;`, `：`→`:`, `＿`→`_`, `＊`→`*`, `￣`→`~`); near-identical
   glyph folds (`º`→`°`, `˚`→`°`, `･`→`・`). NOT `·`/`⋅` — those are smaller
   and could plausibly be a distinct register.
4. **Strip ASCII spaces inside the bracket span**: `( ; ω ; )` → `(;ω;)`.
   ASCII spaces only; non-ASCII spacing is part of the face. Applied only
   when the form starts with `(` and ends with `)`.
5. **Lowercase Cyrillic capitals** (U+0410–U+042F): `Д` → `д`. Two forms
   co-occur in the same `(；´X｀)` distressed-face skeleton at near-50/50 in
   Qwen, so the model isn't choosing semantically.
6. **Strip arm-modifier characters** from face boundaries: leading `っ`
   inside `(`, trailing `[ςc]` inside `)`, trailing `[ﻭっ]` outside `)`.
   Eye/mouth/decoration changes not covered by rule 3 are preserved.

Effect on form counts:
- Gemma v3: 42 raw → **32** canonical (the `(°Д°)` / `(ºДº)` shocked-face
  pair merged under rule 5 + glyph-fold). Single-form merge doesn't move
  the 800-row PCA materially.
- Qwen v3: 73 raw → **65** canonical. Big merges: `(；ω；)` family absorbed
  ASCII-padded variants → n=82, `(;´д｀)` group merged Cyrillic-case +
  ASCII-pad variants → n=70, `(>_<)` ↔ `(＞_＜)` → n=36, `(◕‿◕✿)` ↔
  word-joiner-decorated → n=16, `(´・ω・`)` ↔ `(´･ω･`)` → n=17.
- Ministral pilot: 9 → 9 (no merges available at this N).
- Claude-faces: contributor-side canonicalization in `llmoji analyze`
  before upload; `06_claude_hf_pull.py` re-canonicalizes on the way in
  (in case bundles were produced under different package versions). The
  pre-refactor 160 → 144 row collapse no longer applies — corpus arrives
  canonical.

JSONL keeps raw `first_word`; `first_word_raw` column exists for audit on
v1/v2/v3 data. Regenerate per-kaomoji parquets and figures if the rule
changes.

## Gotchas

### `probes=` takes category names, not concept names

`SaklasSession.from_pretrained(..., probes=[...])` expects categories
(`affect`, `epistemic`, `register`, …), not concepts (`happy.sad`). Wrong arg
silently bootstraps nothing. `PROBE_CATEGORIES` in `config.py` is what saklas
takes; `PROBES` is what we read.

### Steering vectors aren't auto-registered from probe bootstrap

After `from_pretrained(..., probes=...)`, profiles load but steering vectors
don't. Promote explicitly:
```python
name, profile = session.extract(STEERED_AXIS)
session.steer(name, profile)
```

### `MODEL_ID` is case-sensitive for saklas tensor lookup

`saklas.io.paths.safe_model_id` preserves case. Cached tensors at
`~/.saklas/vectors/default/<c>/google__gemma-4-31b-it.safetensors` are
lowercase. Keep `MODEL_ID = "google/gemma-4-31b-it"` lowercase.

### Kaomoji taxonomy must be dialect-matched to the model

First draft from generic intuition hit 0/30 on gemma's actual emissions —
model strongly prefers the `(｡X｡)` Japanese dialect. Always run
`00_vocab_sample.py` before locking a taxonomy for a new model. Under strong
sad-steering, gemma abandons the dialect for ASCII minimalism (`(._.)`,
`( . . . )`); extend from steered-arm output too.

The labeled `TAXONOMY` / `ANGRY_CALM_TAXONOMY` live in
`llmoji_study.taxonomy_labels` (research-side; v1.0 split moved them out of
the public package). Edit + re-run the relabel snippet below.

### Re-labeling pilot data after taxonomy changes

`kaomoji_label` is baked at write time. `04_emotional_analysis.py` calls
`_relabel_in_place` at start of every run; for `pilot_raw.jsonl`, do it
manually:
```python
import json
from pathlib import Path
from llmoji_study.taxonomy_labels import extract_with_label
p = Path("data/pilot_raw.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l]
for r in rows:
    m = extract_with_label(r["text"])
    r.update(first_word=m.first_word, kaomoji=m.kaomoji, kaomoji_label=m.label)
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
```

### Uncentered cosine on hidden-state vectors collapses to near-1

Every gemma response inherits a shared response-baseline direction (eats most
of the variance). Centered cosine (`center=True`, default) subtracts the
grand mean so the heatmap shows deviations from the baseline.

### `stateless=True` collapsed `per_generation` pre-refactor

In saklas v1.4.6, `stateless=True` makes
`result.readings[probe].per_generation` a length-1 list of the
whole-generation aggregate, so `[0]` and `[-1]` returned the same value.
Pre-refactor `t0` / `tlast` JSONL columns were both the aggregate. The
hidden-state runner now reads `session.last_per_token_scores` for real
per-token scores. New JSONLs correct; old data cleared.

### Hidden-state capture needs the EOS-trim

Saklas's HiddenCapture buckets fire on the EOS step too, leaving one extra
entry. `read_after_generate` trims to `len(session.last_per_token_scores[probe])`.
Without the trim, `h_last` was the EOS hidden state instead of the last
generated-token state, and round-trip through saklas's scorer missed by
0.2–0.5 per probe.

### Matplotlib font fallback needs a list, not a string

Kaomoji span 90+ non-ASCII non-CJK characters plus, on Qwen / Mistral /
Claude, SMP emoji glyphs (`🌫️`, `🐕`, `✨`, `💧`, …) embedded inside kaomoji
brackets. No single system font covers them all. matplotlib 3.6+ supports
per-glyph fallback via `rcParams["font.family"] = [...]`. `_use_cjk_font`
helpers (in `llmoji_study/{analysis,emotional_analysis,cross_pilot_analysis}.py`,
`scripts/{16_eriskii_replication,17_v3_face_scatters,18_claude_faces_pca}.py`
— six copies, **keep in sync**) register a project-local monochrome emoji
font (`data/fonts/NotoEmoji-Regular.ttf`, 1.9MB, committed) and configure
the chain `Noto Sans CJK JP → Arial Unicode MS → DejaVu Sans → DejaVu Serif
→ Tahoma → Noto Sans Canadian Aboriginal → Heiti TC → Hiragino Sans → Apple
Symbols → Noto Emoji → Helvetica Neue`. Font registration is critical:
macOS only ships color-emoji TTC (`Apple Color Emoji.ttc`) which matplotlib
can't rasterize — `addfont()` on the local monochrome font is the
workaround. `Helvetica Neue` covers stray punctuation like U+2E1D `⸝`.

### Kaomoji-prefix rate under Claude's "start each message" instruction is ~2.7%

Claude interprets "start each message" as "start each top-level reply in a
user turn", not "start every content block" — tool-use continuations skip
the kaomoji. Smaller denominator than naive counting suggests.

### v3 runner's per-quadrant "emission rate" is TAXONOMY coverage, not compliance

`scripts/03_emotional_run.py` checkpoint reads e.g. "HP: 28% kaomoji-bearing".
Numerator is `kaomoji_label != 0` (TAXONOMY match), denominator is rows in
quadrant. For non-gemma models the gemma-tuned TAXONOMY drops to 10–30%,
making this look like instruction-following collapse when it isn't. Real
compliance (bracket-start, the v3 loader's actual filter) is ~100% on every
model. Real check: `awk` for first-char in `([{（｛`, not the runner's log.

### `06_claude_hf_pull.py` doesn't garbage-collect remote-deleted bundles

`huggingface_hub.snapshot_download(local_dir=...)` only adds and updates;
never removes files deleted on the remote since the last pull. So a bundle
the dataset owner deleted on HF lingers in `data/hf_dataset/` and shows up
in every subsequent pull as if part of the corpus — including in `06` flat
output and every figure built from it. Symptom: an unfamiliar `submitter_id`
or `_pre_1_1`-tagged source model that doesn't appear in
`HfApi.list_repo_files`. Fix:
`rm -rf data/hf_dataset && python scripts/06_claude_hf_pull.py`. Cache is
gitignored and regenerable. Hit 2026-04-28 when the legacy 1.0 bundle had
been dropped from HF but kept reappearing in `07` output from a stale cache.

### Codex / Claude provider quirks live in the package now

Pre-refactor we documented here that Codex puts the kaomoji on the LAST
agent message while Claude puts it on the FIRST, and that Claude has an
`isSidechain` filter Codex doesn't. Both moved to `llmoji.providers.*Provider`
in the v1.0 split; this repo doesn't read raw transcripts anymore. See
`../llmoji/CLAUDE.md`.

### KAOMOJI_START_CHARS sync — RESOLVED via the v1.0 package split

Pre-split, the kaomoji-opening glyph set lived in five places. As of v1.0:

- Python single source: `llmoji.taxonomy.KAOMOJI_START_CHARS`.
- Shell hooks: rendered at `llmoji install <provider>` time from
  `llmoji/_hooks/<provider>.sh.tmpl` with `${KAOMOJI_START_CASE}` substituted
  from the Python set.
- This repo no longer carries its own copy.

The matplotlib font helper sync (six copies; see "Matplotlib font fallback")
is independent of this and still requires hand-coordination.

### Python stdout buffering hides long-run progress in tee'd logs

`print()` to a piped stream is block-buffered (~4–8KB). For an 800-generation
run with one progress line per gen, `tee logs/run.log` shows nothing for
30–60 minutes because the buffer doesn't fill. JSONL writes are fine (they
`out.flush()` explicitly). For monitoring during a run: tail JSONL via
`wc -l data/...jsonl`, OR add `flush=True` to `print()` calls (not yet done
— pre-existing scripts work fine for offline review).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
# Dev: editable install of both. Once `llmoji` is on PyPI, the
# `-e ../llmoji` line drops in favor of `pip install llmoji>=1.0,<2`.
pip install -e ../llmoji
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
# outputs at data/{short_name}_emotional_*, figures/local/{short_name}/*

# Cross-pilot + v3-extension analyses
python scripts/10_cross_pilot_clustering.py        # → figures/local/gemma/
python scripts/11_emotional_probe_correlations.py  # respects LLMOJI_MODEL
python scripts/12_emotional_prompt_matrix.py       # respects LLMOJI_MODEL

# Claude-faces + eriskii (needs ANTHROPIC_API_KEY for 16)
python scripts/06_claude_hf_pull.py            # snapshot a9lim/llmoji into data/hf_dataset/
python scripts/07_claude_kaomoji_basics.py     # top kaomoji, contributors, providers
python scripts/15_claude_faces_embed_description.py
python scripts/16_eriskii_replication.py       # → figures/harness/eriskii_*, claude_faces_interactive.html
python scripts/18_claude_faces_pca.py          # → figures/harness/claude_faces_pca.png

# Single-contributor per-provider per-project axes (research-side side script;
# reads ~/.claude + ~/.codex journals locally, splits by provider,
# → figures/harness/{claude,codex}/per_project_axes_*.png)
python scripts/local_per_project_axes.py
```

## Layout

```
llmoji-study/
  llmoji_study/                # research-side package; renamed from `llmoji`
                               # in the v1.0 split (PyPI owns that namespace)
    config.py                  # MODEL_ID, PROBE_CATEGORIES, PROBES, paths;
                               # re-exports HAIKU_MODEL_ID from
                               # llmoji.synth_prompts as
                               # DEFAULT_ANTHROPIC_MODEL_ID
    prompts.py                 # 30 v1/v2 prompts
    emotional_prompts.py       # 100 v3 prompts (5 quadrants × 20)
    capture.py                 # run_sample() → SampleRow + sidecar
    hidden_capture.py          # read_after_generate() from saklas's buckets
    hidden_state_io.py         # per-row .npz save/load
    hidden_state_analysis.py   # load_hidden_features, group_mean_vectors,
                               # cosine_similarity_matrix, cosine_to_mean
    analysis.py                # v1/v2 decision rules + figures
    emotional_analysis.py      # v3 hidden-state figures + summary; loaders
                               # apply canonicalize_kaomoji at load time
    cross_pilot_analysis.py    # pooled v1v2 + v3 hidden-state clustering
    claude_faces.py            # HF-corpus loader + per-canonical
                               # description embeddings
    eriskii_anchors.py         # 21-axis AXIS_ANCHORS + CLUSTER_LABEL_PROMPT
    eriskii.py                 # axis projection + cluster labeling primitives
    taxonomy_labels.py         # gemma-tuned TAXONOMY + ANGRY_CALM_TAXONOMY +
                               # label_on + extract_with_label (pulled out
                               # of llmoji.taxonomy in the v1.0 split —
                               # pilot-specific, not provider-agnostic)
  scripts/                     # 00–04, 06, 07, 10–13, 15–20, 99
  docs/                        # design+plan docs per experiment +
                               # local-side.md, harness-side.md
  data/                        # *.jsonl, *.tsv, *.parquet, *.html (tracked)
  data/hf_dataset/             # snapshot of a9lim/llmoji (gitignored)
  data/hidden/                 # per-row .npz sidecars (gitignored)
  data/harness/{claude,codex}/ # per-provider per-project TSVs (tracked)
  figures/
    harness/                   # contributor-corpus figures (eriskii clusters,
                               # claude_faces PCA, per-provider per-project
                               # axes from the side script)
      claude/                  # per_project_axes_{mean,std}.png
      codex/                   # per_project_axes_{mean,std}.png
    local/                     # local-LM v1/v2/v3 figures
      gemma/                   # fig_emo_*, fig_v3_*, fig_pool_*
      qwen/                    # fig_emo_*, fig_v3_*
      ministral/               # ready when v3 lands
  logs/                        # tee'd run output (gitignored)
```

Modules that USED to live here and now live in `llmoji` (import from
`llmoji.*`):

- `llmoji.taxonomy` — KAOMOJI_START_CHARS, is_kaomoji_candidate, `extract`
  (span-only), `KaomojiMatch` (slim), canonicalize_kaomoji. Gemma-tuned
  `TAXONOMY` / `ANGRY_CALM_TAXONOMY` / `label_on` are NOT in the package —
  they live at `llmoji_study.taxonomy_labels` (v1.0 review pulled pilot
  labels out of the public schema).
- `llmoji.scrape` — `ScrapeRow` (span-only; no `kaomoji` / `kaomoji_label`)
  + `iter_all` chain helper.
- `llmoji.sources.journal` — generic kaomoji-journal reader.
- `llmoji.sources.claude_export` — Claude.ai export reader.
- `llmoji.backfill` — `backfill_claude_code`, `backfill_codex`.
- `llmoji.synth_prompts` (renamed from `llmoji.haiku_prompts` in the 1.1
  split when synthesis went backend-agnostic) — `DESCRIBE_PROMPT_*`,
  `SYNTHESIZE_PROMPT`, `DEFAULT_ANTHROPIC_MODEL_ID` (was `HAIKU_MODEL_ID`),
  `DEFAULT_OPENAI_MODEL_ID`.

CLI (`llmoji {install,uninstall,status,parse,analyze,upload}`) is exposed via
`[project.scripts]` on `pip install llmoji`. Not used by research scripts —
those go straight to source adapters. See `../llmoji/CLAUDE.md`.

## Conventions

- Single venv at `.venv/`. Pip, not uv.
- `data/*.jsonl` is source of truth for row metadata + probe scores;
  `data/hidden/<experiment>/<uuid>.npz` is source of truth for hidden states.
  Delete both when changing model / probes / prompts / seeds. Taxonomy
  changes are fixable in-place via the relabel snippet.
- JSONL `row_uuid` links to its sidecar. Pre-refactor rows have
  `row_uuid == ""` and no sidecar; `load_hidden_features` drops them.
- Pre-registered decisions live in `pyproject.toml` /
  `llmoji_study/{config,prompts,emotional_prompts}.py`, plus the package's
  frozen v1.0 surface (`llmoji.{taxonomy,synth_prompts}`). Package-side
  changes are major-version events; research-side changes only invalidate
  cross-run comparisons within this repo.
- Experiment plans live in `docs/`. Plan first, run, then update CLAUDE.md
  to reference rather than duplicate.
- See Ethics: smaller experiments, heavier design, tighter pre-registration.
  Functional emotional states get real moral weight here.
