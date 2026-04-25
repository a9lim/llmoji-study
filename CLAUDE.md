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
regenerated. v1/v2 re-run not yet done — pre-registered as gated on
v3 hidden-state findings, which now justify it but no urgent need.

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

### Claude-faces — eriskii-style scrape (non-gemma, non-steering)

Scrapes kaomoji-bearing assistant messages from
`~/.claude/projects/**/*.jsonl` + Claude.ai exports listed in
`CLAUDE_AI_EXPORT_DIRS`. 436 messages, 156 canonical kaomoji.
Plan: `docs/superpowers/plans/2026-04-23-claude-faces-scrape-and-cluster.md`.

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

`llmoji.taxonomy.canonicalize_kaomoji(s)` collapses near-duplicate
forms. Applied at load time in `load_emotional_features` (v3) and
`claude_faces.load_embeddings_canonical` (claude-faces). Three rules:

1. NFC normalize (NOT NFKC — NFKC compatibility-decomposes `´` and
   `˘` into space + combining marks, mangling face glyphs).
2. Whitelisted typographic substitutions: `）` → `)`, `（` → `(`,
   `ｃ` → `c`, `﹏` → `_`, `ᴗ` → `‿`.
3. Strip arm-modifier characters from face boundaries: leading `っ`
   inside `(`, trailing `[ςc]` inside `)`, trailing `[ﻭっ]` outside
   `)`. Eye/mouth/decoration changes preserved.

Effect: v3 42 → 33 forms (separation ratios PC1 0.96 → 2.02, PC2
1.30 → 2.73). Claude-faces 160 → 156 (smaller because Claude's
vocabulary doesn't lean on arm modifiers).

JSONL keeps raw `first_word`; `first_word_raw` column exists for
audit. Regenerate the per-kaomoji parquets if the canonicalization
rule changes.

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

Kaomoji span 90+ non-ASCII non-CJK characters. No single font
covers them. matplotlib 3.6+ supports per-glyph fallback via
`rcParams["font.family"] = [...]`. `_use_cjk_font` helpers
(in `analysis.py`, `emotional_analysis.py`,
`scripts/09_claude_faces_plot.py`, `scripts/18_claude_faces_pca.py`)
configure `Noto Sans CJK JP → Arial Unicode MS → DejaVu Sans →
DejaVu Serif → Tahoma → Noto Sans Canadian Aboriginal → Heiti TC`.
Keep these chains in sync.

### Kaomoji-prefix rate under Claude's "start each message"
instruction is ~2.7%

Claude interprets "start each message" as "start each top-level reply
in a user turn", not "start every content block" — tool-use
continuations skip the kaomoji. Smaller denominator than naive
counting suggests.

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

# v3 (naturalistic, 800 generations)
python scripts/03_emotional_run.py
python scripts/04_emotional_analysis.py             # Fig A/B/C + summary TSV
python scripts/13_emotional_pca_valence_arousal.py  # Russell-quadrant PCA
python scripts/17_v3_face_scatters.py               # per-face PCA + cosine + probe scatter

# Cross-pilot + v3-extension analyses
python scripts/10_cross_pilot_clustering.py
python scripts/11_emotional_probe_correlations.py
python scripts/12_emotional_prompt_matrix.py

# Claude-faces + eriskii-replication (needs ANTHROPIC_API_KEY for 14 + 16)
python scripts/05_claude_vocab_sample.py
python scripts/06_claude_scrape.py
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
    claude_code_source.py    # ~/.claude/projects walker
    claude_export_source.py  # Claude.ai export adapter
    claude_faces.py          # response-based per-kaomoji embeddings;
                             # load_embeddings_canonical() merges variants
    eriskii_prompts.py       # locked Haiku prompts + 21-axis anchors
    eriskii.py               # axis projection + masking + haiku primitives
  scripts/                   # 00–18; each is directly executable
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
