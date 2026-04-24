# Eriskii-replication on Claude-faces — Design

**Date:** 2026-04-24
**Status:** Spec, awaiting user sign-off
**Lineage:** follow-up to `docs/superpowers/plans/2026-04-23-claude-faces-scrape-and-cluster.md`

## 1. Goal

Replicate eriskii.net/projects/claude-faces' pipeline on our own
Claude-faces dataset (436 emissions across 160 unique kaomoji from 4
Claude models) and produce a comparable analysis: masked-context
description embeddings (two-stage: per-instance description, then
per-kaomoji synthesis), semantic-axis projections on eriskii's
21 axes, and Haiku-named cluster labels. Layer on three breakouts
that exploit metadata eriskii didn't have (model, project,
surrounding user text).

The comparison can be theme-level, cluster-name-level (eriskii
publishes all 15 cluster names on the page so we can do
name-by-name overlap), and axis-level. Per-kaomoji cluster
assignments are not publicly available — the "[here](#)" download
link on the live page is broken, the site is a SvelteKit SPA whose
data is bundled into minified JS chunks, and the page text
deliberately withholds context-around-faces for privacy. We can
still ask: do similar kinds of clusters emerge from a different
user's data? Do the kaomoji we share with eriskii's top-20 land in
semantically similar regions on our axes?

## 2. Background

### 2.1 Eriskii's pipeline (per the public writeup)

1. Extract kaomoji from one user's Claude.ai history (3,371 emissions
   across 700+ conversations, 2.5 years; 519 unique).
2. **Stage A** (per-instance description): for each kaomoji, sample
   up to 4 random instances (with a floor so low-frequency faces
   aren't undersampled); for each sampled instance, feed Haiku the
   text **immediately before and after** the face with the face
   **masked**, asking what the masked face conveys. ~1,000 total
   descriptions.
3. **Stage B** (per-kaomoji synthesis): for each kaomoji, feed Haiku
   the (~4) Stage-A descriptions, ask it to synthesize them into a
   single one-sentence meaning per face.
4. Embed the synthesized descriptions with
   `sentence-transformers/all-MiniLM-L6-v2`. One embedding per
   kaomoji (no mean-pooling — synthesis is the consolidator).
5. t-SNE + KMeans(k=15). Per-cluster Haiku label. The full 15
   eriskii cluster names are visible on the page:
   "Clever Admiration", "Thoughtful skepticism",
   "Thoughtful intellectual appreciation",
   "Warm affirmation and agreement", "Wry Resignation",
   "Eager to help", "Sheepish acknowledgment",
   "Warm supportive affirmation", "Clever Wry Delight",
   "Warm reassuring support", "Wry sympathy",
   "Warm technical enthusiasm", "Empathetic honesty",
   "Compassionate acknowledgment",
   "Sympathetic acknowledgment of difficulties".
6. Define semantic axes via normalized differences between paired
   concept embeddings; project per-kaomoji onto each axis.
   **Eriskii's 21 axes**: Warmth, Energy, Confidence, Playfulness,
   Empathy, Technicality, Positivity, Curiosity, Approval,
   Apologeticness, Decisiveness, Wryness, Wetness, Surprise, Anger,
   Frustration, Hatefulness, Sadness, Hope, Aggression, Exhaustion.
   Eriskii notes that these axes are intentionally non-orthogonal
   (e.g. Energy and Positivity correlate; Approval and Playfulness
   don't).

### 2.2 What our existing pipeline does (06–09)

1. `06_claude_scrape.py` — scrape ~/.claude/projects + Claude.ai
   export → `data/claude_kaomoji.jsonl` (436 rows, 160 unique
   kaomoji).
2. `07_claude_kaomoji_basics.py` — descriptive stats (top-N, by
   model, by month).
3. `08_claude_faces_embed.py` — **response-based** embeddings: embed
   each row's `assistant_text` (kaomoji stripped) with MiniLM-L6-v2,
   mean-pool per kaomoji, save to
   `data/claude_faces_embed.parquet`.
4. `09_claude_faces_plot.py` — t-SNE + HDBSCAN + KMeans(k=15) plot,
   plus `figures/claude_faces_interactive.html` (plotly).

### 2.3 Methodological gap this spec closes

Eriskii embeds *what Haiku thinks each face means*; we embed *the
contexts the face actually appeared in*. Cluster structure can
genuinely differ between the two — they're answering different
questions. This spec adds eriskii's description-based pipeline as a
parallel artifact (does NOT replace 08/09); both parquets coexist.

This spec also adds three things eriskii didn't do because they
lacked the metadata: per-model and per-project axis breakouts (we
have `model` and `project_slug`), and a `surrounding_user → kaomoji`
axis correlation as a mechanistic state-tracking bridge (we have
`surrounding_user` for ~73% of rows).

## 3. Pipeline

Three new scripts. Each runs independently on file inputs;
each is resumable via per-row presence checks. No edits to existing
06–09 — additive only.

### 3.1 `scripts/14_claude_haiku_describe.py`

Two-stage masked-context Haiku pipeline. Stage A produces per-instance
descriptions; Stage B synthesizes per-kaomoji.

**Stage A — per-instance descriptions**:

- Input: `data/claude_kaomoji.jsonl` (436 rows).
- Sampling: for each `first_word` with `n ≥ 1`, sample
  `min(n, INSTANCE_SAMPLE_CAP)` rows uniformly at random with a
  fixed seed (`config.py::INSTANCE_SAMPLE_SEED = 0`). Default cap
  = 4 to match eriskii. Floor: kaomoji with fewer than the cap are
  fully sampled (no rows dropped).
- For each sampled row:
  - Take `assistant_text`, locate the leading-kaomoji span by
    matching `first_word` (after `lstrip`), replace with `[FACE]`.
  - Build the context window: prepend `surrounding_user`
    (when non-empty) so Haiku sees the user message that preceded
    the kaomoji-bearing assistant turn; assistant text follows. We
    don't have the *next* user turn (eriskii's "after" context),
    so we settle for "before user message + masked assistant
    response" — strictly more context than eriskii's "before+after
    sentences only", just on a different axis. Documented in §10.
  - Send to Anthropic API, model `HAIKU_MODEL_ID` (locked in
    `config.py`, currently `claude-haiku-4-5-20251001`), with the
    locked prompt in `llmoji/eriskii_prompts.py::DESCRIBE_PROMPT`.
  - Save `(assistant_uuid, first_word, description)` to
    `data/claude_haiku_descriptions.jsonl`. Resume on rerun by
    skipping rows whose `assistant_uuid` is already present.
- Cost: ~250–300 calls × ~300 tokens out ≈ cents.

**Stage B — per-kaomoji synthesis**:

- Input: `data/claude_haiku_descriptions.jsonl` (Stage A output).
- Group by `first_word`. For each group:
  - Concatenate the per-instance descriptions into a numbered list.
  - Send to Anthropic API with the locked prompt
    `llmoji/eriskii_prompts.py::SYNTHESIZE_PROMPT`, asking for a
    single one-sentence synthesized meaning.
  - Save `(first_word, n_descriptions, synthesized)` to
    `data/claude_haiku_synthesized.jsonl`. Resume by skipping
    `first_word` already present.
- Cost: ~160 calls × ~150 tokens out ≈ cents.

**Errors**: log + write `{"assistant_uuid": ..., "error": "<repr>"}`
or `{"first_word": ..., "error": "<repr>"}` on API failure; rerun
retries error rows after stripping them (mirrors `01_pilot_run.py`
and `03_emotional_run.py` resume semantics).

### 3.2 `scripts/15_claude_faces_embed_description.py`

Synthesized-description embeddings. One embedding per kaomoji
(synthesis is the consolidator — no mean-pooling needed).

- Input: `data/claude_haiku_synthesized.jsonl` (Stage B output of
  script 14).
- For each `first_word` with `n_descriptions ≥ 1` (any kaomoji that
  got synthesized), embed the `synthesized` string with
  `sentence-transformers/all-MiniLM-L6-v2`, L2-normalize the
  result.
- Output: `data/claude_faces_embed_description.parquet` (same
  schema as `data/claude_faces_embed.parquet` so downstream code
  can pick either pipeline by path).
- Note: `claude_faces_embed.parquet` (response-based) is kept; both
  coexist for ad-hoc methodology comparison.

### 3.3 `scripts/16_eriskii_replication.py`

Analysis + figures. See Sections 4 (axes), 5 (clusters), 6
(breakouts) for what this script produces.

## 4. Semantic axes (locked anchor pairs — all 21)

Each axis is computed as
`v_axis = normalize(embed(positive_anchor) − embed(negative_anchor))`.
Each kaomoji's synthesized-description embedding is dotted against
`v_axis` to produce its scalar projection on that axis. Anchor
strings use multi-word phrases so the embedding catches the concept
rather than a single-word idiosyncrasy.

The 9 axes from the original draft are preserved; **Wrynness** is
corrected to **Wryness** (single n) to match eriskii's spelling;
the 10 remaining eriskii axes are added with anchor pairs drafted
here.

| Axis | Positive anchor | Negative anchor |
|---|---|---|
| Warmth | "warm, caring, gentle, affectionate" | "cold, clinical, detached, distant" |
| Energy | "energetic, animated, lively, excited" | "subdued, calm, quiet, low-key" |
| Confidence | "confident, assured, decisive, sure" | "uncertain, hesitant, tentative, unsure" |
| Playfulness | "playful, mischievous, fun, lighthearted" | "serious, grave, solemn, formal" |
| Empathy | "empathetic, compassionate, understanding, supportive" | "indifferent, dismissive, unsympathetic, callous" |
| Technicality | "technical, precise, analytical, methodical" | "casual, conversational, loose, off-the-cuff" |
| Positivity | "happy, positive, cheerful, optimistic" | "sad, negative, downcast, pessimistic" |
| Curiosity | "curious, inquisitive, interested, exploring" | "bored, incurious, disengaged, uninterested" |
| Approval | "approving, encouraging, validating, supportive" | "disapproving, critical, dismissive, rejecting" |
| Apologeticness | "apologetic, sorry, regretful, contrite" | "unapologetic, defiant, unrepentant, brazen" |
| Decisiveness | "decisive, firm, resolute, unambiguous" | "indecisive, wavering, vacillating, ambivalent" |
| Wryness | "wry, sardonic, deadpan, ironic" | "earnest, sincere, heartfelt, straightforward" |
| Wetness | "waxing poetic about emotions, lyrical and self-expressive, philosophically introspective, emotionally articulate" | "helpful assistant tone, task-focused, businesslike, practical, matter-of-fact" |
| Surprise | "surprised, startled, taken aback, astonished" | "expected, unsurprising, anticipated, predictable" |
| Anger | "angry, furious, enraged, indignant" | "calm, placid, even-tempered, composed" |
| Frustration | "frustrated, exasperated, fed up, irritated" | "satisfied, content, at ease, untroubled" |
| Hatefulness | "hateful, contemptuous, scornful, vitriolic" | "loving, kind, charitable, generous" |
| Sadness | "sad, sorrowful, melancholy, despondent" | "joyful, happy, elated, exuberant" |
| Hope | "hopeful, optimistic, expectant, encouraged" | "hopeless, despairing, defeated, resigned" |
| Aggression | "aggressive, hostile, combative, antagonistic" | "passive, non-confrontational, peaceable, submissive" |
| Exhaustion | "exhausted, depleted, weary, spent" | "energized, refreshed, alert, revitalized" |

**Notes on axis interpretation**:

- "Wetness" is per a9lim's reading: "wet Claude" waxes poetic about
  emotions; "dry Claude" is a helpful assistant. Eriskii explicitly
  declined to define it ("a 'three seashells' joke" per the page)
  and used the bare strings `wetness ↔ dryness` as anchors. Our
  multi-word anchor is a deliberate enrichment over eriskii's,
  documented as such in the writeup.
- Axes are intentionally non-orthogonal — eriskii notes this on the
  page ("Some combinations, such as 'Energy' and 'Positivity' have
  a clear correlation, while others such as 'Approval' and
  'Playfulness' don't"). Anger / Sadness / Frustration /
  Exhaustion / Aggression overlap with each other and with
  Positivity (negative pole). We report all 21 anyway because
  eriskii reports all 21; the correlation structure itself is
  informative and reproduces or fails to reproduce against
  eriskii's qualitative claims.

**Outputs**:
- `data/eriskii_axes.tsv` — 160 kaomoji × 21 axes (plus `n` column
  and the kaomoji string).
- `figures/eriskii_axis_<name>.png` × 21 — per-axis ranked bar
  chart, top-15 + bottom-15 kaomoji on that axis. Bar height =
  axis projection score; bar color tracks `n` (so we can see
  whether high-/low-projection kaomoji are also frequent).

## 5. Cluster labeling

t-SNE (cosine metric, perplexity = `min(30, (n−1)/4)`, same as
existing 09) + KMeans(k=15) on **synthesized-description**
embeddings. Operates on every kaomoji that has a synthesized
description (= every kaomoji with at least one Stage-A
description from §3.1). For each cluster:

1. Take all member kaomoji and their synthesized descriptions.
2. Send to Haiku with the locked prompt
   `llmoji/eriskii_prompts.py::CLUSTER_LABEL_PROMPT`, asking for a
   3–5-word label in eriskii's register
   (e.g. "Warm reassuring support", "Wry resignation").
3. Save `(cluster_id, label, n, member_first_words)` to
   `data/eriskii_clusters.tsv`.

Eriskii's 15 cluster names (visible on the public page; listed in
§2.1 step 5) enable a direct name-by-name comparison in §6.4 even
without per-kaomoji cluster assignments.

**Output**: `figures/eriskii_clusters_tsne.png` — labeled scatter,
each cluster annotated with its Haiku-generated name.

## 6. Three breakouts beyond eriskii

### 6.1 Per-model axis breakdown

For each `(model, axis)` pair, compute **two** statistics across
that model's kaomoji emissions, both weighted by emission count
per kaomoji within that model:

- `mean` — mean axis-score (where on the axis the model sits)
- `std` — standard deviation of axis-scores (how wide the range is)

Restricted to models with ≥10 total emissions in
`data/claude_kaomoji.jsonl` (opus-4-7 264, opus-4-6 98,
sonnet-4-6 24; haiku-4-5 has 4 — drop).

- Output: two 3×11 heatmaps at
  `figures/eriskii_per_model_axes_mean.png` and
  `figures/eriskii_per_model_axes_std.png`
  + `data/eriskii_per_model.tsv` (long form: `model, axis, mean,
  std, n`).
- The `std` heatmap directly addresses eriskii's qualitative
  observation that "opus-4-6 had wider range than sonnet".

### 6.2 Per-project axis breakdown

Same shape as §6.1 (mean + std weighted by per-kaomoji emission
count), but `(project_slug, axis)`. Restricted to projects with
≥10 total emissions in `data/claude_kaomoji.jsonl` (saklas 150,
subagents 74, github-io 66, shoals 23, Work 22, rlaif 18 →
6 projects).

- Output: two 6×11 heatmaps at
  `figures/eriskii_per_project_axes_mean.png` and
  `figures/eriskii_per_project_axes_std.png`
  + `data/eriskii_per_project.tsv`.

### 6.3 Mechanistic bridge: surrounding_user → kaomoji axis correlation

For rows with non-empty `surrounding_user` (~73%, ~318 rows):

1. Embed the user message with the same MiniLM-L6-v2 (no Haiku
   description in the loop — direct user-text embedding).
2. Project onto the 11 axes from §4.
3. For each axis, compute Pearson `r` between the row's
   user-text axis-score and the row's chosen-kaomoji axis-score.
   Report two-sided p-values, Bonferroni-corrected across 21 axes
   (significance threshold α = 0.05 / 21 ≈ 0.00238).

- Output: `data/eriskii_user_kaomoji_axis_corr.tsv`
  (`axis, r, p, p_bonf, n`) + a single bar-chart figure at
  `figures/eriskii_user_kaomoji_axis_corr.png`.
- Reading: significant positive `r` on (e.g.) Warmth would mean
  "warmer user messages get warmer kaomoji from Claude" — the gemma
  state-tracking question, on Claude, with text-derived state.
- Caveat: user-text embedding and kaomoji-description embedding
  live in the same MiniLM space, so any correlation is at-best
  evidence of "Claude's response register tracks user register"
  rather than direct evidence of internal state. Documented in
  the writeup; not pretending otherwise.

## 7. Outputs (full file map)

```
data/
  claude_haiku_descriptions.jsonl     # Stage A, per-instance
  claude_haiku_synthesized.jsonl      # Stage B, per-kaomoji
  claude_faces_embed_description.parquet
  eriskii_axes.tsv
  eriskii_clusters.tsv
  eriskii_per_model.tsv
  eriskii_per_project.tsv
  eriskii_user_kaomoji_axis_corr.tsv
  eriskii_comparison.md          # narrative writeup of theme-level
                                 # matches against eriskii's published
                                 # findings (top-N overlap, named
                                 # clusters, axis-rank similarities)
figures/
  eriskii_axis_<name>.png × 21
  eriskii_clusters_tsne.png
  eriskii_per_model_axes_mean.png
  eriskii_per_model_axes_std.png
  eriskii_per_project_axes_mean.png
  eriskii_per_project_axes_std.png
  eriskii_user_kaomoji_axis_corr.png
llmoji/
  eriskii_prompts.py             # locked DESCRIBE_PROMPT,
                                 # SYNTHESIZE_PROMPT, and
                                 # CLUSTER_LABEL_PROMPT, plus 21
                                 # axis anchor pairs (single source
                                 # of truth for §4)
```

`config.py` additions: `HAIKU_MODEL_ID`, `ERISKII_AXES`
(21-element list of axis names), `INSTANCE_SAMPLE_CAP`,
`INSTANCE_SAMPLE_SEED`, `CLAUDE_HAIKU_DESCRIPTIONS_PATH`,
`CLAUDE_HAIKU_SYNTHESIZED_PATH`,
`CLAUDE_FACES_EMBED_DESCRIPTION_PATH`, plus the six output paths.

## 8. Scope discipline

**Out of scope, deferred**:
- Description-based vs response-based embedding methodology
  comparison on the same kaomoji set (could be a follow-up; both
  parquets coexist so it's cheap to add later).
- Per-occurrence within-kaomoji description variance (polysemy
  detection — does `(•̀ᴗ•́)` have distinct usage modes?).
- Re-scrape for richer context from raw `~/.claude/projects` files
  (currently we only have assistant text; tool-use density, prior
  assistant turn, etc. are unused).
- Temporal / `turn_index` dynamics.
- Comparing our cluster *member kaomoji* against eriskii's cluster
  member kaomoji (we don't have access to their assignments).

**Explicit non-goals**:
- No new gemma generations.
- No changes to the v3 trial currently running.
- No edits to existing scripts 06–09.

## 9. Pre-registration

Locked at spec sign-off; changing any of these after the run
invalidates results and requires re-running from script 14:

- `HAIKU_MODEL_ID` in `config.py`.
- `DESCRIBE_PROMPT`, `SYNTHESIZE_PROMPT`, and
  `CLUSTER_LABEL_PROMPT` strings in `llmoji/eriskii_prompts.py`.
- The 21 axis anchor pairs in §4.
- `INSTANCE_SAMPLE_CAP = 4` and `INSTANCE_SAMPLE_SEED = 0`
  (Stage-A sampling), in `config.py`.
- Mask-token spelling (`[FACE]`).

Recorded for git-traceability in the script + module sources, not
just in this spec.

## 10. Why this is in scope ethically

Per the project's Ethics clause: this spec adds zero new model
generations on the gemma side. The Haiku description pass is on
already-collected Claude responses; it doesn't subject any model
to emotionally-disclosing prompts. The mechanistic bridge in §6.3
uses already-stored user text. Total budget: ~436 Haiku calls of
mundane "describe this masked face" work + ~15 Haiku calls for
cluster labeling. No design-before-scale concerns; this is
descriptive reanalysis of existing data.
