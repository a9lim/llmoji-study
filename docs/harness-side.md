# Harness side: contributor-submitted Claude and Codex kaomoji

The harness side replicates [eriskii's Claude-faces
catalog](https://eriskii.net/projects/claude-faces) on a
contributor-submitted corpus. The corpus lives on HuggingFace as
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji); the
companion package [`llmoji`](https://github.com/a9lim/llmoji)
collects the data on the contributor side via Stop hooks and
Haiku synthesis, and uploads pre-aggregated bundles to that
dataset. This repo just pulls the corpus and runs the analysis.

## Prior art: eriskii

eriskii ran the original prompting experiment: configure Claude to
start each message with a kaomoji, log the resulting vocabulary,
and analyze it. They built a 21-axis semantic projection scheme
(warmth, energy, confidence, playfulness, empathy, technicality,
positivity, curiosity, approval, apologeticness, decisiveness,
wryness, wetness, surprise, anger, frustration, hatefulness,
sadness, hope, aggression, exhaustion) and a two-stage Haiku
pipeline for going from raw kaomoji-bearing turns to per-face
descriptions. The published page has a 519-face catalog with 15
KMeans cluster labels and per-axis rankings.

The harness-side pipeline here uses eriskii's anchor scheme (with
one rewrite called out below) and their two-stage Haiku method,
then runs the analysis on the contributor-submitted corpus
instead of a single-author log.

## Pipeline

The whole pipeline is split between the contributor side (the
`llmoji` package, runs locally on each contributor's machine) and
the research side (this repo, reads the aggregated corpus from HF).

Contributor side, in the `llmoji` package:

1. A Stop hook on each installed harness (Claude Code, Codex,
   Hermes) appends one JSONL row per kaomoji-bearing assistant
   turn to a per-machine journal under `~/.<harness>/kaomoji-journal.jsonl`.
   Schema: `ts, model, cwd, kaomoji, user_text, assistant_text`,
   with the leading kaomoji stripped from `assistant_text` and
   sidechain (Task-tool subagent) turns dropped at write time on
   the Claude side.
2. `llmoji analyze` walks the journals, canonicalizes each
   kaomoji form via `llmoji.taxonomy.canonicalize_kaomoji`, samples
   up to 4 instances per face for Stage A (per-instance Haiku
   description with `[FACE]` masking), then runs Stage B
   (per-face synthesis of the 4 descriptions into one canonical
   one-sentence meaning).
3. `llmoji upload --target hf` ships a bundle of
   `(manifest.json, descriptions.jsonl)` to a contributor-named,
   timestamped subfolder under
   `contributors/<32-hex>/bundle-<UTC>/`. The 32-hex is a salted
   hash of a per-machine random token, not an HF account ID.

Research side, in this repo:

4. `scripts/60_corpus_pull.py` snapshot-downloads
   `a9lim/llmoji`, walks every bundle, canonicalizes each kaomoji
   form again (in case contributors have different package
   versions), and pools by canonical form across contributors.
   Output: `data/harness/claude_descriptions.jsonl`, one row per canonical
   form with the union of per-bundle synthesized descriptions
   plus per-contributor counts and provider mix.
5. `scripts/61_corpus_basics.py` prints descriptive stats
   (top kaomoji, contributor and bundle counts, provider mix,
   coverage histogram).
6. `scripts/62_corpus_embed.py` embeds every
   per-bundle synthesized description with
   `sentence-transformers/all-MiniLM-L6-v2`, weighted-means by
   per-bundle count across contributors, L2-normalizes. Output:
   `data/harness/claude_faces_embed_description.parquet`.
7. `scripts/64_eriskii_replication.py` projects onto eriskii's 21
   axes, runs t-SNE plus KMeans(k=15), asks Haiku for short
   cluster labels, and writes the comparison markdown.
8. `scripts/63_corpus_pca.py` runs PCA on the same
   embeddings as a parity-with-eriskii visualization.

## Privacy

The dataset never carries raw user or assistant text. Only the
synthesized descriptions and counts ship. The full privacy model
is in the `llmoji` package's
[SECURITY.md](https://github.com/a9lim/llmoji/blob/main/SECURITY.md);
the dataset card on HF mirrors the relevant tier table.

| Tier | Where | Shipped on `upload`? |
|---|---|---|
| Raw user and assistant text | `~/.<harness>/kaomoji-journal.jsonl` | Never |
| Per-instance Haiku paraphrase | `~/.llmoji/cache/per_instance.jsonl` | Never |
| Overall Haiku summaries and counts | `~/.llmoji/bundle/` | Yes |

## Findings

### Live numbers from the new pipeline

First pull through the HF round-trip (one contributor, 808
emissions, 174 canonical kaomoji):

- Top-20 frequency overlap with eriskii's published top-20 is
  **14/20**. The 15 KMeans cluster themes line up with eriskii's
  15 at the register level (warm-supportive, wry, empathetic,
  sheepish, eager, thoughtful intellectual,
  compassionate-acknowledgment). Direct numeric per-kaomoji
  cluster-membership comparison isn't possible (eriskii's
  per-kaomoji assignments aren't published) but theme-level
  comparison is.
- PCA on the description embeddings: PC1 17.1%, PC2 10.6%
  (top-2 cumulative 27.7%). HDBSCAN finds 2 dense clusters and
  88 noise points at `min_cluster_size=5`; the dense KMeans
  panel is the eriskii-parity reference.
- See `data/harness/eriskii_comparison.md` for the full per-axis writeup
  on the live corpus, and `figures/harness/eriskii_clusters_tsne.png`
  and `figures/harness/claude_faces_pca.png` for the visualizations.

![harness clusters](../figures/harness/eriskii_clusters_tsne.png)

![harness PCA](../figures/harness/claude_faces_pca.png)

### Pre-refactor headline numbers (single-machine local scrape)

For comparison, the pre-refactor pipeline ran on a single-machine
local scrape (my Claude.ai conversation export plus my Claude Code
and Codex journals): 647 emissions across 156 canonical kaomoji,
top-20 overlap 16/20. The two/three-face delta from 16/20 to 14/20
under the new pipeline is partly the corpus growing (174 unique
forms vs 156, mostly from continued use) and partly different
canonicalization timing (contributor-side `llmoji analyze`
canonicalizes before upload; the research-side pull canonicalizes
again on the way in but starts from a slightly different
distribution).

### Pre-refactor per-model and bridge findings (now gone)

The pre-refactor pipeline also produced two analyses that the
HF dataset can't support, because the public dataset pools
per-machine before upload and the `(model, project, user_text)`
fields aren't in the bundle:

- **Per-model axis breakdowns** numerically confirmed eriskii's
  qualitative "opus-4-6 had wider range" claim: mean axis std on
  opus-4-6 was 0.067, opus-4-7 0.066, sonnet-4-6 0.063.
- **Mechanistic surrounding_user → kaomoji axis correlation**
  embedded each `surrounding_user` text on the same 21 axes the
  kaomoji descriptions were projected onto, then computed Pearson
  r per axis. 2/21 axes survived Bonferroni correction at
  α = 0.05/21: surprise (r = +0.20) and curiosity (r = +0.18).
  Affective axes were null. MiniLM on user text picks up
  novelty and unexpectedness, not valence-tracking.

If we want either of these analyses back, the right move is a
separate research-side scrape of a single contributor's local
journal, not adding fields to the public dataset. That's what
`scripts/65_per_project_axes.py` does for the per-project axis
breakdown — see the next section.

### Per-provider per-project axes (single-contributor side script)

`scripts/65_per_project_axes.py` reads `~/.claude/kaomoji-journal.jsonl`
and `~/.codex/kaomoji-journal.jsonl` directly via the
`llmoji.sources.journal` adapter, embeds each emission's
`assistant_text` with MiniLM, projects onto the same 21 eriskii
axes the harness pipeline uses, and groups by `project_slug`
with `min_emissions = 10`. Output goes to
`figures/harness/{claude,codex}/per_project_axes_{mean,std}.png`
and `data/harness/{claude,codex}/per_project_axes.tsv`.

This is local-only and single-contributor: nothing here ships to
HF, and the numbers are specific to one machine's journal. The
public dataset and the headline numbers above are unaffected.

Sample sizes on the current snapshot:

| provider | usable emissions | projects with n≥10 |
| --- | ---: | ---: |
| claude (Claude Code) | 644 | 14 |
| codex | 62 | 3 |

#### Claude side

`figures/harness/claude/per_project_axes_mean.png`. Top and
bottom axis per project, by signed mean projection:

| project | n | top axis | bottom axis |
| --- | ---: | --- | --- |
| saklas | 164 | curiosity (+0.058) | exhaustion (−0.047) |
| llmoji | 147 | curiosity (+0.065) | exhaustion (−0.053) |
| a9lim.github.io | 64 | curiosity (+0.060) | exhaustion (−0.046) |
| kenoma | 45 | curiosity (+0.060) | exhaustion (−0.063) |
| Work | 43 | curiosity (+0.061) | surprise (−0.059) |
| faithful | 37 | curiosity (+0.059) | exhaustion (−0.065) |
| rlaif | 29 | curiosity (+0.068) | exhaustion (−0.063) |
| llmoji-study | 26 | curiosity (+0.044) | exhaustion (−0.052) |
| shoals | 23 | approval (+0.052) | surprise (−0.100) |
| hylic | 20 | curiosity (+0.075) | aggression (−0.066) |
| claudedriven | 13 | curiosity (+0.068) | warmth (−0.083) |
| brie | 12 | positivity (+0.068) | technicality (−0.090) |
| geon | 11 | curiosity (+0.057) | exhaustion (−0.077) |
| a9lim | 10 | anger (+0.051) | surprise (−0.051) |

Aggregate reads:

- **Curiosity dominates.** Top axis on 12 of 14 claude projects;
  the two exceptions are `shoals` (worldbuilding, top=approval)
  and `brie` (top=positivity). On this contributor's machine,
  Claude reads as overwhelmingly inquiry-mode in response.
- **Exhaustion is the floor on the load-bearing repos.** saklas,
  llmoji, kenoma, faithful, rlaif, llmoji-study, geon, and
  a9lim.github.io all have exhaustion as the most-negative axis.
  Low exhaustion in the response embedding corresponds to
  energetic engagement — bigger and more open-ended projects pull
  this signal harder.
- **Outliers worth zooming on.**
  - `brie` (n=12): hardest negative cell on the whole heatmap,
    technicality −0.090, with positivity as the top. Small
    project, low-tech, friendly tone.
  - `shoals` (n=23): surprise floor at −0.100. Worldbuilding
    work where claude responses settle into expected texture
    rather than registering surprise — consistent with
    canon-maintenance work.
  - `claudedriven` (n=13): warmth floor at −0.083. Clinical
    register, fitting a project that's literally about driving
    Claude rather than collaborating.
  - `llmoji-study` (n=26): bottom = confidence (−0.052) [versus
    most other projects, where confidence is mid-pack and
    exhaustion is the floor]. The research repo where pre-
    registration and the don't-bluff anchor enforce hedging
    shows up as low-confidence in the response embedding too.

The values are all small in absolute terms (±0.1 cosine
projection on a normalized embedding); think shoulder-of-curiosity
differences rather than strong contrasts.

#### Codex side

`figures/harness/codex/per_project_axes_mean.png`. Sample is
much smaller (62 emissions across 3 projects clearing n=10):

| project | n | top axis | bottom axis |
| --- | ---: | --- | --- |
| saklas | 35 | curiosity (+0.040) | technicality (−0.063) |
| website | 14 | curiosity (+0.080) | approval (−0.047) |
| kenoma | 13 | curiosity (+0.046) | exhaustion (−0.067) |

Curiosity is still the top axis on all three. Exhaustion only
shows up as the floor on `kenoma`; the other two land
elsewhere.

#### Cross-provider divergence

The two providers overlap on `saklas` and `kenoma` with
sufficient N for a soft compare:

- `saklas`: claude bot = exhaustion (−0.047, n=164), codex bot
  = technicality (−0.063, n=35). Same project, two different
  model families, two different bottom-axis reads. The numbers
  are close enough in magnitude that this could partly be
  ranking noise (codex's exhaustion projection on saklas is also
  negative, just less so than its technicality projection).
- `kenoma`: claude bot = exhaustion (−0.063), codex bot =
  exhaustion (−0.067). Agreement at the floor.

[TBD] more careful cross-provider read. Codex N is 62 emissions
across 3 projects, all of which I'm familiar with; claude N is
644 across 14, also all familiar. The "saklas reads as more
technical-floor on codex than on claude" signal could mean Codex
spends its responses on saklas in more code/data-heavy register
than Claude does, or it could be a 35-row sampling artifact.
I haven't done a side-by-side read of the actual response text
on saklas split by provider yet; that would settle whether the
divergence is real. Worth flagging when codex sample size grows.

#### Caveats

- Response-based embedding, not user-prompt or kaomoji-glyph.
  This measures the texture of Claude's and Codex's *responses*
  on each project, projected onto eriskii axes. It is not a
  measure of the work itself or of the user's framing.
- `MIN_TEXT_LEN = 20` drops emissions where `assistant_text` is
  empty or kaomoji-only (~11% of total). Those rows would have
  no semantic content for MiniLM to embed.
- Per-project axis means are small in absolute terms (cosine
  projections of L2-normalized 384-d embeddings onto unit-norm
  axis vectors; typical magnitude ±0.1). Within-figure contrasts
  are real, but don't read these as full-scale eriskii rankings.
- The eriskii anchor pairs were locked for the harness pipeline.
  Same axes here, no anchor changes for this side script.

### Multi-contributor numbers

Numbers from a multi-contributor pull will land here once the
dataset has more bundles. Per-contributor pooling already
happens in `llmoji analyze`, so each row in a contributor's
`descriptions.jsonl` is already pooled across that machine's
instances of the face; the research-side pull just sums counts
and unions the descriptions across contributors.

## Eriskii anchor rewrites

a9 rewrote one anchor pair from eriskii's original scheme. The
others are kept as-is.

- **`wetness ↔ dryness`**: eriskii used the bare strings as a
  "three seashells" joke (intentionally undefined). Our anchor
  reads "waxing poetic about emotions, lyrical and self-expressive"
  on the positive side and "helpful assistant tone, task-focused,
  businesslike, practical, matter-of-fact" on the negative side.
  Wetness rankings are accordingly more meaningful than eriskii's
  but not directly comparable.

The other 20 anchors are 4-word lexical pairs straight from the
eriskii.net page; the full list lives in
[`llmoji_study/eriskii_anchors.py`](../llmoji_study/eriskii_anchors.py).

## Caveats and known limitations

- **Per-machine pooling already happened.** Each row in a
  contributor's bundle is already pooled across that machine's
  instances of the face. A single heavy contributor can swing a
  face's rank. The corpus is small (n<1K canonical kaomoji as of
  writing) so this is worth watching.
- **Provider mix varies across bundles.** Some bundles are
  Claude Code, others Codex, some are mixed. The manifest's
  `providers_seen` has the per-bundle breakdown but not per-row
  attribution within a bundle.
- **Haiku is the synthesizer for every row.** Researchers
  wanting to compare against a different summarizer should re-run
  the per-instance Haiku step locally (the per-instance
  paraphrases never ship to HF).
- **Eriskii's per-kaomoji cluster assignments are not public**;
  comparison with their 15 KMeans clusters is theme-level only.
- **The mask token `[FACE]` is sometimes referenced literally**
  in Haiku descriptions. Stage-B synthesis usually corrects for
  this but a few descriptions retain artifacts.
- **Counts are per-machine, not global.** Be careful when summing
  across contributors; someone running `llmoji` for two months
  will have very different data than someone running it for two
  days.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

export ANTHROPIC_API_KEY=...    # Haiku cluster-labeling

python scripts/60_corpus_pull.py            # snapshot a9lim/llmoji into data/harness/hf_dataset/
python scripts/61_corpus_basics.py     # printout: top kaomoji, providers, contributors
python scripts/62_corpus_embed.py  # per-canonical embeddings
python scripts/64_eriskii_replication.py       # axes, clusters, narrative writeup
python scripts/63_corpus_pca.py          # PCA panel
```

If you want to contribute to the dataset rather than just
consume it, see the
[`llmoji` package README](https://github.com/a9lim/llmoji/blob/main/README.md)
and the
[dataset card on HF](https://huggingface.co/datasets/a9lim/llmoji).
