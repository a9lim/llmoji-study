# Harness side: contributor-submitted Claude and Codex kaomoji

The harness side runs analyses on a contributor-submitted corpus of
kaomoji emissions. The corpus lives on HuggingFace as
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji); the
companion package [`llmoji`](https://github.com/a9lim/llmoji)
collects the data on the contributor side via Stop hooks, runs a
structured Haiku synthesis pass that commits each face's affect /
stance / modality picks to a locked 48-word LEXICON, and uploads
pre-aggregated bundles to that dataset. This repo just pulls the
corpus and runs the analysis.

## Prior art: eriskii

eriskii ran the original prompting experiment: configure Claude to
start each message with a kaomoji, log the resulting vocabulary,
and analyze it. Their writeup at
[eriskii.net/projects/claude-faces](https://eriskii.net/projects/claude-faces)
built a 21-axis MiniLM-on-prose semantic projection (warmth, energy,
confidence, playfulness, …) and a two-stage Haiku pipeline that
went raw kaomoji-bearing turns → per-instance description → per-face
synthesis description, then ran KMeans(k=15) on the embeddings for
register clustering. The published page has a 519-face catalog with
per-axis rankings.

This repo's harness side **previously** ran a parity replication on
that scheme — `scripts/harness/62_corpus_embed.py` MiniLM'd the
synthesized prose, `scripts/harness/64_eriskii_replication.py`
projected onto the 21 axes, `scripts/harness/65_per_project_axes.py`
embedded `assistant_text` from per-machine journals onto the same
axes for per-project register breakdowns. **All three are gone as of
2026-05-06.** They were replaced by the bag-of-lexicon (BoL) pipeline
described below; the eriskii framing survives as motivation but its
21-axis projection is no longer the harness representation.

The pre-refactor numbers from the eriskii-parity era (16/20 → 14/20
top-20 frequency overlap; 15-cluster theme alignment; per-project
axis means with curiosity dominating and exhaustion as the floor on
load-bearing repos; saklas / kenoma cross-provider divergence) are
preserved in [`previous-experiments.md`](previous-experiments.md)
"Eriskii-parity harness pipeline (replaced 2026-05-06 by BoL)".

## Pipeline (post-2026-05-06)

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
   read-through with `[FACE]` masking), then runs Stage B — a
   **structured synthesis pass** that asks Haiku to commit, per
   face per bundle, a pick from the locked 48-word v2 LEXICON: 1–3
   `primary_affect` words (the Russell-circumplex-tagged subset)
   plus 3–5 `stance_modality_function` words (the extension axes).
   Pre-2026-05-02 (llmoji v1.x) this stage produced a free-form
   prose "synthesis description" instead. v2 bundles ship the
   structured `synthesis` object alongside the row's count metadata.
3. `llmoji upload --target hf` ships a bundle of
   `(manifest.json, <sanitized-source-model>.jsonl)` to a
   contributor-named, timestamped subfolder under
   `contributors/<32-hex>/bundle-<UTC>/`. The 32-hex is a salted
   hash of a per-machine random token, not an HF account ID.

Research side, in this repo:

4. `scripts/harness/60_corpus_pull.py` snapshot-downloads
   `a9lim/llmoji`, walks every bundle, canonicalizes each kaomoji
   form again (in case contributors have different package
   versions), and pools by canonical form across contributors and
   source models. Output: `data/harness/claude_descriptions.jsonl`,
   one row per canonical face with per-bundle `synthesis` objects,
   per-bundle counts, source-model and provider mix. Legacy v1.x
   bundles still load (`source_model = "_pre_1_1"`) but their
   free-form `synthesis_description` field is unused downstream.
5. `scripts/harness/61_corpus_basics.py` prints descriptive stats
   (top kaomoji, contributor and bundle counts, provider mix,
   coverage histogram, per-source-model emissions / faces).
6. `scripts/harness/62_corpus_lexicon.py` builds the **bag-of-
   lexicon (BoL) parquet** at
   `data/harness/claude_faces_lexicon_bag.parquet`. For each
   canonical face, count-weighted-pools every per-bundle
   `synthesis` pick into a 48-d L1-normalized soft distribution
   over the lexicon (`bol_from_synthesis` → `pool_bol` in
   `llmoji_study.lexicon`). 19 of those 48 words carry explicit
   Russell-quadrant tags, so collapsing the BoL onto its
   circumplex slots and renormalizing gives a 6-d quadrant
   distribution per face — no encoder, no projection, no post-hoc
   inference. The parquet is `lexicon_version`-stamped;
   `assert_lexicon_v1` hard-fails consumers if the LEXICON ever
   rotates.
7. `scripts/harness/64_corpus_lexicon_per_source.py` builds the
   long-format per-(face, source_model) variant
   (`claude_faces_lexicon_bag_per_source.parquet`) for cross-source
   register comparison. Same builder, no count-pooling across
   source models.
8. `scripts/harness/63_corpus_pca.py` runs PCA on the 48-d BoL
   plus KMeans(k=15) for register clustering. Cluster labels are
   deterministic top-modal-lexicon-word strings (no Haiku call) —
   the cluster IS its lexicon-word signature.
9. `scripts/harness/55_bol_encoder.py` writes a face_likelihood-
   shaped TSV (`data/harness/face_likelihood_bol_summary.tsv`) so
   BoL plugs into the existing 52 / 53 / 54 ensemble pipeline as
   another encoder column — soft 6-quadrant distribution per face,
   compared against Claude-GT via JSD just like the LM-head
   encoders.
10. `scripts/harness/50_face_likelihood.py --model {haiku,opus}`
    runs the **introspection face_likelihood pass** against the
    Anthropic API. Shows each canonical face out of context and
    asks for the affective state it induces — schema v2,
    likelihoods only (the `top_pick` / `reason` / `temperature=0`
    fields were dropped 2026-05-05; the latter per-model — opus
    4.7 deprecated `temperature=0`). Prompt v4 reframes the task
    as introspection on felt state to avoid visual-feature
    priming. Output:
    `data/harness/face_likelihood_{haiku,opus}_summary.tsv`.
11. `scripts/harness/68_three_way_analysis.py` and
    `scripts/harness/69_per_source_drift.py` are the diagnostic
    layer — see "Use / read / act" below.

## Use / read / act — three channels

The 2026-05-06 framing reads three structurally different
measurements of "what does this face mean":

- **use** (Claude-GT): per-face per-quadrant emission counts under
  Russell-prompted elicitation, from the
  `scripts/harness/00_emit.py` pilot. Reads as
  `P(face | prompt-quadrant)`; inverting per-face gives a posterior
  over prompt-quadrants. `claude_gt.load_claude_gt_distribution()`
  is the canonical loader.
- **read** (Opus / Haiku face_likelihood): cold introspection on
  each face symbol with no surrounding context. Measures the face's
  *denoted meaning* independent of how it gets used.
- **act** (BoL): pooled structured synthesis over many in-context
  wild emits. Measures *what affective state the face's
  deployment context expresses, summarized by Haiku*.

Three measurements diverge in patterned ways. Headline: opus ↔
haiku introspection cross-similarity is 0.906 invariant under
emit-weighting (model size doesn't matter for cold symbolic
interpretation). gt ↔ introspection goes UP under emit-weighting;
gt ↔ bol goes DOWN. The `110` agreement cell (opus + haiku read GT;
BoL diverges) covers 27.4% of emit volume on the shared n=40 face
subset.

The interpretive read on BoL has been hedged: the original framing
treated BoL as a deployment-state ground truth, but a parsimonious
counter-hypothesis is that Haiku-as-synthesizer is positivity-biased
on negative-affect contexts, so BoL whitewashes LN/HN-coded
deployment states into LP descriptors. **For deployment
interpretation of negative-affect faces, prefer GT or Opus
introspection over BoL when they disagree.** The use/act gap
remains a real observation; what shifted is whether to read it as
"deployment context redefines symbol meaning" or "synthesizer
positivity bias artifact." Detail and case files in
[`2026-05-06-use-read-act-channels.md`](2026-05-06-use-read-act-channels.md).

## Privacy

The dataset never carries raw user or assistant text. Only the
structured `synthesis` picks and counts ship. The full privacy model
is in the `llmoji` package's
[SECURITY.md](https://github.com/a9lim/llmoji/blob/main/SECURITY.md);
the dataset card on HF mirrors the relevant tier table.

| Tier | Where | Shipped on `upload`? |
|---|---|---|
| Raw user and assistant text | `~/.<harness>/kaomoji-journal.jsonl` | Never |
| Per-instance Haiku read-through | `~/.llmoji/cache/per_instance.jsonl` | Never |
| Structured synthesis picks + counts | `~/.llmoji/bundle/` | Yes |

The structural switch from prose synthesis (v1.x) to
LEXICON-constrained picks (v2.0+) tightens the privacy story: the
shipped object is a count over 48 fixed words, not natural-language
text that could carry incidental identifying information from the
surrounding prompt.

## Findings

### BoL geometry (live)

`scripts/harness/63_corpus_pca.py` PCAs the 309-face BoL parquet.
Two panels:

1. PC1 vs PC2 colored by **inferred Russell quadrant** from the
   BoL's circumplex slots (`bol_modal_quadrant`). The per-face
   color is the synthesizer's structured commit — no encoder, no
   projection on top of a projection.
2. PC1 vs PC2 with KMeans(k=15) labeled by the **top-2 modal
   lexicon words** of each cluster. Deterministic, reproducible,
   fully interpretable (no Haiku call to label clusters in
   natural language).

Output: `figures/harness/claude_faces_pca.png`.

### BoL as a face_likelihood encoder

Solo similarity vs Claude-GT (n=40 strict-Claude-only floor=3
shared face subset, 9 encoders): **0.549 face-uniform / 0.455
emit-weighted** (rank 6/9 face-uniform, dead last 9/9 emit-
weighted). The face-uniform-vs-emit-weighted **inversion** (BoL
gets long-tail faces *better* than top-emitted ones) is consistent
with the whitewashing hypothesis — heavily-emitted modal faces
are exactly where the per-context summarization happens most, so
positivity bias accumulates fastest there.

BoL is not additive over the top solo encoders in the best
ensemble. The current best deployment ensemble on the broader
pooled-GT n=54 subset is `{gemma, gemma_v7primed, ministral,
opus}` at 0.904 emit-weighted / 0.832 face-uniform; BoL doesn't
make that cut. **The encoder still ships** because (a) it's
zero-cost to compute, (b) the inversion it produces is itself an
informative signal about where synthesizer-bias hits hardest, and
(c) the cross-source-model BoL drift surfaces real per-deployment-
register patterns even if the absolute readings are biased. Detail
in [`findings.md`](findings.md) and
[`2026-05-06-use-read-act-channels.md`](2026-05-06-use-read-act-channels.md).

### Per-source-model drift

`scripts/harness/69_per_source_drift.py` splits BoL by source
model. 491 (face, source_model) cells across 8 sources, 112 faces
appear under ≥2 sources. Per-source vs Claude-GT face-uniform
similarity ranges from **0.550** (codex-hook) and **0.525**
(claude-opus-4-7) at the top to **0.058** (gpt-5.4) at the
bottom. Cross-source pairwise: claude-opus-4-7 ↔ codex-hook
**0.630 mean similarity / 59% modal agreement** — the strongest
cross-source agreement isn't claude-vs-claude (claude-opus-4-7 ↔
claude-opus-4-6 = 0.566) but **coding-agent-deployment vs
coding-agent-deployment**. The shared register is the deployment
shape, not the model identity.

Caveat: all BoL synthesis goes through Haiku, so per-source
comparisons measure how Haiku reads each provider's surrounding-
text style, not what each provider's model "thinks." The genuine
provider-shape effect and the Haiku-reads-different-prose-styles
effect both produce the same observable. Outputs:
`data/harness/per_source_drift.tsv` +
`figures/harness/per_source_modal_heatmap.png`.

### Per-project resolver (cross-platform, contributor-side)

`scripts/66_per_project_quadrants.py` resolves per-project kaomoji
emissions to Russell quadrants via three modes: `gt-priority`
(Claude-GT first, BoL fallback), `bol` (BoL for every face),
`gt-only` (strict; mark unresolved as `unknown`). The script reads
local journals + claude.ai exports as deployment-emission sources
and is intended to be run **locally** by individual contributors;
its rendered outputs (per-(project, quadrant) tables, per-project
charts) are deployment-telemetry by construction and are not
committed to this repo. The methodology is the contribution; the
per-machine outputs stay private.

Coverage on the 2026-05-06 expanded GT corpus is **~67% direct GT
+ ~33% BoL fallback** under `gt-priority` (100% combined); ~33%
unknown under strict `gt-only`. These are coverage numbers across
the corpus's face distribution, not specific to any contributor.

### Wild-emit residual analysis

`scripts/67_wild_residual.py` clusters the canonical-kaomoji
HF-corpus faces in 48-d BoL space. The k=6 clustering surfaces
sub-cluster structure beyond the six Russell quadrants, with
deterministic top-2 modal-lexicon-word labels per cluster. The
cluster-summed shares typically split LP-heavy positive register
(relieved / satisfied / peaceful) vs HP-coded energetic register
(excited / triumphant) vs an HN-coded cluster (frustrated /
self-correcting). **HN-S vocabulary is more diverse in the wild
corpus than in the Russell-elicited GT** — the under-sampling
argument from the GT side survives the corpus refresh.

The script's 3D PCA HTML chart uses two channels per face: color =
BoL modal quadrant (uniformly across all faces — surfaces the
use/act gap rather than collapsing to GT), marker shape =
deployment surface (Claude Code only / any claude.ai / neither).
The surface dispatch reads local-machine emission sources, so the
rendered HTML is contributor-specific deployment telemetry —
gitignored, regenerate locally. The cluster-table TSVs
(`data/harness/wild_residual_clusters{,_gt_only}.tsv`) are
corpus-derived and safe to commit.

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
- **Haiku is the synthesizer for every row.** The structural
  positivity-bias concern (see "Use / read / act" above) is a
  consequence; the falsifiable test is re-synthesizing a sample
  with Opus and auditing whether the LP-skew on negative-affect
  contexts persists.
- **Lexicon version is locked to v1.** All BoL parquets are
  stamped with `LEXICON_VERSION=1`. v3 lexicon rotation will
  hard-fail consumers via `assert_lexicon_v1` — no silent version
  mixing across analyses.
- **Counts are per-machine, not global.** Be careful when summing
  across contributors; someone running `llmoji` for two months
  will have very different data than someone running it for two
  days.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

export ANTHROPIC_API_KEY=...    # face_likelihood Haiku/Opus passes

python scripts/harness/60_corpus_pull.py       # snapshot a9lim/llmoji
python scripts/harness/61_corpus_basics.py     # printout: top kaomoji, providers, contributors
python scripts/harness/62_corpus_lexicon.py    # 48-d BoL parquet
python scripts/harness/64_corpus_lexicon_per_source.py  # per-(face, source_model) BoL
python scripts/harness/63_corpus_pca.py        # PCA + KMeans cluster panel
python scripts/harness/55_bol_encoder.py       # BoL → face_likelihood TSV
python scripts/harness/50_face_likelihood.py --model haiku   # introspection encoder
python scripts/harness/50_face_likelihood.py --model opus    # introspection encoder
python scripts/harness/68_three_way_analysis.py             # use/read/act per-face
python scripts/harness/69_per_source_drift.py               # per-source case files
```

If you want to contribute to the dataset rather than just consume
it, see the
[`llmoji` package README](https://github.com/a9lim/llmoji/blob/main/README.md)
and the
[dataset card on HF](https://huggingface.co/datasets/a9lim/llmoji).
