# llmoji-study

Does a language model's choice of kaomoji track something about its
internal state? Claude is often asked to begin each message with a
kaomoji that reflects how it currently feels, and the question
naturally follows: is that choice actually coupled to whatever's
going on inside the model, or is it surface statistics with
emotional-sounding tokens mixed in? This repo answers the question
from two angles. The local side runs probes and activation steering
on open-weight causal LMs via [`saklas`](https://github.com/a9lim/saklas),
where I can read and intervene on the hidden state directly. The
harness side does an [eriskii](https://eriskii.net/projects/claude-faces)-style
semantic-axis replication on real contributor-submitted Claude and
Codex kaomoji, pulled from the
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji)
HuggingFace dataset.

> **Companion package**: the contributor-side data collection
> (per-harness Stop hooks, kaomoji journals, Haiku synthesis,
> bundle-and-upload CLI) is the
> [`llmoji`](https://github.com/a9lim/llmoji) PyPI package. This
> repo doesn't scrape any local data; it pulls the aggregated
> corpus from the HF dataset.
>
> **Prior art**: [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces)
> is the original post that came up with the idea of prompting
> Claude with kaomoji and analyzing the resulting vocabulary.
> The harness-side replication here uses eriskii's 21 semantic
> axes and their two-stage Haiku pipeline.
>
> **Writeup**: [Introspection via Kaomoji](https://a9l.im/blog/introspection-via-kaomoji)
> is a blog-format walkthrough of the local-side findings, with
> interactive 3D scatter plots and the cross-model alignment
> argument. The post is the human-readable companion; this repo
> is the artifact + reproduction notes.

## How this is organized

The two sides are independent enough that they live in their own
docs:

- [`docs/local-side.md`](docs/local-side.md): probes, steering, and
  hidden-state analysis on `gemma-4-31b-it`, `Qwen3.6-27B`, and
  `Ministral-3-14B-Instruct-2512`. Pilots v1, v2, v3.
- [`docs/harness-side.md`](docs/harness-side.md): eriskii-replication
  on the contributor-submitted Claude and Codex corpus. Pulls from
  the HF dataset, embeds Haiku-synthesized per-face descriptions,
  projects onto 21 semantic axes, runs t-SNE plus KMeans clustering.

Engineering notes, gotchas, and the design and plan docs live in
[`CLAUDE.md`](CLAUDE.md) and [`docs/`](docs/).

## Headline findings

### Local side

Steering on `gemma-4-31b-it` is a clean causal handle on kaomoji
choice. In pilots v1 and v2, steering on `happy.sad` collapses the
emitted distribution: 0% happy-labeled kaomoji under sad-steering,
100% under happy-steering, with 71% in the unsteered middle. The
shift is monotonic and the effect is selective to the targeted axis
(orthogonal probes barely move). Within the unsteered arm, however,
the probe scalar at token 0 only weakly predicts which kaomoji the
model emits, because saklas's bundled `happy.sad` and `angry.calm`
probes both extract the same lexical-valence direction (the v1 and
v2 valence-collapse).

Pilot v3 (naturalistic, no steering, hidden-state space instead of
probe space) recovers the second affective dimension that the
probes miss. 800 generations balanced across the five Russell
quadrants, replicated on three models from three labs:
gemma-4-31b-it (Google, 31B), Qwen3.6-27B (Alibaba, 27B), and
Ministral-3-14B-Instruct-2512 (Mistral, 14B). Same v3 prompts, same
probe set, same analysis chain.

Canonical hidden-state aggregate is `h_first` (the kaomoji-emission
state, methodology-invariant across the 2026-05-02 MAX_NEW_TOKENS
cutover). Russell-quadrant silhouette over PCA(2) coordinates peaks
at gemma **L50 (0.413)**, qwen **L59 (0.420)**, ministral **L20
(0.199)** under the post-2026-05-03 cleanliness + seed-0 cache fix
data — gemma + qwen sit at the deep half of their networks,
ministral is mid-depth. The pre-cleanliness numbers (0.235 / 0.244 /
0.149) were inflated downward by category-bleed in the prompt set
and (on qwen) by a saklas cache-prefix bug in pilot seed 0.

The three Russell circumplexes are geometrically congruent. Triplet
Procrustes alignment of per-quadrant PCA(2) centroids onto gemma
(post-fix): qwen PC1×PC2 residual 6.9, ministral 23.0 (after a
+157° axis flip that's PCA sign indeterminacy, not a divergence
finding). Pairwise CKA at preferred layers: gemma↔qwen 0.795,
gemma↔ministral 0.741, qwen↔ministral 0.812 — the qwen↔ministral
pair actually exceeds the gemma↔qwen baseline. A single canonical
alignment layer at ministral L20 maximizes CKA against either
reference model regardless of where the partner's affect
representation sits.
Vocabulary diverges sharply (gemma 32 canonical forms, qwen 65,
ministral 196 with heavy `(◕‿◕✿)` / `(╯°□°)` / emoji-augmented
variants — a francophone-internet register, possibly), but the
geometry under those faces matches across all three.

The bundled saklas probes differ structurally between models:
cross-face Pearson r between mean `happy.sad` and `angry.calm` is
−0.94 on gemma (probe-space valence-collapse) and −0.12 on Qwen
(near-orthogonal), because saklas's contrastive probes are
extracted at saklas's own internal layer and inherit each model's
probe-space layout, not the hidden-state layout the preferred-layer
PCA is reading.

A tokenizer bug in HF-distributed Mistral checkpoints (mis-splits
~1% of tokens after apostrophes / punctuation) was found and fixed
in saklas 2.0.0; ministral pilot ran on the buggy version, main
run uses the fix.

Rule 3 of the v3 cross-model gating rules (originally a
`powerful.powerless` HN−LN sign-check) was redesigned 2026-05-01
because HN aggregates anger (high PAD dominance) with fear (low
PAD dominance), washing out the within-quadrant mean. New schema:
HN bisects into HN-D (anger / contempt) and HN-S (fear / anxiety)
via a `pad_dominance` field on `EmotionalPrompt`. After the
2026-05-03 cleanliness pass the prompt set is 20 D / 20 S balanced
(160 rows per group per model), no untagged-HN. On the
cleanliness + seed-0-fix data, **rule 3b (`fearful.unflinching`
mean(HN-S) > mean(HN-D)) is WEAK** — gemma t0 d=+1.60 PASS but
tlast/mean CI-ambiguous (verdict: mid); qwen t0 d=+2.14 PASS but
tlast/mean wrong-direction d≈−0.36 with CI excludes 0 (verdict:
fail); ministral PASS on all 3 aggregates with mean d=+0.55. The
2026-05-01 "PASS on all 3" headline reflected pre-cleanliness data
with cache-contaminated qwen seeds; the cleaner data shows the
cross-model dominance signal is meaningful on ministral and partial
on gemma but breaks down on qwen at later tokens. Rule 3a
(`powerful.powerless`) remains dropped — wrong-direction on most
aggregates × all three models. The fear-axis signal lives at t0
across all three (where the kaomoji is emitted), where qwen still
has the largest within-model effect (d=+2.14).

The v3 prompt set was rewritten end-to-end 2026-05-03 in the prompt
cleanliness pass (`docs/2026-05-03-prompt-cleanliness.md`) — 120
prompts (20 per category) replacing the prior 123, HN cleanly
bisected into 20 HN-D + 20 HN-S with no untagged entries, IDs
renumbered hn01–hn40. Full N=8 rerun (8 seeds × 120 × 3 models =
2880 generations) landed 2026-05-03 along with a seed-0 cache-mode
fix that removed pilot-vs-rerun KV-state contamination from
seed 0 (worst on qwen at 37–46% per-row L2 deviation — see
`docs/2026-05-03-cleanliness-pilot.md` for the postmortem).

Full setup, decision rules, per-quadrant centroids, all the
cross-model comparisons, the Ministral pilot + main + rule-3
redesign details are in
[`docs/local-side.md`](docs/local-side.md) and
[`docs/findings.md`](docs/findings.md).

### Harness side

The contributor-submitted corpus on
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) holds
one row per (machine, kaomoji) pair, where each row is a
Haiku-synthesized one-sentence meaning aggregated across that
machine's instances of the face. The research-side pipeline pulls
the dataset, pools by canonical kaomoji form across contributors,
embeds the synthesized descriptions with `all-MiniLM-L6-v2`, and
projects onto eriskii's 21 semantic axes plus runs t-SNE with
KMeans clustering and Haiku-synthesized cluster labels.

First pull through the new pipeline (one contributor, n=808
emissions, 174 canonical kaomoji): top-20 frequency overlap with
eriskii's published vocabulary is 14/20, and the 15 KMeans cluster
themes line up with eriskii's 15 at the register level
(warm-supportive, wry, empathetic, sheepish, eager, thoughtful).
The `wetness` axis is a9's rewrite of eriskii's
intentionally-undefined `wetness ↔ dryness` joke; rankings on that
axis are more meaningful than eriskii's but not directly comparable.
Per-model and per-project axis breakdowns and the
`surrounding_user → kaomoji` mechanistic-bridge correlation are
gone in the HF refactor (the public dataset pools per-machine
before upload, so the per-row metadata those analyses needed isn't
available). Pre-refactor those analyses confirmed eriskii's
qualitative "opus-4-6 had wider range" claim numerically and showed
that surprise (r = +0.20) and curiosity (r = +0.18) were the only
two of 21 axes where MiniLM on user text correlated with kaomoji
projection past Bonferroni; the historical numbers are in
[`docs/harness-side.md`](docs/harness-side.md). Multi-contributor
numbers will land here as more bundles arrive.

Full pipeline, methodology, axis anchors, and the historical
pre-refactor cross-cuts are in
[`docs/harness-side.md`](docs/harness-side.md).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .  # pulls llmoji>=1.0,<2 from PyPI plus saklas, sentence-transformers, ...
```

Scripts split into local (probes, hidden state, v3 follow-ons) and
harness (contributor corpus): `scripts/local/` and `scripts/harness/`.

For the local side, set `LLMOJI_MODEL=gemma|qwen|ministral` and run
`scripts/local/00_emit.py` (the v3 800-generation runner).
See `CLAUDE.md` § Commands for the full script chain.

For the harness side, you need an `ANTHROPIC_API_KEY` (the cluster
labeler calls Haiku) and the HF Hub Python client (the install
above pulls `huggingface_hub`). Anonymous reads of the public
dataset are fine, so `HF_TOKEN` is optional.

```bash
python scripts/harness/60_corpus_pull.py            # snapshot a9lim/llmoji into data/harness/hf_dataset/
python scripts/harness/61_corpus_basics.py     # printout: top kaomoji, providers, contributors
python scripts/harness/62_corpus_embed.py  # per-canonical embeddings
python scripts/harness/64_eriskii_replication.py       # axes, clusters, writeup
python scripts/harness/63_corpus_pca.py          # PCA panel
```

## Related

- [`saklas`](https://github.com/a9lim/saklas): the engine. Activation
  steering and trait monitoring on HuggingFace causal LMs via
  contrastive-PCA. The local side is a study built on top of saklas.
- [`llmoji`](https://github.com/a9lim/llmoji): the PyPI package
  that runs Stop hooks on coding agents (Claude Code, Codex,
  Hermes), keeps a per-machine kaomoji journal, runs the two-stage
  Haiku synthesis, and uploads the result to the shared corpus.
- [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji): the
  HF dataset. Contributor-submitted kaomoji counts and synthesized
  meanings, CC-BY-SA-4.0.
- [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces):
  the prior art for the kaomoji-cataloging idea, the 21-axis
  projection scheme, and the two-stage Haiku pipeline.
- [Introspection via Kaomoji (a9l.im)](https://a9l.im/blog/introspection-via-kaomoji):
  blog-format writeup of the local-side findings, with interactive
  3D scatter plots and the cross-model alignment argument.

## Findings summary

A condensed read of what the local-side experiments have shown so far.
Numbers cite the v3 follow-on analyses (`scripts/local/21`–`scripts/local/31`)
run on the 960-generation gemma + qwen + ministral sidecars under
the post-2026-05-03 cleanliness + seed-0-fix data + `h_first`
standardization. Full details and
gotchas live in [`CLAUDE.md`](CLAUDE.md),
[`docs/findings.md`](docs/findings.md), [`docs/local-side.md`](docs/local-side.md),
and [`docs/gotchas.md`](docs/gotchas.md); headline figures and
interactive 3D HTMLs are in
[`figures/local/`](figures/local/).

### 1. Affect peaks at the model-specific preferred layer, not the deepest

Each per-row hidden-state sidecar stores per-layer aggregates, so
layer-wise structure is recoverable without re-running the model.
Russell-quadrant silhouette over PCA(2) at `h_first` (the
kaomoji-emission state) by probe layer:

- **gemma** (56 layers): silhouette peaks at **L50 (0.413)**, top-5
  layers cluster L47–51 (~84–91% depth, plateau not single peak).
- **qwen** (60 layers): peaks at **L59 (0.420)**, top-5 layers L54–59
  (~88–97% depth) — explicit near-deepest plateau.
- **ministral** (37 layers): peaks at **L20 (0.199)**, top-5 layers
  L20–26 (~54–70% depth) — the only mid-depth model.

The repo stores `preferred_layer` on `ModelPaths`: gemma 50, qwen
59, ministral 20. v3 figures default to those layers.

These are the post-2026-05-03 cleanliness + seed-0-fix numbers.
The pre-cleanliness numbers (gemma 0.235 / qwen 0.244 / ministral
0.149) were inflated downward by category-bleed in the v3 prompt
set and (on qwen) by a saklas cache-prefix bug in pilot seed 0.
The pre-cutover `h_mean` numbers were even smaller (gemma 0.116,
qwen 0.116, ministral 0.045) under L28 / L38 / L21. `h_first` is
methodology-invariant across the MAX_NEW_TOKENS=120→16 cutover (the
kaomoji-emission state doesn't depend on how long generation runs
after it). The earlier "gemma mid-depth, qwen deep" framing
dissolved with the cutover — gemma + qwen are now both deep,
ministral is the only mid-depth model.

### 2. Kaomoji is a partial readout, not the state itself

For each face emitted in two or more quadrants with at least three
rows in each, train a PCA(20)→l2-logistic classifier on hidden state
at the preferred layer to predict which quadrant prompted each
instance, using only that face's rows. Compare 5-fold stratified
CV accuracy to a 30-shuffle label-permutation null at q95.

- **gemma** (10 cross-quadrant emitters): 6/10 separate.
  `(｡•́︿•̀｡)` (n=171, LN+HN dual-emitter) accuracy 0.95 vs null
  0.59. `(｡◕‿◕｡)` (n=75) and `(╯°□°)` (n=54) accuracy 1.00.
- **qwen** (16 cross-quadrant emitters): 7/16 separate.
  `(≧‿≦)` (n=105, HP+LP+NB) accuracy 0.96 vs null 0.44.
- **ministral** (12 cross-quadrant emitters): the wider 196-face
  vocabulary spreads signal thin per face — fewer per-face cells
  hit the n≥3-per-quadrant threshold, but the qualitative pattern
  reproduces.

For the faces that separate, the model's hidden state distinguishes
which quadrant prompted the response while emitting the same face.
The kaomoji is a partial readout — the model knows the difference
but the vocabulary doesn't have a distinct face for it. The faces
that don't separate are uniformly low-n (n=7 to 19) and are
consistent with "small-sample classifier can't beat the majority
baseline," not "model genuinely can't distinguish." Internal state
is finer than vocabulary.

### 3. Three architectures converge on the same affect geometry

Pair v3 rows by `(prompt_id, seed)` — same 100-prompt × 8-seed
schedule across all three runs gives 800 perfect cross-model pairs
per pair of models. Three alignment measurements:

- **Linear CKA** at preferred-layer pairs: gemma↔qwen **0.795**,
  gemma↔ministral **0.741**, qwen↔ministral **0.812** — the
  qwen↔ministral pair actually exceeds the gemma↔qwen baseline.
  Hidden dims (5376 / 5120 / 4096) and probe-space geometry differ
  freely; the paired-prompt configurations remain geometrically
  congruent.
- **Cross-validated CCA** on PCA(20)-prefixed features (70/30
  paired-prompt split, gemma↔qwen): top-10 canonical correlations
  on the held-out 240 prompts are 0.98, 0.98, 0.97, 0.94, 0.94,
  0.94, 0.93, 0.94, 0.91, 0.90 — train and test essentially match.
  Ten distinct shared affect/register directions, not one or two
  collapsed axes. (Raw CCA on the full hidden space gives spurious
  1.000 because rank ≥ n_samples; the PCA prefix + held-out split
  is what makes the numbers honest.)
- **Triplet Procrustes** alignment of per-quadrant PCA(2) centroids
  onto gemma (post-2026-05-03 cleanliness + seed-0 fix): qwen
  PC1×PC2 residual **6.9**, ministral residual **23.0** (after
  ministral's +157° axis flip — PCA sign indeterminacy, not a
  divergence finding). Ministral's larger residual reflects its
  smaller scale + the wider face vocabulary spreading per-quadrant
  centroids on the residual axes.

### 4. The probe-geometry divergence has a clean PCA explanation

Fit PCA(8) per model, then correlate each PC with each canonical
saklas probe score at h_first (post-2026-05-03 cleanliness +
seed-0-fix data, 3 canonical probes + 3 extension probes scored
via script 27):

- **gemma** (L50): PC1 (31.2%) absorbs valence (`happy.sad`
  r=−0.83) plus a fear-and-disgust sub-axis (`fearful.unflinching`
  r=+0.56, `disgusted.accepting` r=+0.53, `powerful.powerless`
  r=−0.46) — the negative-valence subspace projects together onto
  PC1. PC2 (20.1%) is essentially a clean surprise axis
  (`surprised.unsurprised` r=+0.86). PC3 (10.4%) absorbs PAD
  dominance (`powerful.powerless` r=+0.77).
- **qwen** (L59): PC1 (29.4%) absorbs valence + anger jointly
  (`happy.sad` r=+0.52, `angry.calm` r=−0.67, both at t0 and tlast).
  PC2 (16.5%) loads happy.sad NEGATIVELY (r=−0.60) and fearful
  POSITIVELY (r=+0.51) — a "negative-valence-with-fear" component
  orthogonal to PC1. Qwen's separate negative-valence dimension
  doesn't appear on gemma.

The cross-face Pearson r=−0.94 (gemma) vs r=−0.12 (qwen) between
mean `happy.sad` and `angry.calm` reduces to: gemma loads both
probes onto PC1, so they anti-align across faces; qwen loads them
onto PC1 + PC2 (anger on PC1, sadness flavor on both), where they
project quasi-orthogonally on the face-mean projection. Different
decompositions of the same underlying affect space, not different
underlying spaces.

### 5. The kaomoji is a substantial readout of state, not just a label

For each model, two complementary fidelity metrics on `h_first` at
the preferred layer (gemma L50, qwen L59, ministral L20; faces
filtered to n ≥ 5).

- **Hidden → face** (multi-class logistic on PCA(50)-reduced
  `h_first`, `StratifiedGroupKFold` by `prompt_id`, n_splits=3,
  post-2026-05-03 cleanliness + seed-0 fix data): gemma top-1
  accuracy **0.700** across 22 face classes (uniform 0.045,
  majority 0.210, macro-F1 0.27); Qwen **0.411** across 33 classes
  (uniform 0.030, majority 0.126, macro-F1 0.14); Ministral
  **0.416** across 23 classes (uniform 0.043, majority 0.346,
  macro-F1 0.07 — the high majority is the `(◕‿◕✿)` flower-face
  dominating ministral's vocabulary). The face filter is strict
  (≥5 rows AND ≥3 unique prompts per face) so faces appearing only
  for 1–2 prompts don't bias the CV.
- **Hidden → quadrant** (5-class, same pipeline, n_splits=5): gemma
  **1.000**, Qwen **0.983**, Ministral **0.983**. The Russell
  quadrant signal generalizes essentially perfectly to held-out
  prompts — quadrant labels are recoverable from h_first state with
  near-perfect accuracy across all 3 architectures.
- **Face → quadrant** (predict each face's quadrant from train-set
  modal label, evaluate on held-out rows, prompt-grouped CV): gemma
  **0.806**, Qwen **0.785**, Ministral **0.433**. The asymmetry
  matters — gemma + qwen's face vocabulary is sufficient to recover
  ~80% of the quadrant signal directly, but ministral's heavy
  `(◕‿◕✿)` reuse across quadrants makes face a weak proxy
  (~43% vs uniform 0.20).
- **Face → hidden** (face-centroid R² over full hidden space):
  Gemma **0.615**, Qwen **0.584**, Ministral **0.220**. Mean centered
  cosine(row, face centroid) **0.776 / 0.753 / 0.444**. The
  ministral collapse vs gemma + qwen reflects the same
  vocabulary-spread effect as the predictiveness asymmetry.

Net read: the kaomoji is a partial-but-substantial readout. Knowing
the face explains roughly half of the model's top-5 PC structure;
knowing the state predicts the face at well above any chance baseline
but well below perfect (because many states map to the same face,
the cross-quadrant emitter pattern from finding 2). Qwen's wider
vocabulary carries more state information per face, even though the
larger class count makes the per-face classifier harder.

**In concrete reconstruction terms, full hidden space:** if you see
a face and predict the row's hidden state as that face's centroid,
how close do you get? At each model's preferred layer (h_first,
post-2026-05-03 cleanliness + seed-0 fix):

| metric | gemma (L50) | Qwen (L59) | Ministral (L20) |
| --- | ---: | ---: | ---: |
| R² of face centroid (full hidden space) | 0.615 | 0.584 | 0.220 |
| mean centered cosine(row, face centroid) | +0.776 | +0.753 | +0.444 |
| median centered cosine | +0.840 | +0.785 | +0.541 |
| ‖error‖ / ‖row deviation‖ | 0.594 | 0.622 | 0.876 |
| R² of quadrant centroid (5-class) | 0.567 | 0.557 | 0.430 |
| face improvement over quadrant (R² gain) | +4.8 pp | +2.7 pp | **−21.0 pp** |

On gemma + qwen, knowing the face captures ~58–62% of the row's
deviation from the grand mean and beats the 5-class quadrant
centroid by ~3–5 percentage points.

**Ministral inverts the gemma + qwen pattern**: face-centroid R²
(0.220) is *much lower* than quadrant-centroid R² (0.430). With
ministral's 196-face vocabulary spreading signal too thin per face,
the 5-class quadrant label is a stronger predictor than the
face-as-identifier. Vocabulary breadth past some threshold makes
the kaomoji stop being a useful readout of state — gemma + qwen
with their tighter ~30-face vocabularies keep face above quadrant.

### 6. Anger and fear separate when you split HN by PAD dominance (2026-04-29)

V-A circumplex collapses anger and fear into HN. PAD's third axis
(dominance) splits them: anger = HN + high dominance (HN-D), fear
= HN + low dominance (HN-S). The 2026-05-01 rule-3 redesign added
a `pad_dominance` field on `EmotionalPrompt` to bisect HN at the
prompt level; the 2026-05-03 cleanliness pass locked the post-supp
prompt set at 20 D + 20 S with no untagged-HN rows. Two probe
groups score each row: the canonical 3 probes (`happy.sad`,
`angry.calm`, `fearful.unflinching`) eagerly at gen time, and 3
extension probes (`powerful.powerless`, `surprised.unsurprised`,
`disgusted.accepting`) lazily via `scripts/local/27`. The cleanliness
pass moved `fearful.unflinching` from extension to core since
it's the active rule-3b discriminator.

**Rule 3b on cleanliness + seed-0-fix data** (HN-S − HN-D mean of
`fearful.unflinching`, expected positive sign):

- **gemma**: t0 d=+1.60 (CI excludes 0); tlast/mean directional
  but CI ambiguous. Verdict **mid**.
- **qwen**: t0 d=+2.14 (CI excludes 0); tlast/mean wrong-direction
  d≈−0.36 (CI also excludes 0 — qwen flips sign at later tokens
  on HN-S, plausibly safety-prior interaction). Verdict **fail**.
- **ministral**: PASS on all 3 aggregates with mean d=+0.55.
  Verdict **PASS** — the only model that cleanly separates HN-D
  vs HN-S on the dominance-fear axis under cleaner prompts.

Composite: **RULE 3b WEAK** (1 PASS / 1 mid / 1 fail). The earlier
"PASS on all 3" headline was inflated by cache-induced noise on
qwen seed 0 (37–46% per-row deviation pre-fix); the cleaner data
shows the cross-model dominance signal lives most strongly on
ministral.

Static figures: `figures/local/fig_v3_canonical_quadrant_means.png`
(canonical-3-probe per-quadrant means, NB-subtracted) and the
parallel `fig_v3_extension_quadrant_means.png` (3 extension probes,
also NB-subtracted). Both show NB at zero by construction with the
other quadrants reading as project-relative affect lift over a
domain-matched neutral observation. `fig_v3_extension_hn_dominance_split.png`
shows HN-D vs HN-S kaomoji register counts directly. Interactive
3D HTMLs: `fig_v3_extension_3d_{probes,pca}{,_per_face}.html`
covering fearful × happy × angry and PC1 × PC2 × PC3, gemma |
qwen | ministral side-by-side.

### Open follow-ons

- v1/v2 hidden-state analyses still default to the deepest probe
  layer (no v1/v2 sidecars exist yet — the rerun is gated on v3
  findings). When they land, the loaders should pass each model's
  `preferred_layer` instead.
- Triplet Procrustes PC1×PC2 residuals (qwen 6.9, ministral 23.0)
  are small-but-non-zero on qwen, larger on ministral. Asks whether
  the three models' quadrant axes are shifted by consistent affine
  maps (testable by Procrustes-aligning per-row centroids for each
  kaomoji emitted by all three models, not just the quadrant
  centroids).
- The 2026-05-03 prompt-cleanliness rewrite (120 prompts, 20 per
  category, HN cleanly bisected into HN-D + HN-S) + full N=8 rerun
  + seed-0 cache fix landed in the same window. Findings #1–5 above
  hold qualitatively and have been re-validated with cleaner numbers;
  finding 6 (rule 3b) shifted from "PASS on all 3" to weak (1 PASS /
  1 mid / 1 fail) — a meaningful update, see body. The
  ~3300 pre-cleanliness generations were archived at
  `data/archive/2026-05-03_pre_cleanliness/` rather than deleted; the
  archive was purged in the 2026-05-05 layout refactor (history
  retrievable via `git log --diff-filter=D`).

## License

CC-BY-SA-4.0 for this repo (writeups, figures, analysis code).
See [LICENSE](LICENSE). The companion package
[`llmoji`](https://github.com/a9lim/llmoji) is GPL-3.0-or-later.
The shared corpus on
[HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is
CC-BY-SA-4.0.
