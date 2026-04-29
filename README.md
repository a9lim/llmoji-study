# llmoji-study

> **this README and the writeups in `docs/` are
> placeholders drafted by claude. i'll properly write a polished version
> once the experiments conclude.**

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
probes miss. On 800 generations balanced across the five Russell
quadrants, hidden-state PCA on gemma at the peak-affect layer L31
gives PC1 19.8% and PC2 7.0%, and the quadrants separate cleanly on
both axes: PC1 reads as valence (HN/LN at +9 to +13, HP/LP/NB at
−3 to −9), PC2 carries arousal (HN +3.7 vs LN −6.0; HP −6.8 vs
LP +2.1). Within-kaomoji consistency to mean is 0.92 to 0.99 across
the 32 canonical faces. The Qwen3.6-27B replication on the same
prompts has 2x the kaomoji vocabulary (65 canonical forms vs 32) at
PC1 14.9% / PC2 8.3%, with a similarly clean Russell circumplex
shape — Procrustes alignment of the per-quadrant centroids in PCA(2)
space rotates the two configurations into each other at +7.8° with
small residual, so the two architectures recover the same
two-dimensional affect geometry. The bundled saklas probes
nevertheless differ structurally between models: Pearson r between
mean `happy.sad` and `angry.calm` across faces is −0.94 on gemma
(probe-space valence-collapse) and −0.12 on Qwen (near-orthogonal),
because saklas's contrastive probes are extracted at saklas's own
internal layer and inherit each model's probe-space layout, not the
hidden-state layout the L31 PCA is reading.

Full setup, decision rules, per-quadrant centroids, all the
cross-model comparisons, and the Ministral vocabulary pilot are in
[`docs/local-side.md`](docs/local-side.md).

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

For the local side, set `LLMOJI_MODEL=gemma|qwen|ministral` and run
`scripts/03_emotional_run.py` (the v3 800-generation runner).
[`docs/local-side.md`](docs/local-side.md) has the per-pilot script
chain.

For the harness side, you need an `ANTHROPIC_API_KEY` (the cluster
labeler calls Haiku) and the HF Hub Python client (the install
above pulls `huggingface_hub`). Anonymous reads of the public
dataset are fine, so `HF_TOKEN` is optional.

```bash
python scripts/06_claude_hf_pull.py            # snapshot a9lim/llmoji into data/hf_dataset/
python scripts/07_claude_kaomoji_basics.py     # printout: top kaomoji, providers, contributors
python scripts/15_claude_faces_embed_description.py  # per-canonical embeddings
python scripts/16_eriskii_replication.py       # axes, clusters, writeup
python scripts/18_claude_faces_pca.py          # PCA panel
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

## Findings summary

A condensed read of what the local-side experiments have shown so far.
Numbers cite the v3 follow-on analyses (`scripts/21`–`scripts/24`) run
on the existing 800-generation gemma and Qwen3.6-27B sidecars. Full
details, gotchas, and the experiment plans live in
[`CLAUDE.md`](CLAUDE.md) and [`docs/local-side.md`](docs/local-side.md);
the four headline figures are in
[`figures/local/cross_model/`](figures/local/cross_model/).

### 1. Affect emerges mid-network on gemma, late on Qwen

Each per-row hidden-state sidecar stores `h_mean` for every probe layer,
so layer-wise structure is recoverable without re-running the model.
Russell-quadrant silhouette over PCA(2) coordinates as a function of
probe layer:

- gemma (56 layers): peaks at **L31 (silhouette 0.184)**, degrades 36%
  to **0.117 at the deepest L57**. Half-peak reached by L7.
- Qwen (60 layers): peaks at **L59 (silhouette 0.313)** and stays at
  **0.304 at L61**. Monotonic refinement to the output.

This invalidates a chunk of the prior writeup. Every v3 figure
defaulted to the deepest probe layer; gemma's was therefore being
read at a 36% degraded snapshot. The repo now stores `preferred_layer`
on `ModelPaths`: gemma=31, Qwen=None (uses deepest). At L31 gemma's
PC1 explained variance jumps from 13.0% to **19.8%**, the per-face
PCA spectrum jumps from 16.4% / 7.4% to **30.4% / 11.2%**, and the
HN/LN-collapse-on-PC1 finding (the original "Qwen has a 2D circumplex
but gemma doesn't" framing) goes away — at L31 HN and LN separate
on PC2 by 9.7 units even though they still share the
`(｡•́︿•̀｡)` face vocabulary at the output.

Qwen's peak silhouette is still 70% higher than gemma's even at the
right layer for each, so the two models' affect representations
aren't equivalent — Qwen's is genuinely cleaner — but the structural
divergence is much smaller than the original L57 numbers suggested.

### 2. Kaomoji is a partial readout, not the state itself

For each face emitted in two or more quadrants with at least three rows
in each, train a PCA(20)→l2-logistic classifier on `h_mean` to predict
which quadrant prompted each instance, using only that face's rows.
Compare 5-fold stratified CV accuracy to a 30-shuffle label-permutation
null at q95.

- gemma (10 cross-quadrant emitters at L31): **6/10 separate**.
  `(｡•́︿•̀｡)` (n=171, the LN+HN dual-emitter) accuracy 0.95 vs null
  0.59. `(｡◕‿◕｡)` (n=75) and `(╯°□°)` (n=54) accuracy 1.00.
- Qwen (16 cross-quadrant emitters at L61): **7/16 separate**.
  `(≧‿≦)` (n=105, HP+LP+NB) accuracy 0.96 vs null 0.44.

For the faces that separate, the model's hidden state distinguishes
which quadrant prompted the response while emitting the same face.
The kaomoji is a partial readout — the model knows the difference but
the vocabulary doesn't have a distinct face for it. The faces that
don't separate are uniformly low-n (n=7 to 19) and are consistent
with "small-sample classifier can't beat the majority baseline,"
not "model genuinely can't distinguish." Internal state is finer
than vocabulary.

### 3. Gemma and Qwen converge on the same affect geometry

Pair the v3 rows by `(prompt_id, seed)` — both runs used the same 100
prompts × 8 seeds, so 800 perfect cross-model pairs. Three measurements
of representational alignment:

- Linear CKA (centered Gram form, computed across the full 56×60 layer
  grid in ~5 seconds): **0.84 at the deepest-layer pair** (gemma L57
  ↔ Qwen L61), **0.86 maximum** (gemma L52 ↔ Qwen L58), 0.80 at the
  preferred-layer pair (gemma L31 ↔ Qwen L61). The two models put
  paired prompts in geometrically similar configurations within their
  respective hidden spaces, despite different hidden dims (5376 vs
  5120) and different probe-space geometry.
- Cross-validated CCA on PCA(20)-prefixed features with a 70/30
  paired-prompt split: top-10 canonical correlations on the held-out
  240 prompts are 0.98, 0.98, 0.97, 0.94, 0.94, 0.94, 0.93, 0.94,
  0.91, 0.90 — train and test essentially match, so no overfit.
  Ten distinct shared affect/register directions, not just one or
  two collapsed axes. (Raw CCA on the full hidden space gives spurious
  1.000 across the board because rank ≥ n_samples; the PCA prefix
  + held-out split is what makes the numbers honest.)
- Procrustes alignment of per-quadrant PCA(2) centroids: **+7.8°
  rotation, residual 5.7** at the preferred-layer pair (down from
  +14.0° / 6.4 at the deepest-layer pair). The Russell circumplex has
  the same shape across architectures; Qwen's version is several
  times longer in absolute scale (LN at PC1 +44 vs gemma's +10) but
  the geometric structure is preserved.

### 4. The probe-geometry divergence has a clean PCA explanation

Fit PCA(8) on `h_mean` per model, then correlate each PC with each of
the five saklas probe scores at t0 (whole-generation aggregate):

- gemma (L31): PC1 absorbs valence directly (`happy.sad` r=−0.69,
  `angry.calm` r=+0.46 — same valence-collapse the v1/v2 probe-space
  analysis hit). PC2 absorbs a humor + warmth + arousal mix
  (`humorous.serious` +0.42, `warm.clinical` −0.39).
- Qwen (L61): PC1 absorbs valence + humor jointly (`happy.sad`
  r=−0.86, `humorous.serious` r=−0.69). PC2 absorbs certainty
  (`confident.uncertain` r=−0.48). PC3 absorbs arousal + warmth
  (`angry.calm` r=−0.61, `warm.clinical` r=+0.48).

The cross-face Pearson r=−0.94 vs r=−0.12 between mean `happy.sad`
and `angry.calm` reduces to: gemma loads both probes onto PC1+PC2
together, so they anti-align across faces; Qwen loads them onto PC1
vs PC3, which are nearly orthogonal in face-space. Different
decompositions of the same underlying affect space, not different
underlying spaces.

### 5. The kaomoji is a substantial readout of state, not just a label

For each model, two complementary fidelity metrics on `h_mean` at the
preferred layer (faces filtered to n ≥ 5):

- **Hidden → face** (multi-class logistic on PCA(50)-reduced `h_mean`,
  5-fold CV): gemma top-1 accuracy **0.71** across 19 face classes
  (uniform baseline 0.05, majority 0.23, macro-F1 0.52); Qwen
  **0.495** across 28 classes (uniform 0.04, majority 0.14, macro-F1
  0.30). Both 13–14× uniform. The cross-model gap is mostly a
  class-count effect — more faces means harder per-class assignment.
  High-frequency faces are recoverable with 70–85% recall (gemma's
  `(๑˃‿˂)` n=181 at 0.87, Qwen's `(≧‿≦)` n=106 at 0.84); some
  low-n same-quadrant siblings have 0% recall (model knows the
  state, picks a different face for it).
- **Hidden → quadrant** (5-class, same pipeline): both models hit
  **1.000** accuracy. Caveat: 5-fold CV is by row, not by prompt;
  with 8 seeds × 100 prompts the same prompt appears in train and
  test folds with different seeds, so this number is inflated by
  prompt-level leakage. Even with that caveat, quadrant identity is
  exactly recoverable from a much-larger-than-PC1+PC2 PCA prefix —
  the v3 quadrant signal isn't only visible in the figures, it
  saturates whatever classifier capacity you give it.
- **Face → hidden**: η² of face identity across the top-5 PCs.
  Gemma: per-PC η² 0.62 / 0.36 / 0.44 / 0.30 / 0.28; weighted by
  explained variance, face identity recovers **49% of the top-5 PC
  subspace**. Qwen: η² 0.81 / 0.53 / 0.54 / 0.13 / 0.36; **60% of
  the top-5 PC subspace**. Qwen has slightly less variance in the
  top-5 (38.0% vs 39.6%) but face identity recovers more of it —
  each Qwen face commits to a tighter slice of the affect manifold
  than each gemma face does.

Net read: the kaomoji is a partial-but-substantial readout. Knowing
the face explains roughly half of the model's top-5 PC structure;
knowing the state predicts the face at well above any chance baseline
but well below perfect (because many states map to the same face,
the cross-quadrant emitter pattern from finding 2). Qwen's wider
vocabulary carries more state information per face, even though the
larger class count makes the per-face classifier harder.

**In concrete reconstruction terms, full hidden space:** if you see
a face and predict the row's hidden state as that face's centroid,
how close do you get? At each model's preferred layer:

| metric | gemma (L31) | Qwen (L61) |
| --- | ---: | ---: |
| R² of face centroid (full hidden space) | 0.260 | 0.287 |
| mean centered cosine(row, face centroid) | +0.486 | +0.523 |
| median centered cosine | +0.550 | +0.537 |
| ‖error‖ / ‖row deviation‖ | 0.857 | 0.838 |
| R² of quadrant centroid (5-class) | 0.254 | 0.264 |
| face improvement over quadrant (R² gain) | +0.6 pp | +2.3 pp |

Reading this against the η² above: the kaomoji captures about a
quarter to a third of the row's variance from the grand mean, with
the residual still ~85% as long in L2. Almost all of that captured
variance is already accounted for by knowing which Russell quadrant
the row came from — face identity adds only +0.6 (gemma) to +2.3
(Qwen) percentage points of full-space R² over quadrant alone. The
49–60% η² in the top-5 PC subspace and the 0.6–2.3 pp R² gain in
full hidden space are the same finding, looked at from two
directions: kaomoji choice tracks the affect direction tightly,
and is roughly independent of the bulk of hidden-state variance,
which is content-related (which prompt, what topic) rather than
affect-related.

### Open follow-ons

- v1/v2 hidden-state analyses still default to the deepest probe layer.
  The same layer-choice artifact may have understated how separable
  the steered conditions actually are; worth re-running at L31 before
  the v1/v2 writeup lands.
- The remaining 7.8° Procrustes rotation between gemma L31 and Qwen
  L61 is non-trivial. Asks whether the two models' quadrant axes are
  shifted by a consistent affine map (testable by Procrustes-aligning
  the per-row centroids for each kaomoji emitted by both models, not
  just the quadrant centroids).
- Same-face-cross-quadrant separability is currently a 6/10 vs 7/16
  result on small per-face samples. Tightening the threshold (raising
  `min_per_quadrant` to 5) and adding a Ministral run when v3 lands
  there would sharpen the "vocabulary as bottleneck" claim.
- Script 25's quadrant classifier hits 1.000 because 5-fold CV doesn't
  hold out by prompt. A `GroupKFold` split keyed on `prompt_id` would
  give the rigorous version; expect quadrant accuracy to drop toward
  the silhouette-implied level (~0.7–0.8) under that stricter split.
  The face classifier and η² numbers are less affected because face
  identity isn't determined by prompt alone (each prompt elicits
  multiple faces across seeds).

## License

CC-BY-SA-4.0 for this repo (writeups, figures, analysis code).
See [LICENSE](LICENSE). The companion package
[`llmoji`](https://github.com/a9lim/llmoji) is GPL-3.0-or-later.
The shared corpus on
[HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is
CC-BY-SA-4.0.
