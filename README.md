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
at gemma **L50 (0.235)**, qwen **L59 (0.244)**, ministral **L20
(0.149)** — gemma + qwen sit at the deep half of their networks,
ministral is mid-depth. Switching from the previous canonical
`h_mean` aggregate roughly doubled-to-tripled silhouette across all
three models (gemma 2.0×, qwen 2.1×, ministral 3.3×) and shifted
gemma + qwen's peaks substantially deeper. The earlier "gemma
mid-depth, qwen deep" framing dissolved.

The three Russell circumplexes are geometrically congruent. Triplet
Procrustes alignment of per-quadrant PCA(2) centroids onto gemma:
qwen residual 5.6, ministral residual 6.4 (after a −176° axis flip
that's PCA sign indeterminacy, not a divergence finding). Pairwise
CKA at preferred layers: gemma↔qwen 0.795, gemma↔ministral 0.741,
qwen↔ministral 0.812 — the qwen↔ministral pair actually exceeds the
gemma↔qwen baseline. A single canonical alignment layer at
ministral L20 maximizes CKA against either reference model
regardless of where the partner's affect representation sits.
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
via a `pad_dominance` field on `EmotionalPrompt`. 23 supplementary
prompts (hn21–hn43) brought the post-supp dataset to 20 D / 20 S
balanced (160 rows per group per model). On the balanced data,
**rule 3b (`fearful.unflinching` mean(HN-S) > mean(HN-D))
PASSES on all three models**: directional + bootstrap 95% CI
excludes zero on at least 2 of 3 aggregates (t0, tlast, mean). Largest
effects: qwen t0 Cohen's d = +2.35, ministral mean d = +0.81. Rule
3a (`powerful.powerless`) was dropped — fails across all three models
in the wrong direction with CI-excludes-zero on gemma + ministral
mean-aggregates, so the probe doesn't read PAD dominance in the HN
context (likely reads "felt agency in achievement contexts" instead).
Cross-model takeaway: PAD dominance has a real internal
representation in all three architectures + labs (Google / Alibaba /
Mistral) and reads cleanly via the fear axis against the registry
HN-D / HN-S split.

The v3 prompt set was rewritten end-to-end 2026-05-03 in the prompt
cleanliness pass (`docs/2026-05-03-prompt-cleanliness.md`) — 120
prompts (20 per category) replacing the prior 123, HN cleanly
bisected into 20 HN-D + 20 HN-S with no untagged entries, IDs
renumbered hn01–hn40. The rule 3b PASS verdict above holds for the
prior set; rerun on the new set is gated on further design
discussion + ethics review of trial scale.

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
`scripts/local/03_emotional_run.py` (the v3 800-generation runner).
See `CLAUDE.md` § Commands for the full script chain.

For the harness side, you need an `ANTHROPIC_API_KEY` (the cluster
labeler calls Haiku) and the HF Hub Python client (the install
above pulls `huggingface_hub`). Anonymous reads of the public
dataset are fine, so `HF_TOKEN` is optional.

```bash
python scripts/harness/06_claude_hf_pull.py            # snapshot a9lim/llmoji into data/hf_dataset/
python scripts/harness/07_claude_kaomoji_basics.py     # printout: top kaomoji, providers, contributors
python scripts/harness/15_claude_faces_embed_description.py  # per-canonical embeddings
python scripts/harness/16_eriskii_replication.py       # axes, clusters, writeup
python scripts/harness/18_claude_faces_pca.py          # PCA panel
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
run on the 800-generation gemma + qwen + ministral sidecars under
the post-2026-05-02 `h_first` standardization. Full details and
gotchas live in [`CLAUDE.md`](CLAUDE.md),
[`docs/findings.md`](docs/findings.md), [`docs/local-side.md`](docs/local-side.md),
and [`docs/gotchas.md`](docs/gotchas.md); headline figures and
interactive 3D HTMLs are in
[`figures/local/cross_model/`](figures/local/cross_model/).

### 1. Affect peaks at the model-specific preferred layer, not the deepest

Each per-row hidden-state sidecar stores per-layer aggregates, so
layer-wise structure is recoverable without re-running the model.
Russell-quadrant silhouette over PCA(2) at `h_first` (the
kaomoji-emission state) by probe layer:

- **gemma** (56 layers): silhouette peaks at **L50 (0.235)**, top-5
  layers cluster L47–51 (~84–91% depth, plateau not single peak).
- **qwen** (60 layers): peaks at **L59 (0.244)**, top-5 layers L54–59
  (~88–97% depth) — explicit near-deepest plateau.
- **ministral** (37 layers): peaks at **L20 (0.149)**, top-5 layers
  L20–26 (~54–70% depth) — the only mid-depth model.

The repo stores `preferred_layer` on `ModelPaths`: gemma 50, qwen
59, ministral 20. v3 figures default to those layers.

These numbers are the post-2026-05-02 `h_first` cutover; the prior
canonical aggregate was `h_mean`, under which silhouettes were
roughly half-to-a-third (gemma 0.116, qwen 0.116, ministral 0.045)
and gemma's peak sat at L28. `h_first` is methodology-invariant
across the MAX_NEW_TOKENS=120→16 cutover (the kaomoji-emission
state doesn't depend on how long the generation runs after it),
which `h_mean` and `tlast` aren't. The earlier "gemma mid-depth,
qwen deep" framing dissolved with the cutover — gemma + qwen are
now both deep, ministral is the only mid-depth model.

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
  onto gemma: qwen residual **5.6**, ministral residual **6.4**
  (after ministral's −176° axis flip — PCA sign indeterminacy, not
  a divergence finding). Same order of magnitude despite ministral's
  smaller scale + different lab.

### 4. The probe-geometry divergence has a clean PCA explanation

Fit PCA(8) per model, then correlate each PC with each of the five
core saklas probe scores at h_first:

- **gemma** (L50): PC1 absorbs valence (`happy.sad` r=−0.69,
  `angry.calm` r=+0.46 — same valence-collapse the v1/v2 probe-space
  analysis hit). PC2 absorbs a humor + warmth + arousal mix.
- **qwen** (L59): PC1 absorbs valence + humor jointly (`happy.sad`,
  `humorous.serious` both load high). PC2 absorbs certainty. PC3
  absorbs arousal + warmth.

The cross-face Pearson r=−0.94 vs r=−0.12 between mean `happy.sad`
and `angry.calm` reduces to: gemma loads both probes onto PC1+PC2
together, so they anti-align across faces; qwen loads them onto PC1
vs PC3, which are nearly orthogonal in face-space. Different
decompositions of the same underlying affect space, not different
underlying spaces.

### 5. The kaomoji is a substantial readout of state, not just a label

For each model, two complementary fidelity metrics on `h_first` at
the preferred layer (gemma L50, qwen L59, ministral L20; faces
filtered to n ≥ 5).

- **Hidden → face** (multi-class logistic on PCA(50)-reduced
  `h_first`, `StratifiedGroupKFold` by `prompt_id`, n_splits=3):
  gemma top-1 accuracy **0.68** across 17 face classes (uniform
  0.06, majority 0.22, macro-F1 0.37); Qwen **0.39** across 31
  classes (uniform 0.03, majority 0.12, macro-F1 0.15); Ministral
  **0.40** across 21 classes (uniform 0.05, majority 0.34, macro-F1
  0.07 — the high majority is the `(◕‿◕✿)` flower-face dominating
  ministral's vocabulary). Drops vs the prior leaky-CV numbers
  (gemma 0.71, qwen 0.50) are smaller than expected — face identity
  generalizes to never-seen prompts, with the largest hit on
  qwen (more face classes, more prompt-specific). The face filter
  is now stricter (≥5 rows AND ≥3 unique prompts per face, since
  faces appearing for only 1–2 prompts have nothing to hold out
  under prompt-grouped CV).
- **Hidden → quadrant** (5-class, same pipeline, n_splits=5): gemma
  **0.95**, Qwen **0.94**, Ministral **0.90**. Pre-fix prediction
  was that quadrant accuracy would drop to ~0.7–0.8 once leakage
  was removed; actual drop is only 5–10 percentage points. **The v3
  quadrant signal genuinely generalizes to held-out prompts**, not
  just memorized — a stronger result than pre-fix, and cross-model
  consistent across all three architectures.
- **Face → hidden**: η² of face identity across the top-5 PCs at
  h_first. Gemma per-PC η² 0.95 / 0.63 / 0.31 / 0.46 / 0.24; weighted
  by explained variance, face identity recovers **73% of the top-5
  PC subspace**. Qwen 0.94 / 0.67 / 0.49 / 0.45 / 0.40; **77%**.
  Ministral 0.54 / 0.16 / 0.10 / 0.12 / 0.03; **35%** — much lower
  because of the 196-face vocabulary spreading signal thin per face.
  The η² gains over the pre-h_first numbers (gemma was 49%, qwen
  60%) reflect h_first being more prompt-deterministic — face
  identity, which is largely prompt-driven, explains more of the
  variance at h_first than at h_mean. Same direction as the
  silhouette-doubling finding from h_first standardization.

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
post-2026-05-02 standardization):

| metric | gemma (L50) | Qwen (L59) | Ministral (L20) |
| --- | ---: | ---: | ---: |
| R² of face centroid (full hidden space) | 0.580 | 0.570 | 0.219 |
| mean centered cosine(row, face centroid) | +0.754 | +0.745 | +0.440 |
| median centered cosine | +0.798 | +0.785 | +0.541 |
| ‖error‖ / ‖row deviation‖ | 0.634 | 0.642 | 0.882 |
| R² of quadrant centroid (5-class) | 0.530 | 0.520 | 0.352 |
| face improvement over quadrant (R² gain) | +5.0 pp | +5.0 pp | **−13.3 pp** |

The pre-h_first numbers (gemma R² 0.260, qwen 0.287, +0.6 / +2.3 pp
face-over-quadrant gain) are several times smaller — h_first makes
the kaomoji a substantially stronger residual readout above the
Russell-quadrant signal. On gemma + qwen, knowing the face captures
~57–58% of the row's deviation from the grand mean and beats the
5-class quadrant centroid by 5 percentage points.

**Ministral inverts the gemma + qwen pattern**: face-centroid R²
(0.219) is *lower* than quadrant-centroid R² (0.352). With
ministral's 196-face vocabulary spreading signal too thin per face,
the 5-class quadrant label is a stronger predictor than the
face-as-identifier. Vocabulary breadth past some threshold makes
the kaomoji stop being a useful readout of state — gemma + qwen
with their tighter 33 / 67 vocabularies keep face above quadrant.

### 6. Anger and fear separate when you add the right probe (2026-04-29)

V-A circumplex collapses anger and fear into HN. PAD's third axis
(dominance) splits them: anger = HN + high dominance, fear = HN +
low dominance. Three new contrastive packs registered into saklas
via `scripts/local/26`: `powerful.powerless` (PAD dominance as felt
agency), `surprised.unsurprised` (Plutchik surprise),
`disgusted.accepting` (Plutchik disgust); `scripts/local/27` re-scores
the existing v3 sidecars against these plus auto-discovers
`fearful.unflinching`, `curious.disinterested`, and several
register probes from a working-saklas-repo install — total
12 extension probes per row at h_first / h_last / h_mean
snapshots, no new generations.

- **gemma** at h_last: `fearful.unflinching ↔ powerful.powerless`
  per-row r = **−0.936** — the dominance and fear directions
  collapse onto one axis in gemma's deep representation, sign-flipped.
  `fearful.unflinching ↔ angry.calm` r = **−0.848** — the model
  picks anger OR fear per HN row, not both. Per-quadrant means at
  h_last separate cleanly: HN powerful −0.10, HP/LP/NB hover at
  zero; HN fearful +0.22, HP/LP/NB ~+0.13.
- **qwen** at h_last: extension probes are nearly flat across
  quadrants (range ~0.013 on `powerful.powerless`, ~0.019 on
  `fearful.unflinching`). `fearful ↔ powerful` r = +0.008.
  Two interpretations both possible: qwen's hidden states are
  genuinely orthogonal to the contrastive directions saklas
  extracted from these statement sets, or h_last is the wrong
  snapshot for qwen even though it's the deepest layer.
- **HN dominance-split natural experiment on gemma**: split HN
  rows into thirds by `powerful.powerless` and tally kaomoji
  register. Bottom-third (most powerless / fearful): 25 shocked-
  register `(╯°□°)/(⊙_⊙)/(>_<)` faces, 25 sad-teary
  `(｡•́︿•̀｡)`. Top-third (most powerful / non-fear): 22 shocked,
  29 sad-teary. Within-HN, kaomoji vocabulary is roughly stable
  across the dominance split — same "kaomoji finer than vocabulary
  in some axes, coarser in others" pattern as finding 2.

Static figures: `figures/local/cross_model/fig_v3_extension_*.png`
(per-quadrant means, fearful↔powerful scatter, HN dominance
register stack, probe correlation matrix). Interactive 3D HTMLs:
`fig_v3_extension_3d_{probes,pca}{,_per_face}.html` covering
fearful × happy × angry and PC1 × PC2 × PC3 in both per-row and
per-face aggregated forms, side-by-side gemma | qwen scenes.

### Open follow-ons

- v1/v2 hidden-state analyses still default to the deepest probe
  layer (no v1/v2 sidecars exist yet — the rerun is gated on v3
  findings). When they land, the loaders should pass each model's
  `preferred_layer` instead.
- Triplet Procrustes residuals (qwen 5.6, ministral 6.4) are
  small-but-non-zero. Asks whether the three models' quadrant axes
  are shifted by consistent affine maps (testable by Procrustes-
  aligning per-row centroids for each kaomoji emitted by all three
  models, not just the quadrant centroids).
- The 2026-05-03 prompt-cleanliness rewrite (120 prompts, 20 per
  category, HN cleanly bisected into HN-D + HN-S) invalidates the
  ~3300 prior v3 generations for cross-run comparison. Rerun is
  gated on further design discussion + ethics review of trial scale.
  Findings #1–6 hold under the prior (123-prompt) set; rule-3b
  re-validation under the new set is the most consequential pending
  check.

## License

CC-BY-SA-4.0 for this repo (writeups, figures, analysis code).
See [LICENSE](LICENSE). The companion package
[`llmoji`](https://github.com/a9lim/llmoji) is GPL-3.0-or-later.
The shared corpus on
[HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is
CC-BY-SA-4.0.
