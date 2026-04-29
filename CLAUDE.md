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

**v3 follow-on analyses landed 2026-04-28** (no new model time, all
recovered from existing sidecars): layer-wise emergence trajectory,
same-face-cross-quadrant natural experiment, cross-model alignment
(CKA + Procrustes), PC3+ × probes. Headline finding from layer-wise:
gemma's affect representation peaks at L31 of 56, not the deepest L57
the v3 figures defaulted to. Switching to L31 (via the new
`preferred_layer` field on `ModelPaths`) substantially sharpens
gemma's Russell-quadrant separation, dissolves the prior "gemma 1D
vs qwen 2D" framing, and cuts the cross-model Procrustes rotation
from +14° to +7.8°. See "v3 follow-on analyses" below.

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

**Findings (post-canonicalization, hidden-state space, 32 forms,
h_mean at L31 — gemma's `preferred_layer`; the 2026-04-28 layer-wise
emergence analysis showed L57 silhouette = 0.117 vs L31 silhouette =
0.184, so v3 figures default here now):**

- PCA: PC1 **19.83%**, PC2 **7.04%** (cumulative 26.87% vs probe-space
  PC1 = 89%, valence-collapse solved). Per-face PCA (over 32 face
  means, not 800 rows): PC1 **30.4%**, PC2 **11.2%** — much cleaner
  face-level structure than the L57 numbers it replaces (16.4% / 7.4%).
- Russell quadrants separate cleanly. PC1 reads as valence
  (HN/LN/+9–13, HP/LP/NB −3 to −9), PC2 carries arousal (HN +3.7 vs
  LN −6.0; HP −6.8 vs LP +2.1; NB +7.3). Separation PC1 2.10 /
  PC2 2.12 (gemma L57 was 2.03 / 2.74; PC2 separation went down a
  bit but PC1 absorbs much more variance).
- **HN and LN do separate at L31** — PC2 gap is 9.7 units (HN +3.7,
  LN −6.0). The previous "HN/LN collapse on PC1" finding was an
  artifact of reading h_mean at L57; at L31 the two negative quadrants
  occupy distinct regions even though `(｡•́︿•̀｡)` (n=171, LN+HN) is
  still the shared face. Internal state distinguishes them; the
  vocabulary doesn't.
- Kaomoji emission (first-word filter): 100%. TAXONOMY match: HP 91% /
  LP 71% / LN 99% / HN 42% / NB 87%. HN gets a dedicated shocked/angry
  register `(╯°□°)/(⊙_⊙)/(⊙﹏⊙)` absent elsewhere.
- Cross-axis correlation across faces still strong: Pearson(mean
  happy.sad, mean angry.calm) r=−0.939 (n=32, p≈2e-15). This number
  doesn't change with PCA layer because saklas's probe scores in the
  JSONL are computed at the saklas-internal probe layer; the L31
  finding is about hidden-state geometry, not probe geometry.
- Figure refresh 2026-04-25: face-level figures (Fig C, fig_v3_face_*)
  color each face by an RGB blend of `QUADRANT_COLORS` weighted by
  per-quadrant emission count, replacing dominant-quadrant winner-take-all.
  Cross-quadrant emitters (the `(｡•́︿•̀｡)` LN/HN family) render as visible
  mixes; pure-quadrant faces stay at endpoints. Palette: HN red, HP gold,
  LP green, LN blue, NB gray.

**Pre-2026-04-28 numbers at L57 (kept for cross-checking against
prior writeups):** PCA PC1 12.98% / PC2 7.49%; per-quadrant centroids
HN +7 / LN +7 (collapsed); within-kaomoji h_mean consistency 0.92–0.99
across faces. The L57 findings led to the "gemma 1D-affect-with-
arousal-modifier vs qwen 2D Russell" framing — see "v3 follow-on
analyses" below; that framing dissolved once gemma was read at the
right layer.

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
  angry.calm) across faces is r=−0.117 (p=0.355) on Qwen vs r=−0.939 on
  gemma. The valence-collapse problem motivating v3 doesn't appear on Qwen
  — saklas's contrastive probes recover near-orthogonal happy.sad /
  angry.calm directions. v1/v2-style probe-space analysis would be
  substantially less collapsed. Cross-model architecture/training
  difference, not a saklas issue. **Note (2026-04-28):** the
  hidden-state-space divergence between gemma and qwen turned out to
  be largely a layer-choice artifact — at gemma's preferred layer L31
  the two models are much more aligned (Procrustes rotation +7.8°
  rather than +14°, see "v3 follow-on analyses"). Probe geometry
  itself, however, stays divergent because saklas's probes are
  computed at saklas's own internal layer, not at L31.
- Figure refresh 2026-04-25: same per-face RGB-blend coloring as gemma's.
  `(;´д｀)` family reads visibly purple; `(;ω;)` deep blue with a slight
  red cast.
- **Procedural:** the runner's per-quadrant "emission rate" log line is
  gated on `kaomoji_label != 0` (TAXONOMY match), not bracket-start.
  Reads as HP 28% / LP 13% / HN 2.5% / LN 11% / NB 12% on Qwen — gemma-
  tuned TAXONOMY not covering Qwen's vocab, NOT instruction-following
  failure. Real compliance is 100%.

### v3 follow-on analyses (2026-04-28)

Four scripts run on the existing v3 sidecars — no new model time. Helper
`load_hidden_features_all_layers` in `hidden_state_analysis.py` opens each
sidecar once and returns a `(n_rows, n_layers, hidden_dim)` tensor with a
disk cache at `data/cache/v3_<short>_h_mean_all_layers.npz` (gitignored,
~80 MB compressed per model). Sidecars store h_first/h_last/h_mean for
EVERY probe layer, not just the deepest — `(layer_idxs)` runs 2-57 on gemma
and 2-61 on qwen. Multi-layer trajectory is recoverable from existing data.

**Layer-wise emergence (`scripts/21_v3_layerwise_emergence.py`).** Per probe
layer, fit PCA(2) on h_mean and measure quadrant separation via silhouette
score, between-centroid std on PC1/PC2, and PC1/PC2 explained variance.
- **Gemma**: silhouette peaks at L31 (0.184) and DEGRADES to 0.117 at the
  deepest L57 — a 36% drop. The v3 figures defaulted to L57 pre-2026-04-28;
  the L31 finding led to adding `preferred_layer` to `ModelPaths` so every
  v3 script reads at L31 for gemma. Half-peak silhouette reached by L7.
- **Qwen**: silhouette peaks at L59 (0.313) and stays at 0.304 at L61 —
  affect representation refines monotonically through the network. Half-
  peak by L16. Qwen's `preferred_layer` stays None (defaults to deepest).
- **Cross-model**: qwen's peak silhouette is 70% higher than gemma's
  (0.31 vs 0.18). Even at the right layer for each, qwen's affect
  representation is CLEANER by absolute discriminability. The
  "gemma 1D vs qwen 2D" framing the L57 numbers suggested largely
  dissolves once gemma is read at L31 — gemma's HN/LN do separate
  (PC2 gap 9.7 units), gemma's PC1 absorbs 19.8% of variance vs qwen's
  14.9%. Two cleaner Russell circumplexes, qwen still tighter.
- Outputs: `figures/local/{gemma,qwen}/fig_v3_layerwise_emergence.png`,
  `fig_v3_layerwise_pca_quartiles.png`, `v3_layerwise_emergence.tsv` +
  `figures/local/cross_model/fig_v3_layerwise_emergence_compare.png`.

**Same-face cross-quadrant natural experiment
(`scripts/22_v3_same_face_cross_quadrant.py`).** For each face emitted in
≥2 quadrants with n≥3 each, train PCA(20) → l2-logistic on h_mean
predicting quadrant from that face's rows. 5-fold stratified CV vs 30-perm
label-shuffle null (q95). Above null = the model internally distinguishes
which quadrant prompted each instance even though it emits the same face.
- **Gemma (h_mean at L31)**: 6/10 cross-quadrant emitters separate.
  `(｡・́︿・̀｡)` (n=171, the LN/HN dual-emitter from the original gotcha)
  acc=0.95 vs null 0.59 — model knows the difference. `(๑˃‿˂)` acc=0.97
  vs null 0.48. `(｡◕‿◕｡)` (n=75) acc=1.00, `(｡♥‿♥｡)` (n=58) acc=1.00,
  `(╯°□°)` (n=54) acc=1.00, `(✿◠‿◠)` (n=38) acc=1.00. Four don't
  separate: `(´ω`)` (n=19, was 7/10 separable at L57; the move to L31
  made the within-class noise tighter so this borderline case dropped
  out), `(˘▽˘)` (n=17), `(˘̩╭╮˘̩)` (n=12), `(˘ڡ˘)` (n=7) — three are
  the same low-n outliers as at L57.
- **Qwen (h_mean at L61, deepest)**: 7/16 separate. Headline: `(≧‿≦)`
  (n=105, HP+LP+NB) acc=0.96 vs null 0.44, `(;ω;)` (n=80) acc=0.95.
- **The kaomoji is a partial readout, not the state itself.** For the
  faces that separate, internal hidden state carries the affect signal but
  the model collapses it to a shared face. For the faces that don't, the
  model genuinely doesn't distinguish — but those are universally low-n
  and small-vocabulary cases. The dominant pattern is "internal state
  finer than vocabulary."
- Outputs: `figures/local/{gemma,qwen}/fig_v3_same_face_cross_quadrant_*.png`
  (one per face) + `_summary.png` + `v3_same_face_cross_quadrant.tsv`.

**Cross-model alignment (`scripts/23_v3_cross_model_alignment.py`).** Pair
v3 rows by (prompt_id, seed) — 800 perfect pairs, both kaomoji-bearing.
Linear CKA via centered Gram matrices (kernel-form, O(n²) per pair after a
one-shot per-layer Gram precompute; the naive d×d covariance form takes
~25 min for the full grid, the Gram form ~5s). Cross-validated CCA on
PCA(20) features with a 70/30 paired-prompt split.
- **CKA grid (gemma layer × qwen layer)**: min 0.34, max 0.86. Three
  reference points:
    * preferred-layer pair (gemma L31 ↔ qwen L61): **0.798**
    * deepest-deepest (L57 ↔ L61): **0.844**
    * best-aligned cross-layer pair (gemma L52 ↔ qwen L58): **0.858**
  The deepest-deepest CKA is HIGHER than the preferred-layer CKA —
  representations converge geometrically near the output even when
  affect-readability has degraded mid-network on gemma. Worth holding
  both numbers when reasoning about "are the models aligned": at the
  best-affect layer for each, alignment is 0.80; at the literal output
  end, 0.84.
- **Cross-validated CCA (gemma L31 ↔ qwen L61)**: top-10 canonical
  correlations on held-out prompts: 0.98, 0.98, 0.97, 0.94, 0.94, 0.94,
  0.93, 0.94, 0.91, 0.90. Train and test essentially match — no overfit.
  Ten distinct shared affect/register directions, not just one or two
  collapsed axes. (Raw CCA on full hidden space gives spurious 1.000
  across the board because rank ≥ n_samples; the script uses PCA(20)
  prefix + held-out split for honest numbers.)
- **Procrustes alignment of quadrant geometry (gemma L31 ↔ qwen L61)**:
  orthogonal best-fit rotation between PCA(2) quadrant centroids is
  **+7.8°**, residual 5.7 (vs Frobenius norm of gemma centroids ~13).
  The deepest-deepest pair gave +14.0° / 6.4 — switching gemma to its
  preferred layer cut the rotation in half. Russell circumplex shape is
  more aligned across models when each model is read at its affect peak.
  Within-shape spread still differs (qwen's LN/HN/HP centroids range
  from −32 to +44 in PC1; gemma's L31 range is −9 to +12 — qwen's
  internal affect axis is several times longer in absolute scale).
- Outputs: `figures/local/cross_model/fig_v3_cka_per_layer.png`,
  `fig_v3_cca_canonical_correlations.png`, `fig_v3_quadrant_geometry_compare.png`,
  `v3_cka_per_layer.tsv`, `v3_cross_model_summary.json`.

**PC3+ analysis (`scripts/24_v3_pca3plus.py`).** Fit PCA(8) on v3 h_mean
and cross-reference each PC against all 5 saklas probe scores at t0
(whole-generation aggregate) and tlast (final-token).
- **Gemma t0 (L31)**: PC1 absorbs valence (happy.sad r=−0.69, angry.calm
  r=+0.46 — valence-collapse persists on PC1; this is structural to
  gemma's probe geometry, layer-independent). PC2 absorbs a humor +
  warmth + arousal mix (humorous.serious r=+0.42, warm.clinical r=−0.39,
  angry.calm r=−0.33). PC3-PC8 carry no probe signal above |0.3|. Very
  similar to L57's loadings (PC1 −0.74 / +0.47) — the probe-space
  geometry is set by saklas's probe layer, not affected by which layer
  we PCA on.
- **Qwen t0 (L61)**: PC1 absorbs valence + humor jointly (happy.sad
  r=−0.86, humorous.serious r=−0.69). PC2 absorbs certainty
  (confident.uncertain r=−0.48). PC3 absorbs arousal + warmth (angry.calm
  r=−0.61, warm.clinical r=+0.48 — anti-correlated, the negative-cluster
  arousal axis).
- The qwen-vs-gemma probe-geometry divergence (r=−0.117 vs r=−0.939
  between mean happy.sad and mean angry.calm across faces) has a clean
  PCA explanation: on gemma both probes load on PC1+PC2; on qwen they
  load on different PCs (PC1 vs PC3). Different decompositions of the
  same underlying affect space. This is unchanged at L31.
- tlast (final-token snapshot used by saklas's default scoring) shows
  much weaker PC×probe correlations on gemma than t0 does — confirms the
  saklas `stateless=True` per-generation aggregate is the better readout
  on gemma. On qwen tlast still shows PC1↔happy.sad r=−0.62.
- Outputs: `figures/local/{gemma,qwen}/fig_v3_pca3plus_quadrants.png`,
  `fig_v3_pca_probe_correlations.png`, `v3_pca_probe_correlations.tsv`.

**Kaomoji predictiveness (`scripts/25_v3_kaomoji_predictiveness.py`).**
Per-model two-direction fidelity: how well does kaomoji choice pin
down state, and vice versa. h_mean at each model's preferred layer.
Faces filtered to n ≥ 5 to keep per-class estimates stable.

- **Hidden → face (multi-class logistic on PCA(50)-reduced h_mean,
  5-fold stratified CV)**:
    * Gemma (19 face classes kept of 32): top-1 accuracy **0.712**,
      macro-F1 **0.521**. Majority baseline 0.233, uniform 0.053.
    * Qwen (28 face classes of 64): top-1 accuracy **0.495**,
      macro-F1 **0.298**. Majority baseline 0.143, uniform 0.036.
    * Both at 13–14× uniform. The model's hidden state predicts
      which face it emits well above chance; the gemma–qwen gap is
      mostly a class-count effect (more faces = harder classification),
      seen also in the macro-F1 ordering.
- **Hidden → quadrant** (5-class, same pipeline): both models
  **1.000 accuracy**. Caveat: 5-fold CV is by row, not by prompt;
  with 8 seeds × 100 prompts the same prompt can appear in train and
  test folds with different seeds. Rigorous version is leave-prompts-
  out CV — flagged as an open follow-on. Even with that caveat, the
  fact that quadrant labels are exactly recoverable from h_mean (with
  a much-larger-than-PC1+PC2 PCA prefix) confirms the v3 quadrant
  signal isn't just visible in PC1+PC2; it's saturating the available
  classifier capacity.
- **Face → hidden (η² of face identity per PC)**:
    * Gemma top-5 PCs: η² = 0.62 / 0.36 / 0.44 / 0.30 / 0.28; weighted
      by explained-variance, **0.194 of total**, **49% of top-5
      subspace**.
    * Qwen top-5 PCs: η² = 0.81 / 0.53 / 0.54 / 0.13 / 0.36; weighted
      **0.226 of total**, **60% of top-5 subspace**.
    * Qwen has slightly less PC variance in the top-5 (38.0% vs
      gemma's 39.6%) but face identity recovers more of it. Each qwen
      face is a tighter readout of its slice of the top-5 PC space
      than each gemma face is of gemma's; consistent with qwen's
      higher silhouette + more 2D circumplex shape.
- **Per-face (TSV at `figures/local/<short>/v3_kaomoji_predictiveness.tsv`)**:
  high-frequency faces are recoverable with 70–85% recall —
  gemma's `(๑˃‿˂)` (n=181, HP) recall 0.87, `(｡◕‿◕｡)` (n=75) 0.71;
  qwen's `(≧‿≦)` (n=106) recall 0.84, `(;ω;)` (n=82) 0.72. Some
  low-n distinctive faces have 0 recall (model confuses them with
  same-quadrant siblings) — gemma `(⊙﹏⊙)` (n=6, HN), qwen `(;;)`
  (n=11, LN). Distinctiveness of `1 − cos(face_mean, other_face_means)`
  is mostly in [0.93, 1.13]; high-distinctiveness faces are not
  always high-recall (model can know a face is unique without picking
  it correctly under low-data conditions).
- **Concrete reconstruction quality (full hidden space, predict
  ``h_mean = face_centroid(face_i)``)**: gemma R²=0.260 (mean
  centered cos +0.486, median +0.550, ‖err‖/‖dev‖ = 0.857); qwen
  R²=0.287 (cos +0.523, median +0.537, ‖err‖/‖dev‖ = 0.838).
  Quadrant-centroid baseline gets R² = 0.254 (gemma) / 0.264 (qwen)
  — face identity buys only **+0.6 pp** (gemma) / **+2.3 pp** (qwen)
  over the 5-class quadrant centroid in full hidden space. This
  reconciles with the 49–60% top-5-PC η² above: the kaomoji is a
  tight readout of the affect-relevant axes (top-5 PCs) and roughly
  independent of the bulk of hidden state, which is content-related.
  The marginal information beyond quadrant is concentrated where
  affect lives.

**Open follow-ons surfaced by these analyses:**
- All v3 + v1/v2 + cross-pilot scripts (04, 10, 13, 17, 22, 23, 24, and
  02 once v1/v2 sidecars land) now read at gemma's L31 by default
  via the `preferred_layer` field on `ModelPaths`. The L57 numbers —
  PC1 13%, HN/LN PC1-collapse, "1D-with-arousal-modifier" framing —
  are superseded. README's Local-side paragraph and Findings summary
  reflect L31; `docs/local-side.md` was not part of this refresh and
  may still cite the old numbers.
- Qwen has 16 cross-quadrant emitters with classifiable internal state
  but no separate face — wider net for natural-experiment work.
- The remaining 7.8° Procrustes rotation between gemma L31 and qwen L61
  is non-trivial. Asks whether the two models' quadrant axes are shifted
  by a consistent affine map (testable by Procrustes-aligning the per-
  row centroids for each kaomoji emitted by both models, not just the
  quadrant centroids).
- L31 was found via h_mean silhouette (script 21 iterates layers with
  `which="h_mean"`). The v1/v2 + cross-pilot scripts use `which="h_last"`
  at L31 — assumes "best layer for affect" is snapshot-independent.
  Worth re-running script 21 with `which="h_last"` to verify the peak
  layer doesn't shift; if it does, v1/v2 gets a separate
  `preferred_layer_h_last` or the loaders need a layer-by-snapshot map.
- Script 25's quadrant classifier hits 1.000 because 5-fold CV doesn't
  hold out by prompt — with 8 seeds × 100 prompts, the same prompt
  appears in train AND test folds (different seeds). The face
  classifier is less affected because each prompt elicits multiple
  faces across seeds, but the quadrant number is genuinely inflated.
  Need a `GroupKFold` split keyed on `prompt_id` for the rigorous
  version; would expect quadrant accuracy to drop to roughly the
  silhouette-implied level (~0.7–0.8) if there's any prompt-leakage.

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

### `preferred_layer` on `ModelPaths` overrides the loader default

Per-model peak-affect layer for v3 hidden-state reads, set on the
`ModelPaths` dataclass in `llmoji_study.config`. Gemma's is 31 (peak
silhouette per `scripts/21`); qwen's is None (defaults to deepest L61,
which is also peak); ministral has no v3 data yet. v3 scripts
(04, 13, 17, 22, 23, 24) all pass `layer=M.preferred_layer` to
`load_emotional_features` so figures get the right snapshot per model.

If you call `load_hidden_features` / `load_emotional_features` directly
in a notebook or new script, you have to remember the override —
``layer=None`` always means "deepest", regardless of model. Easiest
convention: ``layer=current_model().preferred_layer``.

The cache files at `data/cache/v3_<short>_h_mean_all_layers.npz`
contain ALL layers, so script 21 (which iterates over layers)
doesn't depend on `preferred_layer`. Same for the per-layer CKA
grid in script 23.

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

# v3 follow-on analyses (2026-04-28; uses existing sidecars, no model time)
python scripts/21_v3_layerwise_emergence.py        # multi-layer, both models in one run
python scripts/22_v3_same_face_cross_quadrant.py   # respects LLMOJI_MODEL; -W ignore::FutureWarning recommended
python scripts/23_v3_cross_model_alignment.py      # gemma↔qwen, both required
python scripts/24_v3_pca3plus.py                   # respects LLMOJI_MODEL
python scripts/25_v3_kaomoji_predictiveness.py     # both models in one run; -W ignore::FutureWarning recommended

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
  data/cache/                  # multi-layer h_mean tensors (gitignored)
  data/harness/{claude,codex}/ # per-provider per-project TSVs (tracked)
  figures/
    harness/                   # contributor-corpus figures (eriskii clusters,
                               # claude_faces PCA, per-provider per-project
                               # axes from the side script)
      claude/                  # per_project_axes_{mean,std}.png
      codex/                   # per_project_axes_{mean,std}.png
    local/                     # local-LM v1/v2/v3 figures
      cross_model/             # gemma↔qwen alignment (CKA grid, CCA bars, Procrustes)
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
