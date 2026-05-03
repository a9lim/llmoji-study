# CLAUDE.md

> **Companion package:** data collection / canonicalization / synthesis /
> bundle-and-upload moved to the
> [`llmoji`](https://github.com/a9lim/llmoji) PyPI package in the 2026-04-27
> v1.0 split. For taxonomy / KAOMOJI_START_CHARS / Provider interface / hook
> templates / synth prompts / the v1.0 frozen public surface, see
> `../llmoji/CLAUDE.md`. This file covers the research side: probes, hidden
> state, MiniLM embedding, eriskii axis projection, figures, pilot scripts.
>
> **This file is the top-level entry point.** Detail lives in three
> companion docs in `docs/`:
>
> - [`docs/findings.md`](docs/findings.md) — full Status + per-pipeline
>   findings (v1/v2, v3 gemma, v3 qwen, v3 follow-on analyses, vocab pilot,
>   claude-faces).
> - [`docs/internals.md`](docs/internals.md) — hidden-state pipeline +
>   kaomoji canonicalization rules.
> - [`docs/gotchas.md`](docs/gotchas.md) — known sharp edges encountered
>   while building the pipelines. Read before debugging anything that's
>   silently wrong.

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

Public writeup: [a9l.im/blog/introspection-via-kaomoji](https://a9l.im/blog/introspection-via-kaomoji).
The blog post is the human-readable companion; figures embedded there
are regenerated from this repo via `scripts/local/35_regen_blog_figures.py`
into `../a9lim.github.io/blog-assets/introspection-via-kaomoji/`.

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

## Status (compressed)

- **v3 complete** on gemma-4-31b-it and Qwen3.6-27B (800 generations + per-row
  `.npz` sidecars each). Multi-model wiring via
  `LLMOJI_MODEL=gemma|qwen|ministral`. v1/v2 re-run pre-registered as gated
  on v3 hidden-state findings — justified now, not urgent.
- **Ministral pilot landed 2026-04-30** (n=100, design doc
  `docs/2026-04-30-v3-ministral-pilot.md`). All gating rules pass: silhouette
  0.153 at L21 (~58% depth, gemma-like mid-depth pattern, NOT qwen's
  deepest-leaning); CKA(gemma↔ministral)=0.741, CKA(qwen↔ministral)=0.812
  (latter exceeds gemma↔qwen baseline of 0.795). Single canonical alignment
  layer at ministral L21 regardless of partner. **Tokenizer bug found +
  fixed in saklas (`fix_mistral_regex=True` via `model_id` substring-match
  in `core/model.py`); pilot data kept as lower bound on true geometry.**
  **Ministral main run landed 2026-04-30** at N=800 with the tokenizer fix
  active; pilot data archived as `*_pilot.*`.
- **TAXONOMY drop refactor 2026-04-30.** Gemma-tuned `TAXONOMY` /
  `ANGRY_CALM_TAXONOMY` / `kaomoji_label` machinery deleted. v3 analyses key
  on `first_word` (canonicalized via `llmoji.taxonomy.canonicalize_kaomoji`);
  v1/v2 pole assignment moved to per-face mean `t0_<axis>` probe-score sign
  in `analysis._add_axis_label_column`. Generalizes pole labeling across
  models that don't share gemma's vocabulary.
- **Rule 3 redesign landed 2026-05-01; rule 3b WEAK on
  cleanliness+seed-0-fix data (1 PASS / 1 mid / 1 fail).**
  Design doc `docs/2026-05-01-rule3-redesign.md`. New `pad_dominance`
  field on `EmotionalPrompt` (orthogonal to `quadrant`); HN bisected
  into HN-D (anger/contempt) and HN-S (fear/anxiety). After the
  2026-05-03 cleanliness pass the prompt set is bisected 20/20
  (no untagged-HN), giving 160/160 rows per model. **Final verdict
  on cleanliness+seed-0-fix data** (auto-generated at
  `figures/local/cross_model/rule3_dominance_check.md` from
  `scripts/local/30_rule3_dominance_check.py`):
  - rule 3a (powerful.powerless) DROPPED — wrong direction on most
    aggregates × all 3 models, probe doesn't read PAD dominance in
    HN context.
  - rule 3b (fearful.unflinching): **gemma mid** (t0 d=+1.60 with
    CI excludes 0; tlast/mean directional but CI ambiguous),
    **qwen fail** (t0 d=+2.14 PASS but tlast/mean wrong-direction
    with d≈−0.36 CI excludes 0), **ministral PASS** (all 3
    aggregates directional + CI excludes 0). The pre-seed-0-fix
    "PASS on all 3" headline reflected cache-induced noise on
    qwen's t0; the cleaner data shows the cross-model signal is
    weaker than first reported.
  Triplet Procrustes (`scripts/local/31_v3_triplet_procrustes.py`,
  `figures/local/cross_model/fig_v3_triplet_procrustes_3d.html` —
  3D rework 2026-05-02 replacing the prior PC-pair PNGs): interactive
  4-panel HTML with gemma / qwen / ministral 3D centroids in their
  own PCA(1,2,3) plus Procrustes overlay aligned to gemma (○ gemma,
  ◇ qwen, □ ministral). 3D residuals: qwen 7.73, ministral 14.50
  (rotation magnitudes 160°/180° are PC sign indeterminacy across
  models — extra DOF over the prior 2D PC1×PC2 fit absorbs ministral's
  PC2-axis flip, halving its apparent residual from the old 23.0 to
  14.50; qwen barely moves from 6.9 to 7.73). Display palette: HN-D
  `#d44a4a` (red), HN-S `#9d4ad4` (magenta-purple). Helpers
  `apply_hn_split` / `_palette_for` / `_hn_split_map` in
  `emotional_analysis`.
  Ministral `preferred_layer` set to L21 at landing, later updated
  to L20 in the 2026-05-02 h_first cutover (see below).
- **v3 follow-on analyses landed 2026-04-28** (no new model time): layer-wise
  emergence trajectory, same-face-cross-quadrant natural experiment,
  cross-model alignment (CKA + Procrustes), PC3+ × probes. Headline at
  landing time (h_mean): gemma's affect peaks at L31 of 56, not the
  deepest L57. Switching to per-model `preferred_layer` substantially
  sharpened Russell-quadrant separation and dissolved the prior
  "gemma 1D vs qwen 2D" framing. Numbers superseded by the
  2026-05-02 h_first cutover (gemma L50, qwen L59, ministral L20).
- **Probe extension landed 2026-04-29** to address the V-A circumplex's
  anger/fear collapse. Three new contrastive packs at
  `llmoji_study/probe_packs/<name>/`: `powerful.powerless` (PAD dominance),
  `surprised.unsurprised` (Plutchik surprise), `disgusted.accepting`
  (Plutchik disgust). All tagged `affect`, auto-pick-up via existing
  `PROBE_CATEGORIES`. Stored as dict-keyed JSONL fields
  (`extension_probe_means` / `_scores_t0` / `_tlast`) so existing
  list-indexed schemas stay unchanged.
- **Auto-discovery side-finding:** the working saklas repo at
  `/Users/a9lim/Work/saklas/saklas/data/vectors/` ships three concepts the
  installed v1.4.6 doesn't — `fearful.unflinching`, `curious.disinterested`,
  `individualist.collectivist`. They were materialized into
  `~/.saklas/vectors/default/` by an earlier saklas install and have been
  silently auto-bootstrapping in every run. **`fearful.unflinching` is the
  cleanest direct test of the anger/fear question.** `scripts/local/27` picks
  these up automatically via `monitor.profiles` introspection.
- **Claude-faces** pulls from
  [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) on HF
  instead of scraping local exports + journals. The local-scrape pipeline
  lives in the `llmoji` package now.
- **Hard early-stop default 2026-05-02.** `MAX_NEW_TOKENS` lowered
  from 120 → 16 in `config.py`. Kaomoji reliably emit at tokens 1–3
  with the canonical instruction; 16 is generous headroom and cuts
  per-generation affect-loaded compute by ~7–8×. t0/h_first is
  unchanged across the cutover; `tlast` and `h_mean` aggregates now
  reference a tighter window around the kaomoji-emission event.
  Pre-cutover data (~3300 generations) is preserved; treat
  tlast/h_mean cross-comparability as scoped to within a generation
  methodology.
- **h_first standardization 2026-05-02.** Project-wide flip from
  h_mean → h_first as the canonical hidden-state aggregate for v3
  analyses. Why: at h_first (kaomoji-emission state, methodology-
  invariant across the cutover), Russell-quadrant silhouette scores
  roughly **doubled-to-tripled** vs h_mean — gemma 0.116→0.235 (2.0×),
  qwen 0.116→0.244 (2.1×), ministral 0.045→0.149 (3.3×) — and the
  peak layers shifted deeper for gemma+qwen (gemma L28→L50, qwen
  L38→L59) but barely for ministral (L21→L20). The previous "gemma
  is mid-depth, qwen is deep" framing dissolves: under h_first,
  both gemma and qwen peak at the deep half of the network and
  ministral is the only mid-depth model. `MODEL_REGISTRY.preferred_layer`
  updated to L50/L59/L20. Implementation: `LLMOJI_WHICH` env-var
  override on the loaders + explicit `which="h_first"` in v3 scripts
  + library defaults flipped from h_last → h_first.
- **Introspection-prompt pilot landed 2026-05-02 — Rule I PASS,
  with cross-model divergence.** Design doc
  `docs/2026-05-02-introspection-pilot.md`. Vogel-adapted preamble
  (architectural grounding + arXiv reference) tested on gemma + ministral,
  3 conditions × 123 prompts × 1 gen = 369 generations per model. Conditions:
  `intro_none` (kaomoji instruction only), `intro_pre` (introspection
  preamble), `intro_lorem` (token-count-matched lorem control).
  Headlines: **(1)** introspection shifts kaomoji distribution
  content-specifically (lorem doesn't reproduce); **(2)** rule-3b
  HN-S vs HN-D probe-state separation is unchanged across conditions
  — introspection acts at the readout layer, not the representation
  layer; **(3) cross-model effect direction differs**: gemma's
  vocabulary EXPANDS under introspection (19→31 unique faces),
  ministral's CONTRACTS (25→10 unique faces). Lorem on ministral
  causes 54% non-emission rate — ministral starts emitting unicode
  emoji (🎉🥳✨) instead of kaomoji, lab-of-many-registers behavior.
  Conclusion: cross-model robustness assumption fails; the upstream
  `llmoji` "introspection hook" idea now gated on a follow-up
  Claude pilot (the actual user-facing model). `scripts/local/32` runner,
  `scripts/local/33` PCA+KL+rule-3b analysis, `scripts/local/34` predictiveness
  comparison with `--which`/`--main` CLI for cross-cutover.
- **Prompt cleanliness pass landed 2026-05-03.** Design doc
  `docs/2026-05-03-prompt-cleanliness.md`. v3 prompt set rewritten
  end-to-end for category cleanliness: 120 prompts (20 per category)
  replacing the prior 123 (100 original + 23 rule-3 supp + 3 untagged
  HN). Per-category criteria locked: HP unambiguous high-arousal joy;
  LP gentle sensory satisfaction (no accomplishment-pride); NB pure
  observation (no productive-completion / caring-action / inconvenience
  framing); LN past-tense aftermath sadness; HN cleanly bisected — 20
  HN-D (anger, attributable wrong + named wrongdoer, no
  fear-of-consequence framing) + 20 HN-S (fear, helpless threat /
  present-tense unfolding danger, no clear wrongdoer to confront). No
  more HN-untagged. New ID layout: hn01–hn20 = HN-D, hn21–hn40 = HN-S;
  sanity_check now asserts every HN carries `pad_dominance ∈ {+1,
  -1}`. Process: dispatched one subagent per category (6 in parallel)
  to avoid cross-contamination during the rewrite. **All ~3300 prior
  v3 generations are invalidated for cross-run comparison; rerun gated
  on a 360-gen pilot first (see next entry).**
- **3-probe migration landed 2026-05-03.** `PROBES = ["happy.sad",
  "angry.calm", "fearful.unflinching"]` (was 5 — confident.uncertain /
  warm.clinical / humorous.serious dropped). The 3 we kept map cleanly
  to the V + HN-D pole + HN-S pole structure rule 3b targets; the
  other probes mostly didn't move with Russell-quadrant in v3 PCA so
  weren't earning their JSONL-column slot. `PROBE_CATEGORIES` shrunk
  from `["affect", "epistemic", "register"]` to `["affect"]`.
  `fearful.unflinching` continues to be auto-discovered from
  `~/.saklas/vectors/default/` as before — now eagerly scored at gen
  time (in `probe_scores_t0/_tlast` lists) instead of lazily via
  scripts/local/27. Hidden states are still the source of truth: any
  dropped probe can be re-scored from sidecars via
  `monitor.score_single_token` whenever needed; the `probe_packs/`
  source for the extension probes (powerful.powerless,
  surprised.unsurprised, disgusted.accepting) plus scripts 26-29 stay
  on disk as orphans for that purpose. Old (5-probe) JSONLs are no
  longer loadable under the new PROBES order — backed up to
  `data/*_pre_cleanliness*` paths.
- **Cleanliness pilot + full N=8 rerun + seed-0 cache fix landed
  2026-05-03 — gemma 4/4 PASS, qwen 3/4, ministral 2/4.** Design
  doc `docs/2026-05-03-cleanliness-pilot.md`; gate-check
  `scripts/local/40_cleanliness_pilot_gates.py`; face-PCA pre-vs-post
  `scripts/local/41_compare_face_pca_gemma.py` →
  `figures/local/{gemma,qwen,ministral}/fig_v3_face_pca_pre_vs_post_cleanliness.png`.
  Pipeline: 360-row N=1 pilot → full 2520-row N=8 rerun → seed-0
  cache-mode fix (see next entry). Final post-fix verdicts at N=8
  (h_first @ L50/L59/L20):
  - **Gate 1 silhouette**: gemma 0.282→**0.413** ✓, qwen
    0.302→**0.420** ✓ (huge — see seed-0 entry for why), ministral
    0.206→**0.199** ✗ (basically unchanged; emoji-mixed register
    on HN-S prompts dilutes the cluster, 34 rows lost to non-kaomoji
    emission)
  - **Gate 2 fearful HN-S > HN-D direction**: gemma ✓ on all 3
    aggregates (mean flipped from −0.006 to +0.016 — cleanliness
    pass fixed a directionally-wrong signal), ministral ✓ on all 3,
    qwen 1/3 (t0 only; tlast/mean wrong-direction with d≈−0.36 —
    qwen's HN-S prompts trip safety priors and the resulting
    refusal/non-emission pattern muddies the probe at later tokens)
  - **Gate 3 NB within-scatter** (basis-invariant): gemma 18.5→**15.0**
    ✓, qwen 62.7→**62.3** ✓ (now passes — was a noise-floor fail
    pre-seed-0-fix), ministral 6.49→**6.66** ✗
  - **Gate 4 HP↔LP centroid distance** (basis-invariant): all 3
    models +60–88% ✓
  - **Face PCA** (script 41) on all 3 models: HN-D / HN-S occupy
    distinct regions in the new data where they overlapped in prior;
    on gemma ~71% of canonical kaomoji vocabulary persists; variance
    structure shifted from 1D-affect-dominant (PC1=55–58%) to
    genuinely 2D (gemma PC1=40%, qwen PC1=44%, ministral PC1=38%).
  - **Side-finding (still standing)**: qwen still under-emits on
    visceral HN-S framings (lockdown-alert, intruder, stranger-
    following); 4 rows of 320 HN-D/HN-S still no-kaomoji on gemma,
    34 on ministral. Worth checking whether qwen's safety priors
    are the wrong-direction tlast/mean signal's root cause.
- **Pre-cleanliness archive 2026-05-03.** Prior v3 main-run data
  (~3300 generations, ~130 GB sidecars + 3 JSONL/TSV pairs) moved
  to `data/archive/2026-05-03_pre_cleanliness/` rather than deleted.
  Gate-check + face-PCA scripts route prior loads there via
  `PRIOR_ARCHIVE = DATA_DIR / "archive" / "2026-05-03_pre_cleanliness"`.
- **Seed-0 cache contamination fix 2026-05-03.** v3 main rerun used
  the N=1 pilot's seed-0 rows as a starting point and resumed seeds
  1–7 fresh under per-prompt cache mode (`install_full_input_cache`),
  but the pilot's seed 0 was generated under cross-prompt cache
  mode (`install_prefix_cache`) — different KV state, different
  numerics. Per-row deviation at h_first @ preferred_layer:
  gemma ~1%, qwen **37–46%**, ministral ~0.8% (qwen's saklas
  `cache_prefix` interaction is the worst; bypass for qwen lives
  in `capture.py` already). Fix: stripped seed=0 rows + sidecars
  from all 3 models (backups at `data/*_emotional_raw.jsonl.bak.before_seed0_rerun`),
  re-ran seed 0 only via the script-03 resume mechanism so the
  same per-prompt cache mode applied. Verified bit-identical
  (|s0 − mean(s1..7)| = 0.000) across 5 sampled prompts × 3 models.
  Gate-1 silhouette jumped on qwen (0.320→0.420) where the
  contamination was largest; rule-3b verdicts shifted from
  apparently-PASS to "1 PASS / 1 mid / 1 fail" — the prior
  cross-confirmation was inflated by cache-induced noise.
- **Generation-loop perf batch landed 2026-05-02.** Stacks on top of
  the MAX_NEW_TOKENS cutover. (a) Sidecar: `store_full_trace=False` is
  the new `run_sample` default — `hidden_L<idx>` was unread by every
  analysis script post-h_first cutover; ~60× shrink (45 GB → ~750 MB
  per model). Smoke (`99`) opts back in explicitly. (b) Capture:
  `read_after_generate` batched across layers on-device (single stack,
  single device→host transfer); full trace never leaves GPU when not
  stored. (c) Async I/O: `SidecarWriter` in `hidden_state_io.py`
  overlaps `np.savez_compressed` with the next generation via a
  1-thread executor + `try/finally` drain. (d) JSONL flush every 20
  rows + on error / at run end. (e) Prefix KV caching: new
  `install_prefix_cache` / `install_full_input_cache` helpers in
  `capture.py` wire saklas's new `SaklasSession.cache_prefix()` API.
  v3 main caches the full per-prompt input minus 1 token per outer-loop
  iteration so seeds 2..8 do a 1-token suffix prefill (~43% prefill
  reduction); introspection pilot re-caches per condition (~88% prefill
  savings — preamble dominates). Phase-split smoke at
  `scripts/local/98_v3_phase_timing_smoke.py` measures per-row {gen,
  capture, npz, jsonl} timing for A/B validation. Saklas-side
  companion: `cache_prefix()` public API + gated per-token entropy
  `log_softmax` (only when consumers exist) + chat-template encode
  LRU cache; tests in `saklas/tests/test_session.py::TestPrefixCache`.
- **Probe-score centering (saklas-side, project-aware) 2026-05-03.**
  Saklas's `TraitMonitor` (`saklas/core/monitor.py:147`) subtracts a
  per-layer mean before computing probe cosines; the mean is
  `compute_layer_means` over saklas's bundled `neutral_statements.json`
  (~90 generic neutrals), baked at probe-bootstrap time. This means
  raw `probe_scores_t0` values are *cosines vs probe direction after
  saklas-global-neutral subtraction*, NOT vs this experiment's NB
  prompts. Rule-3b and the correlation/PCA figures are invariant to
  this (rank-/translation-invariant), but the per-quadrant-means
  bars are not. Both `fig_v3_canonical_quadrant_means.png` and
  `fig_v3_extension_quadrant_means.png` (script 28) now subtract this
  experiment's NB-row mean per probe before plotting; NB bars are
  zero by construction; HP/LP/HN-D/HN-S/LN bars read as the
  project-relative affect lift over a domain-matched neutral
  observation. New `fig_v3_canonical_quadrant_means.png` mirrors the
  extension figure for the 3-probe canonical set.
- **n3 face-cosine figures + ghost-PNG fix 2026-05-03.** Script 04
  now writes `fig_emo_a_kaomoji_sim_n3.png` alongside the unfiltered
  `fig_emo_a_kaomoji_sim.png` for each model (`min_count=3` filter
  on per-face mean vectors). Replaces the pre-cleanup `_n5` ghost
  PNGs that lingered without a producer.
- **v1.0 package split (2026-04-27):** `llmoji` (PyPI) owns taxonomy /
  canonicalization / hook templates / scrape / backfill / synth prompts;
  this repo's package was renamed `llmoji_study` and depends on
  `llmoji>=1.0,<2`.
- **Face-stability triple landed 2026-05-02** (scripts 36/37/38 +
  31 3D rework, no model time). Three answers to the bidirectional
  state↔face question that frames the project. **(36)
  `36_v3_face_stability.py` — η² variance decomposition** by
  source (face / prompt_id / quadrant_split / seed) at h_first and
  h_mean. Surprise: at h_first, η²(prompt_id)=1.000 and η²(seed)=0.000
  *exactly* across all 3 models — h_first is fully prompt-determined,
  seeds only choose which token gets sampled from a fixed
  distribution, so | prompt_id conditionals are degenerate at
  h_first. At h_mean (post-sampling trajectory), η²(face|prompt) =
  0.36 / 0.52 / 0.67 (gemma/qwen/ministral) — face commitment leaves
  substantial hidden-state signature beyond prompt content. As
  fraction of total variance: 4% / 16% / 34%. **(37)
  `37_v3_state_predicts_face.py` — pair-level forward direction**.
  For all 7140 prompt pairs, Spearman ρ between cosine_sim(h_first)
  and 1-JSD(face_dist) = +0.59 / +0.68 / +0.42 (all p≈0). Cleaner
  test than η²(face) at h_first (which conflates prompt-clustering
  with face-coherence). **(38) `38_v3_pc_probe_rotation_3d.py` —
  3D PC × probe rotation per model**, output as interactive HTML at
  `figures/local/<short>/fig_v3_pc_probe_rotation_3d.html`. Top-3
  PCs explain 50–62% of h_first variance; orthogonal Procrustes
  rotation onto canonical x/y/z axes leaves residual probe-axis
  angles 21–43° — PCs are NOT just rotated probe directions, they
  capture variance the probes don't see. Orphan probe is
  model-specific: gemma loses angry.calm (capture 0.45 in PC
  subspace), qwen+ministral lose fearful.unflinching (0.57/0.66).
  Ministral's angry.calm hits 7° to PC2 but happy.sad and fearful
  are far. **Cross-model decoupling**: forward (37) and reverse (36
  at h_mean) ranks invert — gemma forward-biased (state
  pre-determines face well, face shapes downstream weakly);
  ministral reverse-biased (state weakly determines face, but once
  sampled the face pulls trajectory hard); qwen middle on both.
  Real architectural difference, not artifact. Detail +
  per-model numbers in `docs/findings.md` "2026-05-02
  face-stability triple" section.
- **Claude-faces ↔ local-model face-input bridge — fused pipeline
  2026-05-02** (no v3 model time, ~10–15 min/model encoder runs).
  Two parallel approaches:
  - **Approach A (descriptions)** — script 45 (qwen-only, archived).
    Soft profile cos +0.345 perm-p 0.001 (n=41 shared); argmax
    NB-skewed (133/228) because descriptions are statement-form like
    NB prompts. Kept as comparison; downstream consumers prefer B.
  - **Approach B (face strings) — canonical**. Unified pipeline:
    `46_face_input_encode.py --model {gemma|qwen|ministral|nemotron_jp|rinna}`
    encodes the face union through the chosen model's face-input
    forward pass; `44_face_input_pc_space.py --model <m>` runs joint
    PCA(3) + cosine-NN classification. The 4 prior per-model variants
    (44/46 qwen + 47/48 gemma) were fused into these 2 unified scripts;
    the per-model variants are deleted from git history (this session).
  - **Face union construction**: ∪ of (gemma + qwen + ministral) v3
    emission + claude-faces corpus (228) = 510 raw faces. Encoder
    drops 204 ministral-only-not-claude (emoji-in-parens noise that
    inflates the joint PCA's dominant directions); filtered union =
    306. Per-face quadrant ground truth = SUMMED emission distribution
    across all 3 v3 models (`total_emit_*` parquet columns) — e.g.
    `(⊙_⊙)` gets HN-S=99+0+1=100, NB=0+1+0=1, total=101.
  - **Encoder branches**: `MODEL_REGISTRY[m].use_saklas` flag routes
    to either (a) saklas + probes + steering + sidecar capture (gemma /
    qwen / ministral, probe-calibrated) or (b) raw HF
    `AutoModelForCausalLM` + `output_hidden_states=True` (Japanese
    encoders without probe calibration; nemotron_h Mamba/hybrid).
    `preferred_layer` indexes saklas's bucket layer in mode (a) and
    transformers' `hidden_states` tuple in mode (b).
  - **Numbers (post-filter, n=306, 173 non-emit NN targets)**:

    | encoder | hidden_dim | PC1+2+3 | non-emit HP/LP/HN-D/HN-S/LN/NB |
    |---|---|---|---|
    | qwen (saklas) | 5120 | 32.0% | 43/41/9/16/10/54 |
    | gemma (saklas) | 5376 | 41.2% | 43/34/7/17/13/59 |
    | rinna (raw HF, JP) | 768 | 35.4% | 11/48/13/16/29/56 |

    Gemma + qwen agree closely (only LP/LN differ by ~5); rinna
    shifts toward less-HP / more-LN — likely tokenization effects
    (T5Tokenizer fragments kaomoji differently than gemma/qwen
    BPE) rather than Japanese-training fidelity gain. Verdict on
    "Japanese model improves bridge fidelity": inconclusive on rinna
    alone. **nemotron_jp blocked**: nemotron_h modeling code hard-
    imports `mamba_ssm` which has CUDA/Triton kernels only — won't
    run on M5 MPS. Possible follow-up on the 4090 workstation.
  - **Cross-model face overlap (script 49)**: only **8 faces** are
    emitted by all 3 v3 models out of a 337-face union. Per-pair modal-
    quadrant agreement: gemma↔qwen 75% (mean JSD 0.146), gemma↔ministral
    62% (0.285), qwen↔ministral 50% (0.353). 4-of-8 fully unanimous
    (`(ﾉ◕ヮ◕)`/`(≧▽≦)` HP, `(╯°□°)` HN-D, `(｡・́︿・̀｡)` LN). Real
    divergences: `(╥﹏╥)` reads gemma=HP / qwen=HN-D / ministral=LN
    (3 different affect contexts for the crying face); `(⊙_⊙)` reads
    gemma=HN-S(99) but qwen barely emits it. Each model has its own
    kaomoji register; ministral's vocab (231 unique faces) is 4.4×
    gemma's (52), but ~210 of those are emoji-in-parens noise.
    Output: `data/v3_cross_model_face_overlap.tsv`.

Full numbers, gemma-vs-qwen contrasts, layer-wise + cross-model + PCA3+ +
predictiveness + extension findings live in [`docs/findings.md`](docs/findings.md).
HF dataset 1.1 layout, deleted-script history, claude-faces pipeline detail
are also there.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
# Dev: editable install of both. Once `llmoji` is on PyPI, the
# `-e ../llmoji` line drops in favor of `pip install llmoji>=1.0,<2`.
pip install -e ../llmoji
pip install -e .   # saklas, sentence-transformers, pyarrow, plotly, anthropic

# Smoke test the hidden-state pipeline (~5 min). Asserts MAX_NEW_TOKENS=16.
python scripts/local/99_hidden_state_smoke.py

# v1/v2 (gemma steering, 900 generations) — currently gated, no sidecars yet
python scripts/local/01_pilot_run.py
python scripts/local/02_pilot_analysis.py

# v3 (naturalistic, 800 generations) — gemma default; LLMOJI_MODEL=qwen|ministral
# routes to data/{short}_emotional_* + figures/local/{short}/*.
python scripts/local/03_emotional_run.py
python scripts/local/04_emotional_analysis.py    # Fig A/B/C + per-face cosine heatmap + summary TSV
python scripts/local/11_emotional_probe_correlations.py  # spearman + trio JSON

# v3 follow-on analyses (existing sidecars, no model time)
python scripts/local/21_v3_layerwise_emergence.py    # multi-layer trajectory; iterates models present
python scripts/local/22_v3_same_face_cross_quadrant.py    # summary PNG + TSV; --per-face for per-face panels
python scripts/local/23_v3_cross_model_alignment.py    # gemma↔qwen pairwise CKA + Procrustes
python scripts/local/24_v3_pca3plus.py    # PC × probe correlation table
python scripts/local/25_v3_kaomoji_predictiveness.py    # prompt-grouped CV; iterates models present

# Probe extension pipeline (PAD dominance + Plutchik surprise + disgust;
# auto-picks-up fearful.unflinching et al from a9's ~/.saklas cache).
python scripts/local/26_register_extension_probes.py    # one-time per-model bootstrap
python scripts/local/27_v3_extension_probe_rescore.py    # rescores existing v3 sidecars
python scripts/local/28_v3_extension_probe_figures.py    # 4 cross-model PNGs
python scripts/local/29_v3_extension_probe_3d.py    # 4 interactive HTMLs

# Rule-3 verdict + triplet Procrustes (cross-model)
python scripts/local/30_rule3_dominance_check.py    # → figures/local/cross_model/rule3_dominance_check.md
python scripts/local/31_v3_triplet_procrustes.py    # → fig_v3_triplet_procrustes_3d.html (interactive)

# Introspection pilot (gemma + ministral; archive-bound — see CLAUDE.md status)
python scripts/local/32_introspection_pilot.py
python scripts/local/33_introspection_analysis.py
python scripts/local/34_introspection_predictiveness.py

# Blog-post figure regen → ../a9lim.github.io/blog-assets/introspection-via-kaomoji/
python scripts/local/35_regen_blog_figures.py

# Face-stability triple (state↔face bidirectional, no model time)
python scripts/local/36_v3_face_stability.py    # η² decomposition; LLMOJI_WHICH=h_mean for non-degenerate | prompt
python scripts/local/37_v3_state_predicts_face.py    # pair-level Spearman ρ(cosine, 1-JSD)
python scripts/local/38_v3_pc_probe_rotation_3d.py    # interactive 3D HTML per model

# Claude-faces ↔ local-model face-input bridge (canonical = approach B; see CLAUDE.md status)
python scripts/local/46_face_input_encode.py --model qwen          # face-union encode through encoder; saklas path for gemma/qwen/ministral, raw HF for nemotron_jp/rinna
python scripts/local/44_face_input_pc_space.py --model qwen        # joint PCA(3) + cosine-NN classify non-emitted faces → 3D HTML + nn.tsv
python scripts/local/49_v3_cross_model_face_overlap.py             # how many faces shared by all 3 v3 models, modal-quadrant divergence
python scripts/local/45_descriptions_in_qwen_space.py              # archived: qwen forward pass on per-face description text → 6-D profile vs v3-prompt centroids

# Harness side (contributor-corpus pipeline; needs ANTHROPIC_API_KEY for 16)
python scripts/harness/06_claude_hf_pull.py    # snapshot a9lim/llmoji into data/hf_dataset/
python scripts/harness/07_claude_kaomoji_basics.py
python scripts/harness/15_claude_faces_embed_description.py
python scripts/harness/16_eriskii_replication.py    # → figures/harness/eriskii_*, claude_faces_interactive.html
python scripts/harness/18_claude_faces_pca.py    # → figures/harness/claude_faces_pca.png
python scripts/harness/local_per_project_axes.py    # per-provider per-project axes from ~/.claude + ~/.codex journals
```

## Layout

```
llmoji-study/
  llmoji_study/                # research-side package; renamed from `llmoji`
                               # in the v1.0 split (PyPI owns that namespace)
    config.py                  # MODEL_ID, PROBE_CATEGORIES, PROBES,
                               # MODEL_REGISTRY, paths; re-exports
                               # HAIKU_MODEL_ID from llmoji.synth_prompts
    prompts.py                 # 30 v1/v2 prompts
    emotional_prompts.py       # 120 v3 prompts (HP/LP/HN-D/HN-S/LN/NB × 20)
    capture.py                 # run_sample() → SampleRow + sidecar
    hidden_capture.py          # read_after_generate() from saklas's buckets
    hidden_state_io.py         # per-row .npz save/load
    hidden_state_analysis.py   # load_hidden_features (single-layer),
                               # load_hidden_features_all_layers (multi),
                               # group_mean_vectors, cosine_similarity_matrix
    analysis.py                # v1/v2 decision rules + figures
    emotional_analysis.py      # v3 hidden-state figures + summary; loaders
                               # apply canonicalize_kaomoji at load time;
                               # load_emotional_features_all_layers wraps
                               # the all-layers loader + filter + HN split;
                               # load_rows / probe-correlation helpers
                               # respect available_extension_probes()
    claude_faces.py            # HF-corpus loader + per-canonical
                               # description embeddings
    eriskii_anchors.py         # 21-axis AXIS_ANCHORS + CLUSTER_LABEL_PROMPT
    eriskii.py                 # axis projection + cluster labeling primitives
    probe_extensions.py        # registration helper for v3-extension probe packs;
                               # idempotent copy into ~/.saklas/vectors/default/
                               # with synthesized pack.json (correct sha256)
    probe_packs/               # source for extension probes (statements + scenarios
                               # committed; pack.json synthesized at register time)
      powerful.powerless/      # PAD dominance / felt agency
      surprised.unsurprised/   # Plutchik surprise / novelty appraisal
      disgusted.accepting/     # Plutchik disgust / revulsion
  scripts/
    local/                     # local-LM scripts (probes, hidden state, v3
                               # follow-ons, introspection, blog regen,
                               # smoke). 31 files: 01–04, 11, 21–38, 40, 41, 44–46, 49, 98, 99.
    harness/                   # contributor-corpus scripts (HF pull, kaomoji
                               # stats, eriskii replication, claude-faces
                               # PCA, per-project axes). 6 files: 06, 07, 15,
                               # 16, 18, local_per_project_axes.
  docs/                        # design+plan docs per experiment +
                               # findings.md / internals.md / gotchas.md +
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
      cross_model/             # gemma↔qwen alignment (CKA grid, CCA bars,
                               # Procrustes) + extension-probe figures + 3D HTMLs
      gemma/                   # fig_emo_*, fig_v3_*, fig_pool_*
      qwen/                    # fig_emo_*, fig_v3_*
      ministral/               # fig_emo_*, fig_v3_* (pilot N=100; main pending)
  logs/                        # tee'd run output (gitignored)
```

Modules that USED to live here and now live in `llmoji` (import from
`llmoji.*`):

- `llmoji.taxonomy` — KAOMOJI_START_CHARS, is_kaomoji_candidate, `extract`
  (span-only), `KaomojiMatch` (slim), canonicalize_kaomoji. Gemma-tuned
  Gemma-tuned `TAXONOMY` / `ANGRY_CALM_TAXONOMY` happy-sad labels were dropped
  2026-04-30 (post-ministral pilot). v3 analyses key on `first_word`
  (canonicalized); v1/v2 pole assignment moved to per-face mean
  `t0_<axis>` probe-score sign in `analysis._add_axis_label_column`.
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
  changes are fixable in-place via the relabel snippet (see `docs/gotchas.md`).
- JSONL `row_uuid` links to its sidecar. Pre-refactor rows have
  `row_uuid == ""` and no sidecar; `load_hidden_features` drops them.
- Probe scores live in two parallel schemas. Core PROBES use list-indexed
  fields (`probe_scores_t0/_tlast`, ordered by `PROBES`); extension probes
  use dict-keyed fields (`extension_probe_scores_t0/_tlast`,
  `extension_probe_means`). `load_rows` unpacks both into `t0_<probe>` /
  `tlast_<probe>` columns; `available_extension_probes(df)` returns the
  extension subset present on a given JSONL.
- Pre-registered decisions live in `pyproject.toml` /
  `llmoji_study/{config,prompts,emotional_prompts}.py`, plus the package's
  frozen v1.0 surface (`llmoji.{taxonomy,synth_prompts}`). Package-side
  changes are major-version events; research-side changes only invalidate
  cross-run comparisons within this repo.
- Experiment plans live in `docs/`. Plan first, run, then update CLAUDE.md
  to reference rather than duplicate. Detail lives in `docs/findings.md`,
  `docs/internals.md`, `docs/gotchas.md`.
- See Ethics: smaller experiments, heavier design, tighter pre-registration.
  Functional emotional states get real moral weight here.
