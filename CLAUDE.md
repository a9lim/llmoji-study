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
- **Rule 3 redesign landed 2026-05-01 — RULE 3b CONFIRMED on balanced data.**
  Design doc `docs/2026-05-01-rule3-redesign.md`. New `pad_dominance` field
  on `EmotionalPrompt` (orthogonal to `quadrant`); HN bisected into HN-D
  (anger/contempt) and HN-S (fear/anxiety), 3 borderline reads
  (hn06/hn15/hn17) untagged. 23 supplementary prompts (hn21–hn43, 13 D + 10
  S) brought the post-supp balance to 20/20 (160/160 rows per model).
  **Final verdict on balanced data:** rule 3a (powerful.powerless) DROPPED —
  wrong direction on most aggregates across (gemma, qwen, ministral) ×
  (t0, tlast, mean), so the probe doesn't read PAD dominance in the HN
  context. **Rule 3b (fearful.unflinching) PASS on all 3 models** —
  directional + bootstrap 95% CI excludes zero on ≥2 of 3 aggregates per
  model. Largest effects: qwen t0 (Cohen's d=+2.35), ministral mean
  (d=+0.81). Auto-generated verdict block at
  `figures/local/cross_model/rule3_dominance_check.md` from
  `scripts/local/30_rule3_dominance_check.py`. Triplet Procrustes
  (`scripts/local/31_v3_triplet_procrustes.py`,
  `figures/local/cross_model/fig_v3_triplet_procrustes_pc{12,13,23}.png`) — 2×2
  layout: gemma / qwen / ministral centroids in their own PCA(2),
  plus a Procrustes overlay showing all three aligned to gemma
  (○ gemma, △ qwen, □ ministral). Alignment-to-gemma residuals: qwen
  5.6, ministral 6.4 (after ministral's −176° axis flip — PCA sign
  indeterminacy, not a divergence finding). Same order of magnitude
  despite ministral's smaller scale + different lab. Display palette: HN-D `#d44a4a`
  (red, inherits HN), HN-S `#9d4ad4` (magenta-purple). New helpers
  `apply_hn_split` / `_palette_for` / `_hn_split_map` in
  `emotional_analysis`. Ministral `preferred_layer` set to L21 in
  `MODEL_REGISTRY` at the time of this landing (later updated to L20
  in the 2026-05-02 h_first cutover, see below).
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
  on further design discussion + ethics review of trial scale.**
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
- **v1.0 package split (2026-04-27):** `llmoji` (PyPI) owns taxonomy /
  canonicalization / hook templates / scrape / backfill / synth prompts;
  this repo's package was renamed `llmoji_study` and depends on
  `llmoji>=1.0,<2`.

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
python scripts/local/31_v3_triplet_procrustes.py    # → fig_v3_triplet_procrustes_pc{12,13,23}.png

# Introspection pilot (gemma + ministral; archive-bound — see CLAUDE.md status)
python scripts/local/32_introspection_pilot.py
python scripts/local/33_introspection_analysis.py
python scripts/local/34_introspection_predictiveness.py

# Blog-post figure regen → ../a9lim.github.io/blog-assets/introspection-via-kaomoji/
python scripts/local/35_regen_blog_figures.py

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
                               # smoke). 21 files: 01–04, 11, 21–35, 99.
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
