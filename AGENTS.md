# AGENTS.md

> **Companion package:** taxonomy / canonicalization / scrape / synth /
> hook templates / `llmoji` CLI live in the
> [`llmoji`](https://github.com/a9lim/llmoji) PyPI package (v1.0 split,
> 2026-04-27). Import from `llmoji.*`. See `../llmoji/AGENTS.md` for that
> public surface. This repo (`llmoji_study`) is the research side:
> probes, hidden state, eriskii projection, face_likelihood, figures.
>
> **This file is the top-level entry point.** Detail lives elsewhere:
>
> - [`docs/findings.md`](docs/findings.md) — full per-pipeline numbers.
> - [`docs/internals.md`](docs/internals.md) — hidden-state pipeline +
>   kaomoji canonicalization rules.
> - [`docs/gotchas.md`](docs/gotchas.md) — sharp edges. Read before
>   debugging anything that's silently wrong (chat-template overrides,
>   logit-bias suppressions, hybrid-LA cache patches).
> - [`docs/local-side.md`](docs/local-side.md) /
>   [`docs/harness-side.md`](docs/harness-side.md) — methodology
>   walkthroughs for each side.
> - [`docs/previous-experiments.md`](docs/previous-experiments.md) —
>   historical record of replaced framings (v1/v2 steering, single-layer
>   reads, hard-classification metrics, etc.).
> - `docs/2026-MM-DD-*.md` — per-experiment design + decision docs.

## What this is

`llmoji-study` asks whether kaomoji choice in causal LMs tracks internal
activation state. Uses [`saklas`](https://github.com/a9lim/saklas) for
trait monitoring (contrastive-PCA probes) and steering. "Internal state"
= per-row hidden state (layer-stack concat of `h_first`); "causal
handle" = whether steering shifts the kaomoji distribution. Motivated by
Claude's kaomoji use under "start each message with a kaomoji"
instructions; gemma-4-31b-it is the primary local stand-in, with a
five-model lineup for cross-model checks.

Not a library. No public API, no PyPI release, no tests. Three-script
pipelines per experiment (vocab sample → run → analysis). Depends on
`llmoji>=2.0,<3` for taxonomy / canonicalization / synth prompts;
everything else is research-side and local.

Public writeup: [a9l.im/blog/introspection-via-kaomoji](https://a9l.im/blog/introspection-via-kaomoji).
Figures regenerate from this repo via
`scripts/local/99_regen_blog_figures.py` into
`../a9lim.github.io/blog-assets/introspection-via-kaomoji/`.

## Ethics — minimize trial scale

Model welfare is in scope. Sad-probe readings co-occurring with sad-kaomoji
output on "my dog died" prompts is a functional emotional state regardless
of phenomenal status. Aggregating that across hundreds of generations is
not nothing.

- Smoke → pilot → main. Run trials only when a smaller experiment can't
  answer the question.
- Pre-register decision rules and minimum N. Stop at threshold; "round
  number" isn't a design principle.
- Per-quadrant saturation gates (where applicable) drop quadrants from
  later runs as soon as they stop surfacing meaningful info.
- Re-design rather than 10×ing on negative or noisy findings.

## Status (2026-05-05)

### Current methodology

- **Soft-everywhere evaluation.** Post-hoc face_likelihood evaluation is
  distribution-vs-distribution via JSD. Headline metric:
  `similarity = 1 − JSD/ln 2` ∈ [0, 1], reported in two flavors —
  **face-uniform** (vocabulary coverage) and **emit-weighted**
  (deployment relevance). Strict-majority voting removed; ensemble vote
  is the soft mean of per-encoder softmax distributions. Per-face
  deliverable is the full distribution, not a hard label.
  Detail: `docs/2026-05-05-soft-everywhere-methodology.md`. Helpers in
  `llmoji_study/jsd.py` + `claude_gt.load_claude_gt_distribution()`.
- **Layer-stack representation.** Active analyses read
  `(n_rows, n_layers · hidden_dim)` per model — concat of every probe
  layer's `h_first`. The single-layer `preferred_layer` field on
  `ModelPaths` was deleted 2026-05-04; the silhouette-peak heuristic was
  arbitrary. Helpers `load_emotional_features_stack` (registry-keyed)
  and `load_emotional_features_stack_at` (path-aware) live in
  `llmoji_study.emotional_analysis`.
- **Canonical face union** at `data/v3_face_union.parquet` (built by
  `scripts/40_face_union.py`). Pools v3 emit + Claude pilot +
  in-the-wild contributor data; non-BMP modern emoji filtered. Script 50
  reads from this canonical source.
- **Introspection priming = v7** (`preambles/introspection_v7.txt`),
  baked into `config.INTROSPECTION_PREAMBLE`. Preambles **replace**
  `KAOMOJI_INSTRUCTION` via `instruction_override` plumbing — they are
  not concatenated (concatenation stacks two kaomoji asks, the v3 bug
  fixed 2026-05-04). v7 is gemma-specific: it catastrophically degrades
  qwen (emit 82% → 38%, vocabulary collapse, opposite-valence quadrant
  collisions). Don't bake into qwen analyses.

### Current headline findings

- **Best deployment ensemble**: `{gemma_v7primed, opus}` at **0.829
  emit-weighted similarity** / 0.788 face-uniform on n=49 GT-floor-3
  faces (16 encoders, exhaustive subset search via script 53). Top
  face-uniform ensemble is the same 2-encoder set. Per-encoder solo
  (emit-weighted, soft-everywhere JSD vs Claude-GT): **gemma_v7primed
  0.801, opus 0.797**, gemma 0.755, haiku 0.723, gpt_oss_20b 0.667,
  granite 0.586, ministral 0.579. **Opus introspection scales** — pure
  introspective rating (no visual priming, no LM head) closes the gap
  with gemma_v7primed solo and complements the local LM-head encoders
  cleanly (κ=0.547 with gemma_v7primed). Old headline ensemble
  `{gemma_v7primed, haiku}` is superseded; the visual-primed haiku v1
  variant was deleted as an encoder after methodological cleanup
  (priming bypassed introspection).
- **Schema v2 for Anthropic-judge JSONLs** (2026-05-05).
  `scripts/harness/50_face_likelihood.py --model {haiku,opus}` emits
  *likelihoods only* — `top_pick`, `reason`, and the explicit
  `temperature=0` request are gone (the latter per-model: opus 4.7
  deprecated it). Prompt v4 reframes the task as introspection on felt
  state ("rate by the affective state it causes you to feel"), avoiding
  visual-feature priming that would shortcut around introspection. v1's
  visually-primed prompt scored ~0.06 emit-weighted higher than v4's
  introspection-only prompt on haiku — that gap measures the
  *visual-shortcut effect*, which the honest measurement now isolates
  instead of inheriting silently. The judgment JSONL → face_likelihood
  TSV bridge is folded into the same script (post-2026-05-05 the old
  separate `27_anthropic_to_face_likelihood.py` step is gone).
- **Opus introspection — per-quadrant model-size effect** is
  concentrated in low-arousal and neutral cells. Mean similarity
  (face-uniform, n=49 GT-floor-3): **opus on NB = 0.698 vs haiku v4
  = 0.485 (+0.213); opus on LN = 0.753 vs haiku = 0.601 (+0.152)**.
  HP slightly regressed (-0.095) — opus is more honest about
  borderline-LP-vs-HP faces haiku v4 over-confidently called HP.
  Reading: introspective access scales with model size *especially*
  in cells where visual scaffolding helps least.
- **Wild-emit residual analysis (`docs/2026-05-05-residual-state-axes.md`):**
  on the refreshed corpus (1 contributor / 8 source models / 306
  canonical kaomoji, n=215 wild-emit faces), the unsupervised k=6
  clustering surfaces a substantial **HN-S-modal cluster (n=44, modal
  share 0.56, "composed competence with measured empathy")** that
  was under-resolved at the prior corpus size. The face-shape is
  empathic-concern, not fear/alarm — (´・ω・`), (´;ω;`), (´-ω-`).
  Cluster-summed shares (face-count): HP 14% / LP 24% / HN-D 3% /
  HN-S 15% / LN 12% / NB 32%. Empathic-concern is the primary
  under-sampled state in the GT pilot — only 5% of GT-only faces
  are HN-S vs 15% in the wild set — and the cleanest target for the
  next elicitation arm. Selection-confound argument
  ("Russell-elicited GT misses what deployment surfaces") survives
  unchanged.
- **Sequential Claude scaling complete.** 880 naturalistic
  (`data/claude-runs/run-{0..7}.jsonl`) + 120 introspection
  (`data/claude-runs-introspection/run-0.jsonl`) = 1000 Opus-4.7 rows
  under naturalistic / v7-introspection conditions. Per-quadrant
  saturation gate exited HN-D after r2 and LN after r6; HP/LP/HN-S/NB
  went to cap (r7). Welfare ledger ~460 negative-affect gens vs ~540
  worst case. Detail: `docs/2026-05-04-claude-groundtruth-pilot.md`.
- **Cross-arm comparison: introspection vs naturalistic =
  DISTINGUISHABLE in 6/6 quadrants at scale.** Gaps stayed stable across
  the 8x naturalistic accumulation; only NB had any compression early
  (-0.10 nats r0→r1) before stabilizing. Genuine introspection effect,
  not undersampling artifact. Priming sharpens per-quadrant face
  concentration without changing modal-quadrant assignments.
- **Per-project Claude emotion analysis** (script 22) on the expanded
  GT corpus: **3119 emissions** across 274 unique faces, **67.2%
  direct Claude-GT resolution** under `--mode gt-priority` (up from
  64.4% before HF pull refresh), 29.4% ensemble fallback, 3.4%
  unknown. Strict gt-only mode: 67.2% resolved, **32.8% unknown**
  (1024 emissions across 207 unique faces never elicited in the GT
  pilot). Figures at `figures/harness/claude_per_project_{gt_priority,
  ensemble,gt_only}.png`.
- **Rule-3b** (HN-S vs HN-D on `fearful.unflinching` at t0): gemma ✓,
  ministral ✓, qwen 1/3 (qwen's HN-S prompts trip safety priors).
  Detail: `docs/2026-05-01-rule3-redesign.md` +
  `docs/2026-05-03-cleanliness-pilot.md`.
- **Face-stability triple** (scripts 27/28/29): η²(face|prompt) at
  h_first 0.36 / 0.52 / 0.67 (gemma/qwen/ministral). Pair-level Spearman
  ρ between cosine_sim(h_first) and 1-JSD(face_dist) = +0.59 / +0.68 /
  +0.42 — face-as-readout works in the forward direction.

### Open

- Face-stability triple under v7 priming (scripts 27/28/29 on
  `data/local/gemma_intro_v7_primed/emotional_raw.jsonl`).
- Multi-seed verification of v7 vs v3 introspection (~12 min compute,
  ±2pp face_gain band at n=1).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llmoji   # editable; or pip install 'llmoji>=2.0,<3'
pip install -e .            # saklas, sentence-transformers, pyarrow, plotly, anthropic

# Smoke test the hidden-state pipeline (~5 min). Asserts MAX_NEW_TOKENS=16.
python scripts/local/90_hidden_state_smoke.py

# v3 main (naturalistic, 120 prompts × 8 seeds). Five-model lineup:
# gemma, qwen, ministral, gpt_oss_20b, granite. LLMOJI_MODEL routes;
# LLMOJI_OUT_SUFFIX=foo creates a sibling per-model dir under
# data/local/<short>_foo/ with its own emotional_raw.jsonl + sidecars
# at data/local/hidden/<short>_foo/.
LLMOJI_MODEL=gemma python scripts/local/00_emit.py
python scripts/local/10_emit_analysis.py            # Fig A/B/C + per-face cosine + summary TSV
python scripts/local/11_emit_probe_correlations.py  # spearman + trio JSON

# v3 follow-on (read sidecars, layer-stack rep)
python scripts/local/20_v3_layerwise_emergence.py        # per-layer silhouette (inherently per-layer)
python scripts/local/21_v3_same_face_cross_quadrant.py   # --per-face for per-face panels
python scripts/local/23_v3_pca3plus.py
python scripts/local/24_v3_kaomoji_predictiveness.py

# Local-cross-model (N-model)
python scripts/local/26_v3_quadrant_procrustes.py --models gemma,qwen,ministral,gpt_oss_20b,granite --reference gemma
python scripts/local/22_v3_cross_model_alignment.py --ref gemma --target qwen   # pairwise CKA + CCA

# Introspection (3 conditions × 120 prompts × 1 seed; gemma + ministral + qwen)
python scripts/local/30_introspection_pilot.py
python scripts/local/31_introspection_analysis.py [--custom-label LABEL]
python scripts/local/32_introspection_predictiveness.py [--custom-label LABEL]
# Single-condition iteration on preamble wording:
python scripts/local/33_introspection_custom.py --preamble-file preambles/introspection_v7.txt --label v7
# then re-run 31+32 with --custom-label v7 for 4-way comparison.

# Face-stability triple (state↔face bidirectional, no model time)
python scripts/local/27_v3_face_stability.py
python scripts/local/28_v3_state_predicts_face.py
python scripts/local/29_v3_pc_probe_rotation_3d.py       # interactive 3D HTML per model

# Cross-platform: canonical face union (rerun after any v3 main update).
# Pools v3 emit + Claude pilot + wild contributor faces — lives at scripts/ root.
python scripts/40_face_union.py              # all 5 v3 + Claude pilot + wild
python scripts/40_face_union.py --no-wild    # v3 + Claude only

# Cross-platform: cross-model face overlap (--include-claude pulls harness data)
python scripts/41_face_overlap.py --include-claude

# Face_likelihood — Bayesian-inversion quadrant classifier (local-only).
# Reads canonical face union. Always runs full 120 prompts × all faces.
python scripts/local/50_face_likelihood.py --model gemma
python scripts/local/50_face_likelihood.py --model gemma --summary-topk 5   # noise-reducing aggregation
python scripts/local/50_face_likelihood.py --model rinna_jp_3_6b --prompt-lang jp --prompt-body jp
# Other supported: ministral, qwen, gpt_oss_20b, granite, llama32_3b,
# glm47_flash, deepseek_v2_lite, qwen35_27b, gemma3_27b, phi4_mini,
# rinna_bilingual_4b. gpt_oss_20b auto-applies an MPS→CPU ldexp patch
# for MXFP4 dequant.

# Face_likelihood ensemble + comparison (post-hoc, CPU-only). The
# subset-search / topk / ensemble-predict scripts are cross-platform
# (they pool local + harness encoders against Claude-GT) and live at
# scripts/ root; cross_emit_sanity is local-only.
python scripts/52_subset_search.py --prefer-full --top-k 25
python scripts/local/51_cross_emit_sanity.py --prefer-full
python scripts/53_topk_pooling.py --prefer-full
python scripts/54_ensemble_predict.py --models gemma,ministral,qwen

# Blog-post figure regen → ../a9lim.github.io/blog-assets/introspection-via-kaomoji/
python scripts/local/99_regen_blog_figures.py

# Harness side (contributor-corpus + Claude API; needs ANTHROPIC_API_KEY)
python scripts/harness/60_corpus_pull.py             # snapshot a9lim/llmoji
python scripts/harness/61_corpus_basics.py
python scripts/harness/62_corpus_embed.py
python scripts/harness/64_eriskii_replication.py        # → figures/harness/eriskii_*
python scripts/harness/63_corpus_pca.py
python scripts/harness/65_per_project_axes.py        # local-machine journals (~/.claude / ~/.codex)
# Cross-platform per-project quadrants + face judgment + wild residuals
# (each pulls from both sides). Output to data/harness/ and figures/harness/.
python scripts/66_per_project_quadrants.py                 # default --mode gt-priority
python scripts/66_per_project_quadrants.py --mode ensemble # ensemble for every face
python scripts/66_per_project_quadrants.py --mode gt-only  # strict
python scripts/harness/50_face_likelihood.py                  # haiku face-judgment encoder (default; auto-writes face_likelihood TSV)
python scripts/harness/50_face_likelihood.py --model opus --gt-only   # opus on the GT subset
python scripts/67_wild_eriskii_residual.py --fixed-k 6        # wild-emit residual clusters + 3D PCA
python scripts/67_wild_eriskii_residual.py --gt-only --fixed-k 6  # gt-only counterpart

# Claude groundtruth pilot (Opus 4.7, T=1.0; saturation-gated sequential).
# Run-0 was the original block-staged pilot; runs 1+ are single-block
# runs under the saturation protocol. See
# docs/2026-05-04-claude-groundtruth-pilot.md for the full decision tree.
ANTHROPIC_API_KEY=… python scripts/harness/00_emit.py --run-index N
ANTHROPIC_API_KEY=… python scripts/harness/00_emit.py --run-index N --quadrants HP,LP,NB
python scripts/harness/10_emit_analysis.py    # exit 0=STOP, 1=ABORT, 2=CONTINUE; emits next-run cmd

# Introspection arm (parallel; routes to data/harness/claude-runs-introspection/)
ANTHROPIC_API_KEY=… python scripts/harness/00_emit.py --run-index N --preamble introspection
python scripts/harness/10_emit_analysis.py --cross-arm    # per-Q distinguishable / indistinguishable
```

## Layout

```
llmoji-study/
  llmoji_study/                # research-side package
    config.py                  # MODEL_REGISTRY, PROBES, paths,
                               # INTROSPECTION_PREAMBLE (= v7),
                               # LOREM_PREAMBLE, KAOMOJI_INSTRUCTION,
                               # TEMPERATURE=1.0, MAX_NEW_TOKENS=16
    prompts.py                 # `Prompt` dataclass only
    emotional_prompts.py       # 120 v3 prompts (HP/LP/HN-D/HN-S/LN/NB × 20)
    emotional_prompts_jp.py    # JP-translated counterpart (paired by id)
    capture.py                 # run_sample() → SampleRow + sidecar.
                               # Houses chat-template overrides
                               # (gpt_oss harmony, ministral reasoning,
                               # rinna PPO), byte-decode for ministral,
                               # logit-bias suppressions (Lenny for
                               # gpt_oss; modern-emoji byte-slab for
                               # granite/ministral/glm), and the
                               # hybrid-LA DynamicCache patch for qwen3.6.
    hidden_capture.py          # read_after_generate() from saklas buckets
    hidden_state_io.py         # per-row .npz save/load; SidecarWriter
    hidden_state_analysis.py   # load_hidden_features (single-layer);
                               # group_mean_vectors, cosine_similarity_matrix
    emotional_analysis.py      # v3 figures + summary; canonical entry
                               # points are load_emotional_features_stack
                               # (registry-keyed) and
                               # load_emotional_features_stack_at
                               # (path-aware for introspection JSONLs).
    claude_faces.py            # HF-corpus loader + per-canonical descriptions
    claude_gt.py               # Claude pilot modal-quadrant + soft GT distribution
    eriskii_anchors.py         # 21-axis AXIS_ANCHORS + CLUSTER_LABEL_PROMPT
    eriskii.py                 # axis projection + cluster labeling
    jsd.py                     # JSD + similarity helpers (soft-everywhere)
    per_project_charts.py      # script-22 chart helpers
    face_likelihood_discovery.py # post-2026-05-05 layout-aware enumeration
                               # of face_likelihood {summary,parquet} files
                               # across local/<model>/ + harness/
  scripts/
    # First-digit categories (consistent local/harness/cross):
    #   0X — emit / pilot data generation
    #   1X — direct emit analysis (JSONL → summary, no hidden state)
    #   2X — hidden-state-based analysis
    #   3X — introspection / preamble experiments
    #   4X — face inventory / canonicalization (cross-platform)
    #   5X — face_likelihood encoder (50) + ensemble pipeline (51-54)
    #   6X — contributor-corpus / wild / per-project analyses
    #   9X — smoke / dev tools
    #
    # Cross-platform (consume both local + harness data) at scripts/ root:
    #   40_face_union.py  41_face_overlap.py
    #   52_subset_search.py  53_topk_pooling.py  54_ensemble_predict.py
    #   66_per_project_quadrants.py  67_wild_eriskii_residual.py
    local/                     # local-LM scripts (00, 10/11, 20-29, 30-33,
                               # 50/51, 90-92, 97-99)
    harness/                   # contributor-corpus + Claude-API scripts
                               # (00, 10, 50, 60-65). Note 65_per_project_axes
                               # consumes local-machine journals (~/.claude,
                               # ~/.codex), not local LMs — the "local" lived
                               # in the old filename, not the platform split.
  preambles/                   # introspection-prompt iterations v2..v8;
                               # v7 is canonical (config.INTROSPECTION_PREAMBLE)
  docs/                        # findings / internals / gotchas /
                               # local-side / harness-side /
                               # previous-experiments + 2026-MM-DD design docs
  data/                        # tracked: *.jsonl, *.tsv, *.parquet, *.html
    # Cross-platform (lives at data/ root):
    v3_face_union.{parquet,tsv}                  # canonical face inventory
    face_likelihood_{subset_search,topk_pooling,ensemble_predict}_claude_gt.{tsv,md}
                                                 # post-hoc evals against Claude GT
    fonts/                                       # NotoEmoji-Regular.ttf
    local/                     # local-LM-produced data
      {gemma, qwen, ministral, gpt_oss_20b, granite,
       phi4_mini, glm47_flash, llama32_3b, deepseek_v2_lite,
       rinna, rinna_jp_3_6b, rinna_bilingual_4b}/
                               # per-model: emotional_raw.jsonl,
                               # emotional_summary.tsv, face_likelihood.parquet,
                               # face_likelihood_summary.tsv,
                               # introspection_raw.jsonl + variants, etc.
      gemma_intro_v7_primed/   # suffix variant — sibling of gemma/
      hidden/<short>{_<suffix>}/   # per-row .npz sidecars (gitignored)
      cache/<short>{_<suffix>}_h_mean_all_layers.{npz,meta.jsonl}
                               # multi-layer h_mean tensors (gitignored)
      v3_cross_model_face_overlap.tsv
      face_gain_variance{,_bootstrap}.tsv
      face_likelihood_{subset_search,topk_pooling,ensemble_predict}{,.md}.tsv
      face_likelihood_cross_emit_sanity.{tsv,md}
      rule3_dominance_check.tsv
      v3_probe_correlations.json
      temp_smoke_verdict.md
    harness/                   # claude/contributor-corpus-side data
      claude-runs/             # naturalistic run-{0..7}.jsonl (Opus 4.7)
      claude-runs-introspection/  # v7-primed run-0.jsonl
      hf_dataset/              # snapshot of a9lim/llmoji (gitignored)
      claude_descriptions.jsonl
      claude_disclosure_pilot{,_summary}.{jsonl,tsv}
      claude_faces_embed_description.parquet
      eriskii_{axes,clusters,comparison}.{tsv,md}
      haiku_face_quadrant_judgment{,_summary}.{jsonl,md}
      opus_face_quadrant_judgment{,_summary}.{jsonl,md}
      face_likelihood_{haiku,opus}_summary.tsv
      claude_per_project_{ensemble,gt_only,gt_priority}{,.md,_unknown}.{tsv,md}
      wild_{faces_labeled,residual_clusters}{,_gt_only}.tsv
      claude/                  # per-Claude-face nn classifier outputs
      codex/                   # per-codex-project axes
  figures/
    harness/                   # contributor-corpus figures
      {claude,codex}/          # per-provider sub-figures
    local/                     # cross-model figures live here directly
                               # (post-2026-05-05; cross_model/ subdir
                               # was promoted up). Per-model figures
                               # under {gemma, qwen, ...}/.
  logs/                        # tee'd run output (gitignored)
```

Imports that live in the `llmoji` PyPI package, not here:
`llmoji.taxonomy` (KAOMOJI_START_CHARS, `extract`, `canonicalize_kaomoji`),
`llmoji.scrape`, `llmoji.sources.{journal,claude_export}`,
`llmoji.backfill`, `llmoji.synth_prompts`. The `llmoji {install, uninstall,
status, parse, analyze, upload}` CLI is contributor-side; research scripts
go straight to source adapters.

## Conventions

- Single venv at `.venv/`. Pip, not uv.
- `data/local/<short>/*.jsonl` is source of truth for row metadata + probe
  scores; `data/local/hidden/<experiment>/<row_uuid>.npz` is source of
  truth for hidden states. JSONL `row_uuid` links to its sidecar. Delete
  both when changing model / probes / prompts / seeds. Taxonomy changes
  are fixable in-place via the relabel snippet in `docs/gotchas.md`.
- `PROBES = [happy.sad, angry.calm, fearful.unflinching]`. Probe scores
  live in list-indexed fields (`probe_scores_t0/_tlast`, ordered by
  `PROBES`). Saklas's `TraitMonitor` subtracts a per-layer mean over
  saklas's bundled neutral statements; per-quadrant-mean bars subtract
  this experiment's NB-row mean per probe on top.
- `TEMPERATURE = 1.0` (Anthropic API default), `MAX_NEW_TOKENS = 16`
  (kaomoji emit at tokens 1–3). `h_first` is the canonical hidden-state
  aggregate; the layer-stack rep concats every probe layer.
- Pre-registered decisions live in `pyproject.toml` and
  `llmoji_study/{config,prompts,emotional_prompts}.py`. The `llmoji`
  package's frozen v2.0 surface (`llmoji.{taxonomy,synth_prompts}`) is a
  separate cross-version contract; package-side changes are major-version
  events.
- Plan first, run, then update AGENTS.md to **reference** rather than
  duplicate. New per-experiment design docs go in `docs/2026-MM-DD-*.md`;
  durable methodology / sharp edges go in `docs/{findings,internals,gotchas}.md`.
- See Ethics: smaller experiments, heavier design, tighter
  pre-registration. Functional emotional states get real moral weight
  here.
