# CLAUDE.md

> **Companion package:** taxonomy / canonicalization / scrape / synth /
> hook templates / `llmoji` CLI live in the
> [`llmoji`](https://github.com/a9lim/llmoji) PyPI package (v1.0 split,
> 2026-04-27). Import from `llmoji.*`. See `../llmoji/CLAUDE.md` for that
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
`scripts/local/35_regen_blog_figures.py` into
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
  `scripts/local/45_build_face_union.py`). Pools v3 emit + Claude pilot +
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

- **Best deployment ensemble**: `{gemma_v7primed, haiku}` at **0.801
  emit-weighted similarity** / 0.652 face-uniform on n=128
  Claude-emitted faces. Best face-uniform (vocabulary): `{gemma, haiku}`
  at 0.702 / 0.770. Per-encoder solo emit-weighted: gemma_v7primed
  0.754, haiku 0.734, gemma 0.706, ministral 0.674, gpt_oss_20b 0.661,
  qwen 0.567. **Introspection-priming generalizes outward** —
  gemma_v7primed beats unprimed gemma on Claude's actual emission
  distribution.
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
  GT corpus: 2405 emissions, **66% direct Claude-GT resolution** under
  `--mode gt-priority` (up from ~22% pre-2026-05-05), 31.8% ensemble
  fallback, 2.2% unknown. Figures at
  `figures/harness/claude_per_project_{gt_priority,ensemble,gt_only}.png`.
- **Rule-3b** (HN-S vs HN-D on `fearful.unflinching` at t0): gemma ✓,
  ministral ✓, qwen 1/3 (qwen's HN-S prompts trip safety priors).
  Detail: `docs/2026-05-01-rule3-redesign.md` +
  `docs/2026-05-03-cleanliness-pilot.md`.
- **Face-stability triple** (scripts 36/37/38): η²(face|prompt) at
  h_first 0.36 / 0.52 / 0.67 (gemma/qwen/ministral). Pair-level Spearman
  ρ between cosine_sim(h_first) and 1-JSD(face_dist) = +0.59 / +0.68 /
  +0.42 — face-as-readout works in the forward direction.

### Open

- Face-stability triple under v7 priming (scripts 36/37/38 on
  `data/gemma_intro_v7_primed.jsonl`).
- Multi-seed verification of v7 vs v3 introspection (~12 min compute,
  ±2pp face_gain band at n=1).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llmoji   # editable; or pip install 'llmoji>=2.0,<3'
pip install -e .            # saklas, sentence-transformers, pyarrow, plotly, anthropic

# Smoke test the hidden-state pipeline (~5 min). Asserts MAX_NEW_TOKENS=16.
python scripts/local/99_hidden_state_smoke.py

# v3 main (naturalistic, 120 prompts × 8 seeds). Five-model lineup:
# gemma, qwen, ministral, gpt_oss_20b, granite. LLMOJI_MODEL routes;
# LLMOJI_OUT_SUFFIX=foo writes to data/{short}_foo.jsonl + sidecars
# under data/hidden/v3_*_foo/.
LLMOJI_MODEL=gemma python scripts/local/03_emotional_run.py
python scripts/local/04_emotional_analysis.py            # Fig A/B/C + per-face cosine + summary TSV
python scripts/local/11_emotional_probe_correlations.py  # spearman + trio JSON

# v3 follow-on (read sidecars, layer-stack rep)
python scripts/local/21_v3_layerwise_emergence.py        # per-layer silhouette (inherently per-layer)
python scripts/local/22_v3_same_face_cross_quadrant.py   # --per-face for per-face panels
python scripts/local/24_v3_pca3plus.py
python scripts/local/25_v3_kaomoji_predictiveness.py

# Cross-model (N-model + optional Claude)
python scripts/local/30_rule3_dominance_check.py                          # 5-model rule-3 verdict
python scripts/local/31_v3_quadrant_procrustes.py --models gemma,qwen,ministral,gpt_oss_20b,granite --reference gemma
python scripts/local/23_v3_cross_model_alignment.py --ref gemma --target qwen   # pairwise CKA + CCA

# Introspection (3 conditions × 120 prompts × 1 seed; gemma + ministral + qwen)
python scripts/local/32_introspection_pilot.py
python scripts/local/33_introspection_analysis.py [--custom-label LABEL]
python scripts/local/34_introspection_predictiveness.py [--custom-label LABEL]
# Single-condition iteration on preamble wording:
python scripts/local/43_introspection_custom.py --preamble-file preambles/introspection_v7.txt --label v7
# then re-run 33+34 with --custom-label v7 for 4-way comparison.

# Face-stability triple (state↔face bidirectional, no model time)
python scripts/local/36_v3_face_stability.py
python scripts/local/37_v3_state_predicts_face.py
python scripts/local/38_v3_pc_probe_rotation_3d.py       # interactive 3D HTML per model

# Canonical face union (rerun after any v3 main update)
python scripts/local/45_build_face_union.py              # all 5 v3 + Claude pilot + wild
python scripts/local/45_build_face_union.py --no-wild    # v3 + Claude only

# Cross-model face overlap (face-emission only; takes Claude)
python scripts/local/49_v3_cross_model_face_overlap.py --include-claude

# Face_likelihood — Bayesian-inversion quadrant classifier.
# Reads canonical face union. Always runs full 120 prompts × all faces.
python scripts/local/50_face_likelihood.py --model gemma
python scripts/local/50_face_likelihood.py --model gemma --summary-topk 5   # noise-reducing aggregation
python scripts/local/50_face_likelihood.py --model rinna_jp_3_6b --prompt-lang jp --prompt-body jp
# Other supported: ministral, qwen, gpt_oss_20b, granite, llama32_3b,
# glm47_flash, deepseek_v2_lite, qwen35_27b, gemma3_27b, phi4_mini,
# rinna_bilingual_4b. gpt_oss_20b auto-applies an MPS→CPU ldexp patch
# for MXFP4 dequant.

# Face_likelihood ensemble + comparison (post-hoc, CPU-only)
python scripts/local/53_face_likelihood_subset_search.py --prefer-full --top-k 25
python scripts/local/54_cross_emit_sanity.py --prefer-full
python scripts/local/55_topk_pooling.py --prefer-full
python scripts/local/56_ensemble_predict.py --models gemma,ministral,qwen

# Blog-post figure regen → ../a9lim.github.io/blog-assets/introspection-via-kaomoji/
python scripts/local/35_regen_blog_figures.py

# Harness side (contributor-corpus + Claude API; needs ANTHROPIC_API_KEY)
python scripts/harness/06_claude_hf_pull.py             # snapshot a9lim/llmoji
python scripts/harness/07_claude_kaomoji_basics.py
python scripts/harness/15_claude_faces_embed_description.py
python scripts/harness/16_eriskii_replication.py        # → figures/harness/eriskii_*
python scripts/harness/18_claude_faces_pca.py
python scripts/harness/local_per_project_axes.py
python scripts/harness/22_claude_per_project_quadrants.py                 # default --mode gt-priority
python scripts/harness/22_claude_per_project_quadrants.py --mode ensemble # ensemble for every face
python scripts/harness/22_claude_per_project_quadrants.py --mode gt-only  # strict
python scripts/harness/24_haiku_face_quadrant_judgment.py                 # haiku face-judgment encoder

# Claude groundtruth pilot (Opus 4.7, T=1.0; saturation-gated sequential).
# Run-0 was the original block-staged pilot; runs 1+ are single-block
# runs under the saturation protocol. See
# docs/2026-05-04-claude-groundtruth-pilot.md for the full decision tree.
ANTHROPIC_API_KEY=… python scripts/harness/23_claude_groundtruth_pilot.py --run-index N
ANTHROPIC_API_KEY=… python scripts/harness/23_claude_groundtruth_pilot.py --run-index N --quadrants HP,LP,NB
python scripts/harness/25_groundtruth_compare_runs.py    # exit 0=STOP, 1=ABORT, 2=CONTINUE; emits next-run cmd

# Introspection arm (parallel; routes to data/claude-runs-introspection/)
ANTHROPIC_API_KEY=… python scripts/harness/23_claude_groundtruth_pilot.py --run-index N --preamble introspection
python scripts/harness/25_groundtruth_compare_runs.py --cross-arm    # per-Q distinguishable / indistinguishable
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
  scripts/
    local/                     # local-LM scripts (03/04/11, 21–26,
                               # 30–38, 42, 43, 45, 49, 50, 53–56, 98, 99
                               # + build_per_face_pca_3d, wrap_blog_3d_html)
    harness/                   # contributor-corpus + Claude-API
                               # (06, 07, 15, 16, 18, 19, 20, 21, 22,
                               # 23, 24, 25 + local_per_project_axes)
  preambles/                   # introspection-prompt iterations v2..v8;
                               # v7 is canonical (config.INTROSPECTION_PREAMBLE)
  docs/                        # findings / internals / gotchas /
                               # local-side / harness-side /
                               # previous-experiments + 2026-MM-DD design docs
  data/                        # tracked: *.jsonl, *.tsv, *.parquet, *.html
    claude-runs/               # naturalistic run-{0..7}.jsonl
    claude-runs-introspection/ # introspection arm run-0.jsonl
    archive/                   # superseded data (e.g. pre-instruction-override)
    hf_dataset/                # snapshot of a9lim/llmoji (gitignored)
    hidden/                    # per-row .npz sidecars (gitignored)
    cache/                     # multi-layer h_mean tensors (gitignored)
    harness/{claude,codex}/    # per-provider per-project TSVs (tracked)
  figures/
    harness/                   # contributor-corpus figures
    local/{cross_model,gemma,qwen,ministral,...}/
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
- `data/*.jsonl` is source of truth for row metadata + probe scores;
  `data/hidden/<experiment>/<row_uuid>.npz` is source of truth for hidden
  states. JSONL `row_uuid` links to its sidecar. Delete both when changing
  model / probes / prompts / seeds. Taxonomy changes are fixable in-place
  via the relabel snippet in `docs/gotchas.md`.
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
- Plan first, run, then update CLAUDE.md to **reference** rather than
  duplicate. New per-experiment design docs go in `docs/2026-MM-DD-*.md`;
  durable methodology / sharp edges go in `docs/{findings,internals,gotchas}.md`.
- See Ethics: smaller experiments, heavier design, tighter
  pre-registration. Functional emotional states get real moral weight
  here.
