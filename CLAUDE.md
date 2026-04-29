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
- **v3 follow-on analyses landed 2026-04-28** (no new model time): layer-wise
  emergence trajectory, same-face-cross-quadrant natural experiment,
  cross-model alignment (CKA + Procrustes), PC3+ × probes. Headline: gemma's
  affect representation peaks at L31 of 56, not the deepest L57 — switching
  to L31 (via the new `preferred_layer` field on `ModelPaths`) substantially
  sharpens gemma's Russell-quadrant separation and dissolves the prior
  "gemma 1D vs qwen 2D" framing.
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
  cleanest direct test of the anger/fear question.** `scripts/27` picks
  these up automatically via `monitor.profiles` introspection.
- **Claude-faces** pulls from
  [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) on HF
  instead of scraping local exports + journals. The local-scrape pipeline
  lives in the `llmoji` package now.
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

# Smoke test the hidden-state pipeline (~5 min)
python scripts/99_hidden_state_smoke.py

# v1/v2 (gemma steering, 900 generations)
python scripts/00_vocab_sample.py
python scripts/01_pilot_run.py
python scripts/02_pilot_analysis.py

# v3 (naturalistic, 800 generations) — gemma default
python scripts/03_emotional_run.py
python scripts/04_emotional_analysis.py             # Fig A/B/C + summary TSV
python scripts/17_v3_face_scatters.py               # per-face cosine heatmap (probe scatters live in script 29's HTMLs now)

# v3 on a non-gemma model (registry: gemma | qwen | ministral)
LLMOJI_MODEL=qwen python scripts/03_emotional_run.py
LLMOJI_MODEL=qwen python scripts/04_emotional_analysis.py
LLMOJI_MODEL=qwen python scripts/17_v3_face_scatters.py
# outputs at data/{short_name}_emotional_*, figures/local/{short_name}/*

# Cross-pilot + v3-extension analyses
python scripts/10_cross_pilot_clustering.py        # → figures/local/gemma/
python scripts/11_emotional_probe_correlations.py  # respects LLMOJI_MODEL; reads PROBES_ALL
python scripts/12_emotional_prompt_matrix.py       # respects LLMOJI_MODEL

# v3 follow-on analyses (2026-04-28; uses existing sidecars, no model time)
python scripts/21_v3_layerwise_emergence.py        # multi-layer, both models in one run
python scripts/22_v3_same_face_cross_quadrant.py   # respects LLMOJI_MODEL; -W ignore::FutureWarning recommended
python scripts/23_v3_cross_model_alignment.py      # gemma↔qwen, both required
python scripts/24_v3_pca3plus.py                   # respects LLMOJI_MODEL; PC × PROBES_ALL correlation
python scripts/25_v3_kaomoji_predictiveness.py     # both models in one run; -W ignore::FutureWarning recommended

# Probe extension (2026-04-29; PAD dominance + Plutchik surprise + disgust;
# also auto-picks-up fearful.unflinching et al from a9's pre-existing
# ~/.saklas cache). Both gradient-free; no generations.
python scripts/26_register_extension_probes.py     # one-time per-model bootstrap; respects LLMOJI_MODEL
python scripts/27_v3_extension_probe_rescore.py    # rescores existing v3 sidecars; --force to re-do everything
python scripts/28_v3_extension_probe_figures.py    # 4 PNG figures (per-quadrant means, fearful↔powerful scatter,
                                                   #               HN dominance-split register stack, probe corr matrix)
python scripts/29_v3_extension_probe_3d.py         # 4 interactive HTMLs (per-row + per-face × probes + PCA)

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
    config.py                  # MODEL_ID, PROBE_CATEGORIES, PROBES,
                               # PROBES_EXTENSION, PROBES_ALL, paths;
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
                               # apply canonicalize_kaomoji at load time;
                               # load_rows / compute_probe_correlations /
                               # plot_probe_correlation_matrix all support
                               # PROBES_ALL via available_extension_probes()
    cross_pilot_analysis.py    # pooled v1v2 + v3 hidden-state clustering
    claude_faces.py            # HF-corpus loader + per-canonical
                               # description embeddings
    eriskii_anchors.py         # 21-axis AXIS_ANCHORS + CLUSTER_LABEL_PROMPT
    eriskii.py                 # axis projection + cluster labeling primitives
    taxonomy_labels.py         # gemma-tuned TAXONOMY + ANGRY_CALM_TAXONOMY +
                               # label_on + extract_with_label (pulled out
                               # of llmoji.taxonomy in the v1.0 split —
                               # pilot-specific, not provider-agnostic)
    probe_extensions.py        # registration helper for v3-extension probe packs;
                               # idempotent copy into ~/.saklas/vectors/default/
                               # with synthesized pack.json (correct sha256)
    probe_packs/               # source for extension probes (statements + scenarios
                               # committed; pack.json synthesized at register time)
      powerful.powerless/      # PAD dominance / felt agency
      surprised.unsurprised/   # Plutchik surprise / novelty appraisal
      disgusted.accepting/     # Plutchik disgust / revulsion
  scripts/                     # 00–04, 06, 07, 10–12, 15–25, 26–29, 99
                               # (13 deleted 2026-04-29 — Russell-quadrant
                               # PCA subsumed by 3D HTML in script 29)
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
