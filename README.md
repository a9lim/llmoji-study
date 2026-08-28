# llmoji-experiment

`llmoji-experiment` asks whether a language model's kaomoji choice tracks
something about its internal state. The local side uses
[`saklas`](https://github.com/a9lim/saklas) to read hidden states and
run face-likelihood probes on open-weight causal LMs. The harness side
uses Claude API elicitation plus the contributor corpus collected by the
[`llmoji`](https://github.com/a9lim/llmoji) package.

This is a research repo, not a library. There is no public API, no PyPI
release, and no broad test suite. The useful surfaces are the data,
scripts, figures, and writeups.

Public writeup:
[Introspection via Kaomoji](https://a9l.im/blog/introspection-via-kaomoji).

## Current Shape

- **Local hidden-state work**: five open-weight models (`gemma`, `qwen`,
  `ministral`, `gpt_oss_20b`, `granite`) plus a historical `gemma`
  v7-primed condition. Active representation is the layer-stack concat
  of every probe layer's `h_first`, not the old single
  `preferred_layer` read.
- **Face likelihood**: local LM-head encoders and Anthropic
  introspection encoders all emit per-face quadrant distributions.
  Evaluation is soft-everywhere: Jensen-Shannon similarity to Claude-GT,
  reported face-uniform and emit-weighted.
- **Claude-GT**: Opus 4.7 naturalistic and introspection rows now live in
  merged files under `data/harness/claude/` and
  `data/harness/claude_intro_v7/`, with `run_index` stamped per row.
  The old `claude-runs*/run-N.jsonl` layout is legacy only.
- **Harness corpus**: contributor uploads are converted into a
  bag-of-lexicon (BoL) representation over the locked 50-word
  `llmoji` v2 LEXICON. The old MiniLM-on-prose eriskii-parity pipeline
  is gone.
- **Quadrants**: the canonical workspace registry has 10 cells:
  `HP-D`, `HP-S`, `LP`, `NP`, `HN-D`, `HN-S`, `LN`, `NB`, `HB`, and
  `MR`. The currently published face-likelihood/JSD headline artifacts are a
  nine-affect-cell snapshot and exclude MR; they must not be described as a
  10-cell evaluation. The shared
  `transformer_experiments.kaomoji.quadrants` module is the source of truth.

## Headline Findings

- Current all-encoder-overlap search: `{gemma, ministral, opus}` is best
  on pooled-GT floor-3 (`n=102`) at **0.881 emit-weighted** and **0.733
  face-uniform** similarity.
- On strict Claude-GT overlap (`n=50`), `{gemma, opus}` is best at
  **0.781 emit-weighted** and **0.708 face-uniform**.
- Emitted lookup tables cover the broader 770-face union. The current
  pooled table (`{gemma, ministral, opus}`) scores **0.847
  emit-weighted** and **0.669 face-uniform** over 243 GT-scored faces;
  the strict Claude table (`{gemma, opus}`) scores **0.786
  emit-weighted** and **0.717 face-uniform** over 70 GT-scored faces.
- Opus introspection is the top solo encoder in both current searches.
  The gain over Haiku is largest on low-arousal and neutral cells.
- Local hidden states recover a shared affect geometry across model
  families. The exact PCA axes differ, but quadrant centroids preserve
  the same Russell/PAD structure.
- BoL is useful as an interpretive layer and diagnostic encoder, but the
  Haiku synthesis route appears positivity-biased on negative-affect
  contexts. Prefer Claude-GT or Opus introspection over BoL when they
  disagree on deployment meaning.

Full numbers live in [`docs/findings.md`](docs/findings.md).
The compact machine-readable headline record is
[`data/summary/results.json`](data/summary/results.json); public figures are
mapped in [`figures/README.md`](figures/README.md).

## Reproducing

```bash
python --version  # shared venv Python 3.12
uv pip install -e ..
uv pip install -e .
```

For local hidden-state sanity:

```bash
python scripts/local/90_hidden_state_smoke.py
```

For the full local analysis chain after an emit refresh:

```bash
scripts/run_local_chain.sh
```

For harness-side corpus and Anthropic-judge regeneration:

```bash
ANTHROPIC_API_KEY=... scripts/run_harness_chain.sh
```

For everything in dependency order:

```bash
ANTHROPIC_API_KEY=... scripts/run_all.sh
```

The chain scripts are the current orchestration surface. Individual
script comments are still useful, but old dated design docs should be
treated as historical unless they are listed in the docs map below.

## Docs Map

- [`AGENTS.md`](AGENTS.md): current operating notes for agents and the
  shortest command map.
- [`docs/findings.md`](docs/findings.md): current result summary and
  numbers worth citing.
- [`docs/local-side.md`](docs/local-side.md): local hidden-state and
  face-likelihood methodology.
- [`docs/harness-side.md`](docs/harness-side.md): contributor corpus,
  BoL, Claude-GT, use/read/act, and privacy.
- [`docs/gotchas.md`](docs/gotchas.md): active debugging sharp edges.
- [`docs/internals.md`](docs/internals.md): hidden-state sidecars and
  canonicalization rules.
- [`docs/previous-experiments.md`](docs/previous-experiments.md):
  compact ledger of retired framings and deleted docs.

Still-current dated docs:

- [`docs/2026-05-04-claude-groundtruth-pilot.md`](docs/2026-05-04-claude-groundtruth-pilot.md):
  Claude-GT collection protocol, updated for the merged-file layout.
- [`docs/2026-05-05-soft-everywhere-methodology.md`](docs/2026-05-05-soft-everywhere-methodology.md):
  JSD similarity pivot.
- [`docs/2026-05-06-use-read-act-channels.md`](docs/2026-05-06-use-read-act-channels.md):
  GT use vs introspective read vs BoL act.
- [`docs/2026-05-06-nn-lb-future-cells.md`](docs/2026-05-06-nn-lb-future-cells.md):
  deferred NN/LB cell plan.
- [`docs/2026-05-07-claude-gt-v4-extension-pilot.md`](docs/2026-05-07-claude-gt-v4-extension-pilot.md):
  HP-D/NP/HB Claude-GT extension outcome.
- [`docs/2026-05-08-saturation-threshold-recal.md`](docs/2026-05-08-saturation-threshold-recal.md):
  `PER_Q_JS_MAX` recalibration and LN amendment.
- [`docs/2026-05-05-residual-state-axes.md`](docs/2026-05-05-residual-state-axes.md):
  generated wild-residual cluster report.

## Related

- [`llmoji`](https://github.com/a9lim/llmoji): contributor package for
  hooks, canonicalization, synthesis, and upload.
- [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji): shared
  contributor corpus.
- [`saklas`](https://github.com/a9lim/saklas): activation steering and
  trait monitoring engine.
- `attractor-experiment`: the MR / meta-register basin research line, split
  off from this repo.
- [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces):
  original kaomoji vocabulary catalog and prior art.

## License

CC-BY-SA-4.0 for this repo (writeups, figures, analysis code). See
[LICENSE](LICENSE). The companion package
[`llmoji`](https://github.com/a9lim/llmoji) is GPL-3.0-or-later. The
shared corpus on HuggingFace is CC-BY-SA-4.0.
