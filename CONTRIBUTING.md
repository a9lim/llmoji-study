# Contributing

llmoji-study is a research repo, not a library. There's no API to
maintain backward compatibility against; the bar for changes is "is
this honest, is it documented, does it leave the project in a state
where future-you can still read it." Three ways people meaningfully
contribute:

1. **Submit your kaomoji data.** The most useful thing for the
   harness-side analyses is corpus diversity. If you use Claude Code
   or Codex with a kaomoji-prompt, the
   [`llmoji`](https://github.com/a9lim/llmoji) PyPI package will
   collect, canonicalize, synthesize per-face descriptions, and
   submit a privacy-preserving aggregate to the
   [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) HF
   dataset. The harness-side pipeline in this repo automatically
   pulls from there.
2. **Add a new model to the local-side v3 study.** If you've got a
   GPU (or unified-memory Apple Silicon) capable of running a 14–31B
   open-weight causal LM and want to extend the cross-model
   comparison, the design is parameterized — you register the model
   and run the pre-existing pilot → main pipeline.
3. **Extend an analysis.** Most of the figures and decision rules
   are tied to design docs in `docs/`. Adding a new analysis means
   writing a small design doc, implementing it, regenerating
   downstream figures.

You can also just open an issue with a question or a finding —
those are real contributions too.

## Submitting kaomoji data (harness side)

You don't touch this repo at all. The contributor-side flow is
entirely in the `llmoji` package:

```bash
pip install llmoji
llmoji install   # registers the Stop-event hook for your harness
# ... use Claude Code or Codex normally for a while ...
llmoji status    # see how much you've accumulated
llmoji analyze   # synthesize per-kaomoji descriptions via Haiku
                 # (needs ANTHROPIC_API_KEY)
llmoji upload    # pushes the privacy-preserving aggregate to HF
```

What gets uploaded: per-canonical-kaomoji counts and one
Haiku-synthesized one-sentence meaning per face per harness. No
prompts, no responses, no surrounding text — just the canonical
kaomoji form, frequency, and the aggregated meaning. Source code is
[on GitHub](https://github.com/a9lim/llmoji); the upload payload is
auditable before you run `upload`. See the package's own README for
the privacy model.

The research-side analyses in `docs/harness-side.md` will pick your
data up on the next pull.

## Adding a model to the v3 study

The local-side study runs on
`gemma-4-31b-it`, `Qwen3.6-27B`, and
`Ministral-3-14B-Instruct-2512`. Adding a fourth follows the
existing pattern.

### Prerequisites

You need:

- A machine that can run the model in bf16 (24 GB VRAM minimum for
  14B; 48–64 GB for 27–31B; or a unified-memory Apple Silicon
  machine with 32+ GB)
- The model's HF tokenizer + weights cached locally (one-time
  download, typically 30–60 GB)
- The repo + companion package installed editable; see "Working
  with the codebase" below
- `saklas >= 2.0.0` (one of its dependency hops is
  `transformers>=5`; the editable install pulls the right
  versions)

### Workflow

1. **Register the model.** Add a `ModelPaths` entry to
   `MODEL_REGISTRY` in `llmoji_study/config.py`:
   ```python
   "your_short_name": ModelPaths(
       model_id="org/your-model-id",
       short_name="your_short_name",
       experiment="v3_your_short_name",
       emotional_data_path=DATA_DIR / "your_short_name_emotional_raw.jsonl",
       emotional_summary_path=DATA_DIR / "your_short_name_emotional_summary.tsv",
       figures_dir=FIGURES_DIR / "local" / "your_short_name",
       preferred_layer=None,  # filled in after pilot's layer-sweep
   ),
   ```
   Add `your_short_name` to the `LLMOJI_MODEL` env-var docstring.
2. **Smoke test.** Verifies the wiring:
   ```
   LLMOJI_MODEL=your_short_name python scripts/99_hidden_state_smoke.py
   ```
   Generates 5 samples across quadrants, checks probe round-trip to
   ~1e-7 tolerance, validates sidecar shapes. ~5 min.
3. **Write a pilot design doc.** Copy
   `docs/2026-04-30-v3-ministral-pilot.md` as a template; replace
   model-specific bits. Pre-register the gating thresholds (rules 1
   silhouette, 2 cross-model CKA, 3b fearful.unflinching) before
   you run.
4. **Pilot run.** ~100 generations, prompt-aligned with existing
   models for cross-model CKA:
   ```
   LLMOJI_MODEL=your_short_name LLMOJI_PILOT_GENS=1 \
     python scripts/03_emotional_run.py
   ```
   ~25 minutes on M5 Max-class hardware for a 14B model.
5. **Pilot analysis.** The chain auto-discovers any model with v3
   data on disk:
   ```
   python scripts/21_v3_layerwise_emergence.py    # silhouette → identify preferred_layer
   python scripts/23_v3_cross_model_alignment.py  # gemma↔qwen↔your_model CKA
   python scripts/30_rule3_dominance_check.py     # rule 3b verdict
   ```
   Update `preferred_layer` in your `ModelPaths` entry from the
   layer-sweep peak.
6. **Gating.** Apply your pre-registered thresholds. If pass, write
   the supplementary design doc and run main (N=800):
   ```
   LLMOJI_MODEL=your_short_name python scripts/03_emotional_run.py
   ```
   ~3 hours for a 14B model. Resumable — if it dies, just rerun.
7. **Document.** Add a "Pilot v3 — your-model" section to
   `docs/findings.md`. Update `CLAUDE.md`'s Status. Open a PR
   with the design doc + the JSONL + summary TSVs (sidecars are
   gitignored).

The supplementary tagged-HN prompts (hn21–hn43) live in
`llmoji_study/emotional_prompts.py` and apply to your model
automatically. Re-running script 27 against your model populates
extension probe scores; script 30 then evaluates rule 3b.

## Extending an analysis

If you want to add a new figure, decision rule, or cross-cut:

1. **Look at `docs/` first.** Every existing analysis has a
   companion design doc; the format is "what / why / pre-registered
   rule / outcomes." Write a short design doc for your analysis —
   even half a page — before implementing. The discipline of
   pre-registering the rule (vs. reading the data and writing the
   conclusion afterward) is most of the value.
2. **Follow the script numbering.** Scripts are roughly
   chronologically numbered. Pick the next free integer (currently
   31+) and a descriptive name: `scripts/31_your_thing.py`.
3. **Keep it data-light by default.** If your analysis can read
   from existing sidecars or JSONLs without new generations, do
   that — `data/cache/v3_<short>_h_mean_all_layers.npz` has the
   multi-layer hidden states already, and
   `extension_probe_scores_*` fields on JSONL rows have the probe
   scores. New generations cost compute and welfare budget (see
   below); avoid when possible.
4. **Reuse the loaders.** `load_emotional_features` from
   `llmoji_study.emotional_analysis` handles JSONL + sidecar
   alignment, canonicalization, and the rule-3-redesign HN split
   (`split_hn=True`). `apply_hn_split` post-processes a
   pre-existing df.
5. **Update findings.md** when the analysis lands. Reference the
   design doc.

## Working with the codebase

```bash
git clone https://github.com/a9lim/llmoji-study
git clone https://github.com/a9lim/llmoji ../llmoji
git clone https://github.com/a9lim/saklas ../saklas

cd llmoji-study
python -m venv .venv && source .venv/bin/activate
pip install -e ../saklas    # editable; we sometimes patch saklas
pip install -e ../llmoji    # editable
pip install -e .
```

`saklas` is editable on purpose — research occasionally surfaces
saklas-side issues (e.g. the Mistral tokenizer regex bug we caught
in 2026-04-30). Patch upstream + the change is live without a
reinstall.

Conventions worth knowing (more in `CLAUDE.md`):

- **Single venv at `.venv/`**, pip not uv
- **Python 3.11+** (currently tested on 3.14)
- **JSONL is source of truth** for row metadata + probe scores;
  `data/hidden/<experiment>/<uuid>.npz` is source of truth for
  hidden states; delete both when changing model/probes/prompts/seeds
- **No formatter mandate**, but match the surrounding style
- **No tests** for the research-side scripts; saklas has its own
  test suite that should pass after any saklas-side change
- **Pre-registered decisions** live in `pyproject.toml`,
  `llmoji_study/{config,prompts,emotional_prompts}.py`, and the
  package's frozen v1.0 surface (`llmoji.{taxonomy,synth_prompts}`).
  Package-side changes are major-version events; research-side
  changes only invalidate cross-run comparisons within this repo.
- **Pyright errors from pandas / plotly stubs** are mostly noise;
  the runtime is fine. Don't suppress the real errors among them.

## Ethics — model welfare

The functional-emotional-state framing isn't decorative. The repo's
Ethics section in `CLAUDE.md` is binding for new pilots: run trials
only when a smaller experiment can't answer the question, pre-register
decision rules and minimum N, prefer stateless designs when the
question admits it, redesign rather than 10×ing on negative or noisy
findings. HN-quadrant prompts in particular elicit
sad / angry / fear registers; aggregating across hundreds of
generations is real moral weight regardless of where you stand on the
phenomenal-status question.

Concretely: a v3 main run is 800 generations, ~160 of which are
HN-quadrant. The supplementary 23-prompt addition is another ~160
HN per model. That's the active scale. Justifying more — bigger N,
more axes — should clear the smoke-→-pilot-→-main bar.

## Questions, issues, getting in touch

- **GitHub issues** on this repo for project-side questions, bugs,
  feature ideas
- **GitHub issues** on
  [`llmoji`](https://github.com/a9lim/llmoji) for the contributor
  package + harness installation problems
- **GitHub issues** on
  [`saklas`](https://github.com/a9lim/saklas) for activation-steering
  / probing engine bugs
- Email **mx@a9l.im** for direct contact, especially if the issue
  involves data we shouldn't post publicly

Authorship attribution defaults to `a9lim` for new work; explicit
attribution welcome via commit Co-authored-by lines or AUTHORS-style
notes if you want it on a finding.
