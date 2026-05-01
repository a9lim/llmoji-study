# Gotchas

Known sharp edges encountered while building the v1/v2/v3 pipelines.
Read this before debugging anything that's silently wrong.

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

### Kaomoji vocabulary differs sharply across model lineages — RESOLVED via TAXONOMY drop

Pre-2026-04-30: gemma-tuned `TAXONOMY` happy/sad labels (in
`llmoji_study.taxonomy_labels`) didn't cover qwen / claude / ministral
faces. Vocab-discovery scripts 00 / 19 / 20 were used to identify
which faces to add per model. The whole machinery is now obsolete:
v3 analyses key on `first_word` (canonicalized via
`llmoji.taxonomy.canonicalize_kaomoji`), and v1/v2 pole assignment
moved to per-face mean `t0_<axis>` probe-score sign in
`analysis._add_axis_label_column`. No model-specific dictionaries
needed; everything generalizes by construction.

### Re-extracting pilot data after canonicalization rule changes

`first_word` is baked at write time. `04_emotional_analysis.py` calls
`_relabel_in_place` at start of every run, which re-extracts via
`llmoji.taxonomy.extract` and drops legacy `kaomoji` /
`kaomoji_label` fields if present. For other JSONLs do it manually:
```python
import json
from pathlib import Path
from llmoji.taxonomy import extract
p = Path("data/pilot_raw.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l]
for r in rows:
    m = extract(r["text"])
    r["first_word"] = m.first_word
    r.pop("kaomoji", None)
    r.pop("kaomoji_label", None)
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

### v3 runner's per-quadrant emission rate now reads first_word — RESOLVED in TAXONOMY drop

Pre-2026-04-30 this checkpoint counted `kaomoji_label != 0` (TAXONOMY
match) and read 10–30% on non-gemma models even though real
instruction-following compliance was ~100%. Misled at least one
mid-run abort. After the TAXONOMY drop the numerator is just
`first_word` truthiness — bracket-leading kaomoji presence — which
matches the v3 loader's actual filter. Now reports real compliance on
every model.

### Mistral tokenizer ships a buggy pre-tokenizer regex — fixed in saklas 2.0.0

HF-distributed Mistral checkpoints (Mistral-Small-*, Ministral-*,
third-party finetunes carrying the family name) ship a buggy
pre-tokenizer regex that mis-splits ~1% of tokens — e.g. `"'The'"`
tokenizes as `["'", "T", "he", "'"]` instead of `["'", "The", "'"]`.
Bug is in encoding (text→tokens), not generation; affects words
preceded by apostrophes / punctuation, so v3 prompts with `I'm`,
`don't`, `it's` get slightly OOD tokenization. Saklas 2.0.0 fixes
this by passing `fix_mistral_regex=True` to
`AutoTokenizer.from_pretrained` whenever `model_id` substring-matches
`"mistral"` (case-insensitive). See
[discussion 84](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84).
Pre-fix data: noisy but not broken — geometry findings (silhouette,
CKA, probe scores) are robust. Post-fix should match or strengthen
the signal. Cross-version compatibility verified: 2.0.0 reproduces
1.4.6 probe scores within 5e-7 on existing sidecars.

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

