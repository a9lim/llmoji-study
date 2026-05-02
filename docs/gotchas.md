# Gotchas

Known sharp edges encountered while building the v1/v2/v3 pipelines.
Read this before debugging anything that's silently wrong.

## Gotchas

### saklas `cache_prefix` produces contaminated KV state on Qwen3.6

`session.cache_prefix(full_input[:-1])` followed by `session.generate(...)`
on the same prompt produces **identical off-prompt text** for every seed
on Qwen3.6 — markdown headers, code documentation, math answers,
unrelated content. The byte-equal cache-hit check passes, the suffix
generation runs, but the cached past_key_values encode a corrupted
attention state on qwen specifically. Gemma + Mistral are unaffected by
the same code path.

Discovered 2026-05-03 during the cleanliness-pass full N=8 rerun:
qwen's seed=0 (no cache, from the 1-seed pilot) was correct, but every
subsequent seed at every prompt produced
`"# 1. Introduction\n\n## 1.1. Purpose\n\nThis document"` regardless of
the actual prompt. 840 generations were lost.

**Workaround**: `llmoji_study/capture.py::install_full_input_cache`
no-ops when `session.config.model_id` contains `"qwen"`. Qwen runs
~30-50% slower without the optimization but produces correct output.
Root cause is on the saklas side (qwen-tokenizer / cache-prefix
interaction); proper fix is a follow-on task in saklas.

The other prefix-cache helper, `install_prefix_cache` (cross-prompt
common prefix used by N=1 pilots), works correctly on qwen — only the
per-prompt full-input variant trips the bug.

### Mixing cache modes across pilot+resume contaminates seed-0 hidden states

If a pilot run uses one cache mode (e.g. `install_prefix_cache` for
N=1) and the resumed full run uses another (e.g. `install_full_input_cache`
for N>1), the pilot's seed-0 sidecars persist with KV state from the
first mode while seeds 1..N get recomputed under the second. Even
when the suffix decoded by either mode is byte-equal, the saved
hidden states diverge — `cache_prefix` is not transparent at the
per-token-hidden-state level (this is true on all 3 models, just
worst on qwen via the bug above). Per-row L2 deviation at h_first
@ preferred_layer measured 2026-05-03: gemma ~1%, qwen 37–46%,
ministral ~0.8%.

Discovered when seed-0 PCA scatter rendered visibly off-cluster from
seeds 1–7 in the cleanliness rerun. Symptom: per-prompt grouping in
PCA space shows seed 0 in a different position than seeds 1..N for
the same prompt, while seeds 1..N are bit-identical.

**Fix**: when resuming a pilot into a full run, delete the pilot's
seed-0 rows + sidecars and let the resume mechanism regenerate
seed 0 under the same cache mode as 1..N. Verification: hidden
states should be bit-identical (|s0 − mean(s1..N)| ≈ 0 at full
fp32 precision). See `data/*_emotional_raw.jsonl.bak.before_seed0_rerun`
for the 2026-05-03 incident backups.

### Probe scores are saklas-neutral-centered, not project-NB-centered

`probe_scores_t0` / `probe_scores_tlast` / `probe_means` are
**mean-centered cosines** — saklas's `TraitMonitor`
(`saklas/core/monitor.py:147`) subtracts a baked per-layer mean
before the cosine. The mean is `compute_layer_means` over saklas's
bundled `neutral_statements.json` (~90 generic neutrals), persisted
at `~/.saklas/models/<safe_id>/layer_means.safetensors`.

Implication: the centering is global / saklas-bundled, NOT against
this experiment's NB-quadrant prompts. The NB bar in any
per-quadrant probe-mean figure is non-zero by default — it's the
gap between saklas's neutrals and our NB framings. For
project-relative reads (e.g. "affect lift over a domain-matched
neutral observation"), subtract the per-probe mean over project NB
rows before plotting; `_plot_quadrant_means` in
`scripts/local/28_v3_extension_probe_figures.py` does this. Rule-3b
diffs are unaffected (the centering shift cancels in HN-S − HN-D).
Pearson/Spearman correlation matrices are also unaffected (additive
shift cancels in covariance).

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
`ModelPaths` dataclass in `llmoji_study.config`. Under h_first
(canonical since 2026-05-02): gemma L50, qwen L59, ministral L20.
v3 scripts pass `layer=M.preferred_layer` to
`load_emotional_features` so figures get the right snapshot per model.

If you call `load_hidden_features` / `load_emotional_features` directly
in a notebook or new script, you have to remember the override —
``layer=None`` always means "deepest", regardless of model. Easiest
convention: ``layer=current_model().preferred_layer``.

The cache files at `data/cache/v3_<short>_h_mean_all_layers.npz`
(legacy filename; contents reflect the active `which`) contain ALL
layers, so script 21 (which iterates over layers) doesn't depend on
`preferred_layer`. Same for the per-layer CKA grid in script 23 and
the triplet Procrustes in script 31.

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

### t0/h_first probe scores are prompt-deterministic

At h_first (the state right before the first generated token),
scalar probe scores are determined by the prompt + model, not the
sampling seed. Per v3 main run, all 8 seeds × N prompts × any
probe collapse to **N unique 3-tuples at 4-decimal precision** —
matches the unique-prompt count exactly (N=123 on the prior prompt
set; N=120 on the post-2026-05-03 cleanliness-pass set, when
re-measured). Sampling stochasticity affects which token gets drawn
FROM the t0 distribution, not the t0 state itself.

Implication for visualizations: 3D probe scatters at t0/h_first
look sparse because 8 seeds-per-prompt overplot. Visual richness
from seed-variance lives at h_last/h_mean (response-evolved state).
3D PCA of the same data also collapses to N unique points but
the 5376-dim → PCA(3) spread makes overplotting less visually
obvious. Use h_last for probe scatters specifically when
seed-variance matters; h_first for everything else (cleaner
quadrant geometry, methodology-invariant across the
MAX_NEW_TOKENS cutover).

### `MAX_NEW_TOKENS` changed mid-project (120 → 16, 2026-05-02)

Pre-2026-05-02 data was captured with `MAX_NEW_TOKENS=120` (full
response). Post-2026-05-02 data uses 16-token early-stop — kaomoji
emit at tokens 1–3, 16 is generous headroom, ~7–8× compute cut.
**`t0` is unchanged** across the cutover. **`tlast` and `h_mean`
aggregates reference different windows on each side**: pre-cutover
they cover the full ~120-token response; post-cutover they cover a
~12–13-token window after the kaomoji. Don't pool tlast/mean across
the cutover line without thinking about it. Pre-cutover data that
matters: v1/v2 (~900 rows), v3 main on gemma (800), qwen (800),
ministral (800) — all under the long-form aggregate. Post-cutover
data: introspection pilot (gemma, 369 rows) and anything new.

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
per-glyph fallback via `rcParams["font.family"] = [...]`. The canonical
`_use_cjk_font` helper lives in `llmoji_study.emotional_analysis`;
`analysis.py` imports it (single source of truth post-2026-05-04
dedupe). Scripts that work outside the
emotional-analysis pipeline (`scripts/harness/16_eriskii_replication.py`,
`scripts/harness/18_claude_faces_pca.py`) keep local copies for now —
**keep in sync**. The helper registers a project-local monochrome emoji
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
`rm -rf data/hf_dataset && python scripts/harness/06_claude_hf_pull.py`. Cache is
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

