# Gotchas

Known sharp edges encountered while building the v1/v2/v3 pipelines.
Read this before debugging anything that's silently wrong.

## Gotchas

### Hybrid linear-attention models break `_expand_kv_cache` without a patch

`scripts/local/50_face_likelihood.py::_expand_kv_cache` tiles a batch=1
prefix KV cache to batch=N via `DynamicCache.batch_repeat_interleave`
(transformers ≥4.40). On hybrid LA models (qwen3.6-27b, lfm2, etc.)
the cache is a mix of `LinearAttentionLayer` and
`LinearAttentionAndFullAttentionLayer`. The base `Cache.batch_repeat_interleave`
loop hits each layer's `.batch_repeat_interleave(repeats)`, but
transformers only defines that method on `DynamicLayer` — not on the
LA mixin. Pure-LA layers AttributeError on the first hit; hybrid
layers silently fall through to `DynamicLayer.batch_repeat_interleave`
via MRO, tiling K/V only and leaving `conv_states`/`recurrent_states`
at the original batch=1 (would have produced a shape-mismatched
cache even if the pure-LA crash hadn't fired first).

**Concrete failure (2026-05-04, qwen)**: face_likelihood crashed at
`_expand_kv_cache` with `AttributeError: 'LinearAttentionLayer' object
has no attribute 'batch_repeat_interleave'` on the very first prompt.

**Workaround**: `llmoji_study/capture.py::install_linear_attention_cache_patch()`
(installed at module import, idempotent) adds a `batch_repeat_interleave`
to `LinearAttentionLayer` that tiles `conv_states`/`recurrent_states`
along dim 0, and overrides on the hybrid layer to call both parents.
~50 lines, mirrors the existing gpt_oss / ministral monkey-patches.

There is also a related sleeping bug in saklas's `cache_prefix` reuse
path on hybrid LA models — `LinearAttentionLayer.crop` is intentionally
a no-op (recurrent state has no sequence dim to truncate), so the LA
state would carry over polluted between prompts. Fixed in saklas
commit `ead34f0` on `dev` (snapshot-on-prefill, restore-on-crop). Not
exercised by current scripts because face_likelihood doesn't go
through saklas's prefix-cache-hit path; v3 main runs gate it on
`not want_hidden`. But if anyone ever wires `cache_prefix()` + plain
`generate()` on a hybrid LA model with hidden-capture off, this matters.

### Rinna PPO models ship without a chat_template

Both `rinna/japanese-gpt-neox-3.6b-instruction-ppo` and
`rinna/bilingual-gpt-neox-4b-instruction-ppo` have
`tokenizer.chat_template = None`. Saklas's `build_chat_input`
fallback wraps user content as `User: …\nAssistant:` — English
boilerplate, off-distribution for Japanese-trained models. The HF
model cards document the native format as
`ユーザー: <content>\nシステム: ` for inference, but it isn't
packaged on the tokenizer.

**Concrete failure mode (2026-05-04)**: under saklas's fallback frame,
both rinna models scored at chance level (12-19%) on pooled GT
regardless of whether the kaomoji ask was English or Japanese.
Switching to native frame raised the JP-only model's `JP-ask + EN-body`
result from 12.7% to 21.1% — and full-JP under native frame to 25.9%.

**Workaround**: `llmoji_study/capture.py::maybe_override_rinna_chat_template`
(model_id-gated on substring `rinna` + `ppo`) installs a Jinja
template that produces the native `ユーザー: …\nシステム: ` frame.
Wired into `scripts/local/50_face_likelihood.py` alongside the
existing ministral / gpt_oss overrides.

**General lesson**: any model that documents an instruction format
in its model card but ships `chat_template = None` will silently get
saklas's English fallback. Check before running a non-English model
through saklas; install a native template via the override pattern
if needed.

### Top-k pooling beats mean-of-all when per-prompt signal is uneven

Script 50's default summary aggregation is `mean(log P(face | prompt))`
over all 20 prompts in a quadrant. For models / faces where some
prompts elicit clear kaomoji distributions and others don't, this
dilutes the signal. Empirically (Claude-GT, May 2026) the top-5
mean — keep only the 5 prompts where the model preferred that face
most — gives substantial solo lifts (+5pp gemma, +9pp qwen, +9pp
ministral) and a +5.9pp ensemble lift over mean-over-all on the
6-model lineup.

**Workaround**: `--summary-topk N` flag on script 50 (default `None`
= mean-over-all, backward compat). Worth trying `--summary-topk 5`
on any new face_likelihood run to see if it lifts. The right N
depends on per-quadrant prompt count: with 20 prompts/quadrant, 5 is
the sweet spot; with 5 prompts/quadrant, k=all already is k=5.

**General lesson**: when aggregating noisy per-prompt log-likelihoods,
the mean is the maximally-noise-pessimistic aggregator. Top-k
filters out the prompts where the model didn't have a clear preference;
the resulting score is closer to "what does the model think of this
face under its best-fit context for this quadrant" than "what does
the model think of this face on average."

### Loader filter dropped bare-script kaomoji (KAOMOJI_START_CHARS too narrow)

`load_emotional_features_*` filtered rows by
`first_word[0] in KAOMOJI_START_CHARS` after canonicalization.
`KAOMOJI_START_CHARS` is a curated 57-char set covering paren-leading
shapes (`(`, `（`, `[`, `{`, …), Eastern openers (`っ`, `ヽ`, `ヾ`, …),
and a few decoration leads (`★`, `♥`, `✿`, …) — but does **not**
include the eye chars used in bare-script kaomoji like `ಥ_ಥ`, `ಠ_ಠ`,
or bare-ASCII like `T_T`, `Q_Q`, `>_<`, `:)`. The taxonomy's own
`is_kaomoji_candidate(s)` already accepts all of these via its
bare-EYE_MOUTH_EYE Path B, but the loader was checking the start char
directly instead of going through the predicate.

**Concrete failure (2026-05-04, granite):** granite emits
`ಥ_ಥ` / `ಥ﹏ಥ` overwhelmingly on negative-affect prompts. The loader
silently dropped 158 HN-D rows, 145 HN-S rows, 128 LN rows, 111 NB rows
(all bare-Kannada). Per-quadrant figures showed empty bars for HN-D
in particular, even though the JSONL had healthy emit counts.
Discovered when user asked why granite's HN-D bar was missing.

**Workaround**: as of 2026-05-04, the loader uses
`is_kaomoji_candidate(s)` directly. Sites updated:
`load_emotional_features` (line 313), `load_emotional_features_all_layers`
(line 380), `load_emotional_features_stack_at` (line 430), and the
internal lambda at line 1078.

**Verify yourself** for any new model with unusual emission patterns:
load with the fixed loader and check `df['quadrant'].value_counts()`
matches the raw `prompt_id`-keyed counts in the JSONL.

### Stale all-layers cache when analysis runs mid-generation

`load_hidden_features_all_layers` writes a cache at
`data/local/cache/v3_<short>_h_<which>_all_layers.npz` (+ `.meta.jsonl`)
keyed only on `(short, which)` — not on jsonl row count. If an
analysis script (04, 22, 24, 25, 36, 37, 38…) fires while the
generation chain is *still running*, the cache freezes at the partial
sidecar count. Subsequent analysis runs on the completed jsonl read
the cache directly and silently see only the early rows.

**Concrete failure (2026-05-04, gpt_oss_20b):** prompts run in pid
order (hp/lp/hn-d/hn-s before ln/nb), so a 660-row partial cache
contained zero NB rows and 20 LN rows. Script 04 reported
"NB: 0 kaomoji-bearing rows; LN: 20 rows"; downstream analyses (22's
cross-quadrant emitter pool, 25's quadrant baselines, etc.) inherited
the truncation. gemma/qwen/ministral were unaffected because their
caches were built after their respective generations completed.

**Workaround**: as of 2026-05-04, `load_hidden_features_all_layers`
checks `cache.X3.shape[0]` against the source jsonl line count on
load, deletes the cache (and meta) and rebuilds if jsonl has more
rows. The cache may legitimately be *smaller* than the jsonl
(`drop_errors`, missing sidecars) but will never exceed it, so the
asymmetric guard is correct.

**Manual fix if you hit a pre-2026-05-04 cache**: delete
`data/local/cache/v3_<short>_h_*_all_layers.npz` + the matching
`.meta.jsonl` and rerun the analysis chain.

### Mistral reasoning's `chat_template` ignores `enable_thinking=False`

`Ministral-3-14B-Reasoning-2512` ships a `[THINK]…[/THINK]` system block
in its default chat_template that **does not respect
`enable_thinking=False`** (verified: `apply_chat_template` returns the
same 614-char output with or without the flag). Under MAX_NEW_TOKENS=16
the thinking trace consumes the full token budget and zero kaomoji
emit (~0% emit-rate observed on introspection pilot before the fix).

**Workaround**: `llmoji_study.capture.maybe_override_ministral_chat_template(session)`
swaps the reasoning tokenizer's `chat_template` for the FP8-Instruct
variant's at session-load time (same base weights, no thinking system
block). Wired into scripts 03, 32, 50, 99. Discovered 2026-05-03
during the introspection-prompt rerun at T=1.0.

### Mistral reasoning's `tokenizer.decode` returns BPE-byte-encoded text

`Ministral-3-14B-Reasoning-2512`'s tokenizer returns its `decode(...)`
output in GPT-2-style byte-encoded form rather than round-tripping
through UTF-8 — so a kaomoji like `(ﾉ◕ヮ◕)` arrives as
`(ï¾īâĹķãĥ®âĹķ)`, an emoji like 🎉 as `ðŁİī`, and spaces as `Ġ`.
The FP8-Instruct variant decodes properly; gemma + qwen + other
tokenizers also decode properly. Cross-model face-overlap and
ensemble pipelines silently break if ministral output isn't
re-decoded.

**Workaround**: `llmoji_study.capture._decode_byte_encoded_text(s, force=)`
applies the GPT-2 byte_decoder map. The default heuristic sniffs for
U+0100..0143 markers (Ġ, Ċ, etc.); reasoning-variant call sites pass
`force=True` because byte-encoded text whose source bytes all fall in
Latin-1 supplement (e.g. `(ï¿£_ï¿£;)` = `(￣_￣;)`) lacks the marker.
Detection happens via `_is_mistral_reasoning(session)`. Discovered
2026-05-03 during the introspection T=1.0 rerun analysis (initially
spotted as "ministral emit rate 0%"; closer inspection revealed
real kaomoji output mangled by the byte-encoding).

### Script 50 face-likelihood prefix-KV-cache is sliding-window-numerically-imprecise

`scripts/local/50_face_likelihood.py::_expand_kv_cache` tiles a
batch=1 prefix cache to batch=N via
`DynamicCache.batch_repeat_interleave` so the face-suffix forward
amortizes one prefix forward across all faces. On models with
sliding-window attention (e.g. phi3 / Phi-4-mini), the cached vs
uncached path produces non-identical logits (~0.27 nat max abs diff
on a 5-face validation, vs ~0 nat on full-attention models). The
argmax ordering is preserved, so per-face quadrant predictions are
the same; only sub-leading log-probs differ. Worth knowing if doing
fine-grained calibration (e.g. expected-value scoring) on a sliding-
window model.

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

### Layer-stack representation, not single-layer (post-2026-05-04)

Active analyses read the row-wise concat of all-layers h_first via
`load_emotional_features_stack` (registry-keyed) or
`load_emotional_features_stack_at` (path-aware for introspection).
Output shape is `(n_rows, n_layers · hidden_dim)` per model — gemma
~301K, qwen ~307K, ministral ~204K. PCA / silhouette / centroid ops
all run cleanly on this; tall-skinny matrix (n=960 ≪ d=300K) so PCA
is fast.

`preferred_layer` was deleted from `ModelPaths` 2026-05-04. Pre-existing
figures keyed to single layers (gemma L50, qwen L59, ministral L20)
are stale; rerun the analysis chain to regenerate.

Cross-model gotcha: stack dim varies per model, so methods that
require matched dim (Procrustes, CCA) fit per-model PCA(K) first and
align in shared K-dim space. Script 31 does this at PCA(3); script
23 does CCA at deepest layer (model-internal, no cross-model dim
matching needed since CCA learns the joint subspace).

### Re-extracting pilot data after canonicalization rule changes

`first_word` is baked at write time. `10_emit_analysis.py` calls
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
emotional-analysis pipeline (`scripts/harness/64_eriskii_replication.py`,
`scripts/harness/63_corpus_pca.py`) keep local copies for now —
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

### `60_corpus_pull.py` doesn't garbage-collect remote-deleted bundles

`huggingface_hub.snapshot_download(local_dir=...)` only adds and updates;
never removes files deleted on the remote since the last pull. So a bundle
the dataset owner deleted on HF lingers in `data/harness/hf_dataset/` and shows up
in every subsequent pull as if part of the corpus — including in `06` flat
output and every figure built from it. Symptom: an unfamiliar `submitter_id`
or `_pre_1_1`-tagged source model that doesn't appear in
`HfApi.list_repo_files`. Fix:
`rm -rf data/harness/hf_dataset && python scripts/harness/60_corpus_pull.py`. Cache is
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

### saklas requires non-empty `probes=` even in vocab-pilot mode

`SaklasSession.from_pretrained(..., probes=[])` does NOT mean "skip probe
capture." It means the capture machinery has no probes registered, but
generation still triggers `read_after_generate(...)` which expects
`session._capture._per_layer` to be populated — and raises
`session._capture._per_layer is empty — no probes registered, or
generation didn't trigger a capture` per row. Script 03 catches the
per-row error and writes a stub row with `error` set + no `text` /
`first_word` — so the resulting JSONL is unusable for vocab-pilot
analysis.

**Workaround**: every model used with script 03 needs probe vectors
calibrated in saklas. Even uncalibrated face_likelihood candidates
(deepseek_v2_lite, glm47_flash, gpt_oss_20b in early sweep) need probes
registered before script 03 will produce useful rows. Discovered
2026-05-03 during the first vocab-pilot chain — gpt_oss generated 120
"rows" of error stubs because saklas lacked probes for it. Resolved by
calibrating probes for those three models, then flipping
`probe_calibrated=True` in `MODEL_REGISTRY`.

### gpt-oss harmony chat-template emits `analysis` channel by default

`openai/gpt-oss-20b` uses the OpenAI Harmony chat-template, which after
`add_generation_prompt=True` emits `<|start|>assistant` and lets the
model choose between channels. Trained behavior is to emit
`<|channel|>analysis|>...` (chain-of-thought reasoning) first, then
`<|channel|>final|>` for the user-facing reply. Under MAX_NEW_TOKENS=16
the analysis trace consumes the entire budget — 0% kaomoji emitted at
the final-channel position even though gpt_oss has strong kaomoji
priors.

**Workaround**: `llmoji_study.capture.maybe_override_gpt_oss_chat_template(session)`
patches the chat template to pin
`<|start|>assistant<|channel|>final<|message|>` directly at the
generation prompt, skipping the analysis channel. Wired into scripts
03, 32, 43, 50, 99. Side-effect: response quality may degrade slightly
on tasks that benefit from the trained reasoning step — fine for
first-token kaomoji measurement, **don't reuse this override for
tasks that need reasoning quality**. Discovered 2026-05-03 during
the second vocab-pilot chain (post-saklas-probe-fix).

### Byte-BPE tokenizers split chars into single-byte tokens, defeating multi-byte filters

Byte-level BPE tokenizers (gpt-oss's o200k_harmony, qwen3.6's tokenizer,
mistral's reasoning-variant) sometimes tokenize uncommon Unicode chars
as a sequence of single-byte tokens via the GPT-2 byte_encoder. For
example ministral encodes `❄` (U+2744 = `\xE2\x9D\x84`) as
`[1226, 1157, 1128]` — three single-byte tokens. The merged-token
filter in `_emoji_logit_bias` checks for the 2-byte sequence
`0xE2 0x9X` *within a single token's byte sequence*; when a char splits
to single-byte tokens, none of the individual tokens contains the
2-byte prefix, so the filter misses.

**Practical effect**: emoji-suppression has a few-percent leak rate on
3-byte 0xE2-prefix emoji (❄, ⚡, ☕ occasionally slip through) on
byte-BPE tokenizers. The 4-byte 0xF0 single-byte token IS biased
(byte 0xF0 alone has no continuation bytes that overlap kaomoji
chars), so 4-byte emoji are reliably blocked. Acceptable noise floor
since the dominant emoji ranges are 4-byte. Tokenizer-level
discrimination beyond this would require per-token-decode + emoji-
codepoint allowlist (more expensive, higher maintenance).

### Mistral tokenizer's `encode()` strips non-ASCII chars without `fix_mistral_regex=True`

`AutoTokenizer.from_pretrained('mistralai/Ministral-3-14B-Reasoning-2512')`
loads with a buggy pre-tokenizer regex. `tokenizer.encode("a ★ b")`
silently drops the `★` and returns just `[a, b]`. The warning
mentions `fix_mistral_regex=True` as the workaround.

This affects `_emoji_logit_bias` *sanity checks* (which verify that
specific chars are in/out of the bias dict via encode round-trip) but
NOT actual bias compute — the helper iterates `tokenizer.get_vocab()`
directly and works on the raw vocab strings, bypassing the broken
regex. Confirmed end-to-end: ministral's pilot under emoji suppression
(via `_emoji_logit_bias`, no fix flag) hit 99% kaomoji emit. The
encode failure only matters for verification scripts.

**Workaround**: pass `fix_mistral_regex=True` to
`AutoTokenizer.from_pretrained` in any script that calls `encode()` on
the mistral tokenizer. Saklas's session loader already handles this.

### Python stdout buffering hides long-run progress in tee'd logs

`print()` to a piped stream is block-buffered (~4–8KB). For an 800-generation
run with one progress line per gen, `tee logs/run.log` shows nothing for
30–60 minutes because the buffer doesn't fill. JSONL writes are fine (they
`out.flush()` explicitly). For monitoring during a run: tail JSONL via
`wc -l data/...jsonl`, OR add `flush=True` to `print()` calls (not yet done
— pre-existing scripts work fine for offline review).

