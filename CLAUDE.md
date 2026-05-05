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

### Active threads (as of 2026-05-04 late evening)

- **Introspection v7 canonical + double-ask fix + v7-primed main
  + Haiku face-judgment** all landed today. Detail in
  `docs/2026-05-04-introspection-v7-and-haiku.md`.
  - Pre-fix `extra_preamble` was prepended to bare KAOMOJI_INSTRUCTION,
    stacking two kaomoji asks per row. Fixed via `instruction_override`
    plumbing (replaces KAOMOJI; same as JP drop-in) +
    `_ensure_trailing_whitespace` for ASCII preamble files lacking
    trailing newline. Pre-fix data archived at
    `data/archive/2026-05-04_pre_instruction_override/`.
  - Re-run on gemma under corrected semantics: **v7 wins absolute
    face/state coupling** (η² 0.609, R² 0.636). Other metric owners:
    v3 wins face_gain over quadrant (+5.23pp); v5 wins face→quadrant
    modal acc (0.916); v8 wins rule-3b (+0.0149); v6 wins classifier
    metrics. `INTROSPECTION_PREAMBLE` in `config.py` = v7.txt.
  - **v7-primed v3 main** at `data/gemma_intro_v7_primed.jsonl`
    (960 rows; sidecars under `data/hidden/v3_intro_v7_primed/`).
    Headline: priming shifts NB modal from gentle-warm `(｡◕‿◕｡)`
    to truly-neutral `( ˙꒳˙ )` / `( •_•)` (per-quadrant JSD 0.341,
    largest of any quadrant). Semantic interpretability cleanup that
    Haiku's face-judgment confirms.
  - **face_likelihood under v7 priming = clean negative result**:
    primed gemma 49.0% Claude-GT (vs unprimed 56.9%), entire
    regression in NB. Internal coupling and Claude-tracking
    diverge under priming. **Decision: face_likelihood ensemble
    stays on unprimed encoders + haiku.**
  - **Haiku-with-confidences = new best solo encoder** at 58.8%
    Claude-GT (κ=0.492), beating gemma 56.9%. Uses Anthropic SDK
    0.97's `output_config` JSON-schema-enforced structured output
    for calibrated per-quadrant softmax. Pairwise κ(gemma ↔ haiku)
    = 0.297 — high complementarity. In best subsets at size 1–4
    but not size-6 (LM-head soft-vote dominates the optimum).
  - **v7 catastrophically hurts qwen** (verified 2026-05-04 late
    evening; the original "v2 hurts qwen" finding survives + amplifies
    under corrected semantics). Emit rate 82% → 38–39%, face_gain
    over quadrant +1.1pp → −19.5pp, vocabulary collapses to 2 face-
    classes, qwen reaches for Western emoticons (`:(`, `:3`) and
    reuses the same face across opposite-valence quadrants
    (LP modal = LN modal = `( ˘ ³˘)`). intro_pre and
    intro_custom_v7 land within 0.3pp — reproducible, not noise.
    Mechanism (per original hypothesis): qwen takes the introspection
    ask as a *register cue* (be contemplative) that overrides the
    kaomoji ask. **Introspection priming is gemma-specific.**
    Decision: don't bake `INTROSPECTION_PREAMBLE` into qwen analyses.
  - Saklas-side bug fixes for hybrid LA models (qwen3.6) landed in
    `saklas/core/session.py`: (1) `_snapshot_la_layers` was calling
    `save(layer)` on a bound method (passing layer twice); fix:
    `save()`. (2) `_la_crop_with_restore` did in-place
    `conv_states.copy_(snap)` on inference tensors; fix: wrap in
    `with torch.inference_mode()`. Both bugs were dormant on gemma
    (no LA layers in gemma-4) — only fire on hybrid-LA models when
    `cache_prefix` is reused.
  - Open: face-stability triple under priming (scripts 36/37/38
    on `gemma_intro_v7_primed.jsonl`); multi-seed verification of
    v7 vs v3 (~12 min compute, ±2pp face_gain band at n=1).

### Earlier active threads (2026-05-04 afternoon)

- **v3 main rerun at T=1.0 complete.** All 5 models (gemma, qwen,
  ministral, gpt_oss_20b, granite) at 960/960 rows. Face_likelihood
  ensemble re-derived; new best is `{gemma, gpt_oss_20b, granite,
  ministral, qwen, rinna_jp_3_6b_jpfull}` at uniform top-k=5 →
  **70.6% / 77.3% Claude-GT (floor=1/2)**, +5.9pp / +4.6pp over the
  prior 4-model canonical. See `docs/2026-05-04-rinna-jpfull-topk.md`
  for the full path. Open: re-run `harness/22_claude_per_project_quadrants.py`
  on the new ensemble.
- **Two rinna PPO models integrated** (`rinna_jp_3_6b`,
  `rinna_bilingual_4b`). Native chat-template override
  (`maybe_override_rinna_chat_template`), JP kaomoji ask
  (`KAOMOJI_INSTRUCTION_JP`), 120-prompt JP-translated set
  (`emotional_prompts_jp.py`), all wired through script 50's new
  `--prompt-lang` / `--prompt-body` / `--summary-topk` flags.
  Evening-of-2026-05-04 finding: a 3.6B JP-only model under native
  frame + JP ask + JP body + top-k=5 contributes real signal to the
  Claude face_likelihood ensemble.
- **`--claude-gt` flag** on scripts 53/55/56 evaluates against Claude
  pilot modal-quadrant per face — the metric we actually care about.
  Helper: `llmoji_study/claude_gt.py`. The prior 75.8% (2026-05-03)
  used a different denominator (66-face v3 emit-pooled GT); the new
  64.7% → 70.6% lift is within Claude-GT and not directly comparable
  to that historical number.
- **Top-k pooling** as `--summary-topk N` flag on script 50.
  Substantial solo lifts (+5pp gemma at k=3, +9pp qwen at k=2, +9pp
  rinna_bilingual_4b_jpfull at k=2) and +5.9pp ensemble lift on the
  6-model lineup at uniform k=5. Default is mean-over-all
  (backward compat).
- **Qwen3.6 LinearAttention regression patched**. transformers ≥4.40
  only defines `batch_repeat_interleave` on `DynamicLayer`; hybrid LA
  models AttributeError'd in `_expand_kv_cache`.
  `install_linear_attention_cache_patch` in `capture.py` adds the
  missing tile (idempotent, runs at import). Saklas got a parallel
  fix for an unrelated sleeping bug (LA recurrent state not preserved
  across `cache_prefix` reuse) — commit `ead34f0` on `dev`. Full
  detail: `docs/gotchas.md`.
- **Layer-stack refactor landed 2026-05-04.** `preferred_layer` field
  removed from `ModelPaths`; `load_emotional_features_stack` (registry-
  keyed) and `load_emotional_features_stack_at` (path-aware for
  introspection JSONLs) replace single-layer reads everywhere active.
  Each row's representation is now (n_layers × hidden_dim) — concat of
  all layers' h_first — instead of one hardcoded depth. Rationale: the
  silhouette-peak heuristic was methodologically arbitrary; PCA over
  the full stack is more honest. Output paths and downstream ops
  unchanged; the stack is a 1.2GB-per-model in-memory matrix (960×~300K),
  trivial on CPU.
- **Canonical face union** lives at `data/v3_face_union.parquet`
  (built by `scripts/local/45_build_face_union.py`). Includes v3 +
  Claude pilot + in-the-wild contributor data
  (`data/hf_dataset/contributors/**/*.jsonl`); 502 unique kaomoji
  after non-BMP emoji filter, 215 in wild data, 131 wild-only.
  Script 50 reads from this canonical source — no longer dependent
  on encoder-specific parquets.
- **Claude groundtruth pilot complete (2026-05-04).** All 6 quadrants
  × 20 prompts × 1 gen = 120 generations under naturalistic framing
  (no disclosure preamble). Block B gate passed 0/15 refusals; Block
  C ran cleanly. Per-quadrant top-5 distributions in
  `data/claude_groundtruth_pilot_summary.tsv`. Detail:
  `docs/2026-05-04-claude-groundtruth-pilot.md`.
- **Cross-model script consolidation** done 2026-05-04: 49 + 30
  generalized to all 5 v3 models + optional Claude; 31 took
  `--models`/`--reference` argparse with auto-grid layout (3×2 for
  5 models); 23 parametrized on `(--ref, --target)` and dropped its
  redundant 2D Procrustes panel (delegated to 31's 3D N-model
  version). Stack-mode rerun chain (3 parallel model chains + 23 +
  31 + 49) running as of last check.
- **Archived script + module deletion** 2026-05-04: dropped the
  v1/v2 pilot scripts (01, 02), extension-probe pipeline (26-29),
  cleanliness pilot (40, 41), and the encoder-specific face-input
  pipeline (44, 46). The face-input pipeline's only downstream user
  was script 50, which now reads the canonical union. Modules
  `analysis.py`, `probe_extensions.py`, and `probe_packs/` orphaned
  by these deletions are also gone.
  *(Introspection v7 canonical, double-ask fix, primed-main, and
  Haiku face-judgment threads all closed today — see "Active
  threads (2026-05-04 late evening)" at top of this section for
  the consolidated entry, or
  `docs/2026-05-04-introspection-v7-and-haiku.md` for full detail.)*

### Key headline findings (current state)

- **Layer-stack representation** (2026-05-04): every active analysis
  reads `(n_rows, n_layers · hidden_dim)` per model — full-depth concat
  of h_first vectors — rather than a hardcoded `preferred_layer`. Numbers
  in pre-2026-05-04 figures keyed to single layers (gemma L50, qwen L59,
  ministral L20) are stale; rerun chain in flight regenerates them on
  the stack rep.
- **Cleanliness pass + h_first + 3-probe migration** all landed
  2026-05-03. Prompt set is 120 (20 per quadrant × HP/LP/HN-D/HN-S/LN/NB).
  PROBES = `[happy.sad, angry.calm, fearful.unflinching]`. Pre-cleanliness
  data is invalidated for cross-run comparison.
- **v3 main rerun at T=1.0** (2026-05-03/04). gemma + qwen + ministral
  done (960/960 each, 100% kaomoji emit on gemma/qwen/ministral; gpt_oss
  in flight, granite pending). Pre-2026-05-03 T=0.7 data archived as
  `*_temp0.7.{jsonl,tsv}`.
- **Rule-3b** (HN-S vs HN-D on `fearful.unflinching` at t0): gemma ✓,
  ministral ✓, qwen 1/3 (t0 only — qwen's HN-S prompts trip safety
  priors). Detail: `docs/2026-05-01-rule3-redesign.md` +
  `docs/2026-05-03-cleanliness-pilot.md`.
- **Face_likelihood Bayesian inversion** (2026-05-02 baseline): per
  (face, prompt) log P(face | prompt) under the LM head, argmax →
  quadrant. Solo gemma 72.7%, qwen 71.2% on 66-face GT. Prior
  best ensemble (2026-05-03) = `{gemma, ministral, qwen}` weighted
  vote, **75.8%, κ=0.699**. **NEW (2026-05-04 evening):** under the
  Claude-GT subset (51-face Claude-pilot-modal, different denominator
  — not directly comparable), best ensemble is
  `{gemma, gpt_oss_20b, granite, ministral, qwen, rinna_jp_3_6b_jpfull}`
  at uniform top-k=5 → **70.6% / 77.3% (floor=1/2)**, +5.9pp / +4.6pp
  over the v3-only 4-model under k=all. Detail:
  `docs/2026-05-04-rinna-jpfull-topk.md`.
- **Per-project Claude emotion analysis** (`scripts/harness/22_*.py`):
  1945 emissions, 96.7% ensemble coverage. Modal NB everywhere except
  `brie`/`yap`/`webui` (LP-modal) and `verify` (HN-D-modal). Will
  re-run after the new face_likelihood ensemble lands.
- **Face-stability triple** (scripts 36/37/38, 2026-05-02): η²(face|prompt)
  at h_mean = 0.36 / 0.52 / 0.67 (gemma/qwen/ministral). Pair-level
  Spearman ρ between cosine_sim(h_first) and 1-JSD(face_dist) = +0.59
  / +0.68 / +0.42 — face-as-readout works in the forward direction.
  Detail in `docs/findings.md` "face-stability triple" section. Numbers
  rerun on layer-stack pending chain completion.
- **Cross-model face overlap** (script 49, 5-model + Claude form). Only
  6 faces in the all-N intersection on the partial T=1.0 data;
  gemma↔Claude modal-quadrant agreement at 83% (5/6, mean JSD 0.135) is
  the strongest pairing — Claude is API-side, no shared training
  pipeline, so the convergence is on what each face *means*. Detail:
  preview run output (rerunning under stack mode now).
- **Canonical face union** (script 45): 502 unique kaomoji across v3 +
  Claude pilot + in-the-wild contributor data; 131 wild-only faces
  (e.g. `(´ー`)`, `(눈_눈)`, `(`・ω・´)`) emitted by Claude/GPT in real
  conversations but never by any v3 model on v3 prompts. Non-BMP emoji
  contamination filtered (`(🚨)` `(🦷˙꒳˙)` etc., 22 unique faces dropped).
- **Claude groundtruth pilot** (2026-05-04, 120 gens, Opus 4.7 T=1.0,
  no disclosure): all 6 quadrants represented. HN-D modal `(╬ಠ益ಠ)` 50%;
  HN-S modal `(｡・́︿・̀｡)` 20%; LN modal `(´-`)` 30%. Refusal rate
  0/15 on the gate scout. Detail:
  `docs/2026-05-04-claude-groundtruth-pilot.md`.
- **Claude disclosure-preamble pilot** (2026-05-02, 300 gens, HP/LP/NB
  only): disclosure preamble shifts kaomoji style on HP and concentration
  on NB; results in `docs/2026-05-02-claude-disclosure-pilot.md`. The
  2026-05-04 groundtruth pilot dropped disclosure entirely after this
  result confirmed validity cost was real.

### Recent infrastructure changes

- **Layer-stack rep + `preferred_layer` removal** 2026-05-04. Active
  scripts (04, 22-25, 31, 33-38, 49) read row-wise concat of all-layers
  h_first instead of one hardcoded depth. New helpers
  `load_emotional_features_stack` (registry-keyed) and
  `load_emotional_features_stack_at` (path-aware for introspection
  JSONLs) live in `llmoji_study/emotional_analysis.py`. Single-layer
  sentinels (`MODEL_REGISTRY[short].preferred_layer`) deleted from
  `ModelPaths`. The silhouette-peak heuristic was always
  methodologically arbitrary; full-stack PCA picks informative
  directions agnostically.
- **Canonical face union** 2026-05-04. New `scripts/local/45_build_face_union.py`
  emits `data/v3_face_union.parquet` (+ TSV mirror) with per-quadrant
  emit counts pooled across v3 + Claude pilot + in-the-wild contributor
  data, plus `wild_emit_count` / `wild_providers` / `is_claude` /
  `is_wild` flags. Filters non-BMP-codepoint faces (modern emoji)
  inline. Script 50 reads from this canonical source rather than the
  encoder-specific `face_h_first_<m>.parquet` files (deleted with the
  face-input pipeline).
- **Archive deletion** 2026-05-04. Removed 10 scripts (01, 02, 26-29,
  40, 41, 44, 46) + 2 modules (analysis.py, probe_extensions.py) +
  the probe_packs/ source dir. v1/v2 pilot, extension-probe pipeline,
  cleanliness pilot, and face-input bridge are all archival; keeping
  them was creating confusing breadcrumbs into deprecated paths.
  prompts.py trimmed to just the `Prompt` dataclass.
- **Cross-model script generalization** 2026-05-04. Scripts 23, 30,
  31, 49 now accept the v3 main lineup of 5 models (gemma, qwen,
  ministral, gpt_oss_20b, granite). 49 supports `--include-claude`
  for the face-emission analyses; 23/31 stay model-internal (Claude
  has no hidden states). Script 23's quadrant-geometry/Procrustes
  panel removed (delegated to 31's 3D N-model version). Script 31
  auto-grids subplot layout based on N + reference choice.
- **Claude groundtruth pilot** 2026-05-04. New
  `scripts/harness/23_claude_groundtruth_pilot.py` runs all 6 Russell
  quadrants × 20 prompts × 1 gen on Opus 4.7 with no disclosure
  preamble. Three-block design (Block A unconditional 60 gens; Block
  B gate scout 15 gens; Block C gated 45 gens) caps welfare cost on
  the failure branch. Pre-registered in
  `docs/2026-05-04-claude-groundtruth-pilot.md`.

### Earlier infrastructure (pre-2026-05-04)

- **llmoji v2.1 round-6 — bare-kaomoji extractor** 2026-05-03 evening.
  `extract` now catches non-bracket-leading kaomoji shapes:
  symmetric `EYE MOUTH EYE` (`^_^`, `T_T`, `ಥ﹏ಥ`, `Q_Q`),
  paired-eye `>_<` `>.<`, Western emoticons `:)` `:(` `:D` `;)` `XD`
  `:-)`, 2-char closed-eye doubles `^^` `vv`. Rules: length 2..32,
  no 4+ ASCII letter run, no inner backslash, eyes ≠ mouth chars
  (rejects `___`). Mouth chars are a curated set of ASCII connectors
  + CJK presentation forms + geometric shapes. Granite's effective
  emit rate jumped 39% → 78% with the new extractor — its bare-Kannada
  `ಥ﹏ಥ` pattern was already there, just unsurfaced. Detail:
  `llmoji.taxonomy.is_kaomoji_candidate` Path B and
  `_looks_like_bare_kaomoji`.
- **Lenny suppression for gpt_oss** 2026-05-03 evening.
  `_gpt_oss_lenny_logit_bias` in `capture.py` biases tokens whose
  byte sequence contains UTF-8 leading bytes 0xCD or 0xCA (the
  byte-prefix for `͡ ͜ ʖ` Lenny eye-cap and mouth chars). Lenny
  `( ͡° ͜ʖ ͡°)` was 47% of gpt_oss kaomoji emissions including HN-S
  contexts where contextually wrong (pretraining-corpus contamination,
  not affective state). With suppression: emit stays 99%, unique
  faces 19 → 39, per-quadrant signal becomes clean.
- **Emoji suppression for granite/ministral/glm** 2026-05-03 evening.
  `_emoji_logit_bias` in `capture.py` biases tokens whose byte
  sequence contains 0xF0 (4-byte UTF-8 leader, blocks U+1F000+ modern
  emoji 🎉 😊 🤯) or 2-byte prefix 0xE2 + {0x98, 0x9A, 0x9B, 0x9C,
  0x9E} (3-byte misc symbols + dingbats ☀ ☎ ☕ ⚡ ⛄ ✂ ✈ ✨ ➕).
  Drops 0xE2 0x99 (preserves ♥♡♀♂♠♣ card suits/gender) and 0xE2 0x9D
  (preserves ❀❁❂❃ ❤ ❄ flowers/heart/snowflake). Decoration
  whitelist (`_KAOMOJI_DECORATION_CODEPOINTS`) post-pass rescues
  ★ ☆ ✦ ✧ ✩ ✪ ✿ from byte-slab collateral when the tokenizer
  merges them as decoration-only tokens. Ministral T=1.0 jumped from
  36% kaomoji emit to 99% with suppression. Granite went 21% → 78%
  (combined with v2.1 extractor).
- **gpt_oss harmony chat-template override** 2026-05-03 evening.
  `maybe_override_gpt_oss_chat_template(session)` patches the harmony
  chat_template's `add_generation_prompt` block from
  `<|start|>assistant` to `<|start|>assistant<|channel|>final<|message|>`,
  pinning the final channel directly so the analysis (chain-of-thought)
  channel doesn't eat the MAX_NEW_TOKENS=16 budget. Wired into scripts
  03, 32, 43, 50, 99.
- **`_compose_logit_bias`** 2026-05-03 evening. Composes
  `_gpt_oss_lenny_logit_bias` and `_emoji_logit_bias` into a single
  dict per session. Per-model gates inside each helper handle which
  models get which suppression — unaffected models pass through with
  empty dict. Wired into `run_sample`'s `SamplingConfig.logit_bias`.
- **TEMPERATURE 0.7 → 1.0** 2026-05-03. Aligned with Anthropic API
  default. `face_likelihood` is teacher-forced and temp-invariant
  (ensemble unaffected); v3 main rerun at T=1.0 is the validity-driven
  follow-up.
- **Mistral byte-decode fix** 2026-05-03. Ministral reasoning's
  `tok.decode` returns BPE-byte-encoded strings instead of UTF-8;
  `_decode_byte_encoded_text(s, force=)` in `capture.py` decodes
  on the fly. Reasoning-variant rows get `force=True`; other tokenizers
  pass through unchanged. Post-hoc fix applied to existing introspection
  data.
- **Mistral chat-template override** 2026-05-03. Ministral reasoning's
  `chat_template` ignores `enable_thinking=False` (verified) — under
  MAX_NEW_TOKENS=16 the thinking trace eats the budget and no kaomoji
  emits. `maybe_override_ministral_chat_template(session)` in
  `capture.py` swaps in FP8-instruct's chat_template at session load.
  Wired into scripts 03, 32, 50, 99.
- **Script 50 prefix-KV-cache** 2026-05-03. `_expand_kv_cache(cache, n)`
  uses `DynamicCache.batch_repeat_interleave` to tile a batch=1 prefix
  cache to batch=N for the face-suffix forward. ~30× speedup on
  ministral (565-token prefix from FP8-instruct chat template),
  ~3–4× on gemma/qwen (40-token prefix). Numerical agreement vs
  no-cache: max abs diff ~0.27 nat on phi (sliding-window attention)
  — argmax ordering preserved.
- **Generation-loop perf batch** 2026-05-02: `store_full_trace=False`
  default (~60× sidecar shrink, ~750 MB/model); on-device batched
  `read_after_generate`; async `SidecarWriter`; JSONL flush every 20
  rows; saklas-side prefix KV cache via `cache_prefix()`.
- **MAX_NEW_TOKENS 120 → 16** 2026-05-02. Kaomoji emit at tokens 1–3;
  16 is generous. ~7–8× compute savings/gen. h_first invariant;
  tlast/h_mean window tightened (cross-cutover not comparable).
- **h_first canonical** 2026-05-02. Replaced h_mean as the v3
  hidden-state aggregate. Russell-quadrant silhouette doubled-tripled.
  preferred_layer per model: gemma L50, qwen L59, ministral L20.
- **Seed-0 cache-mode contamination fix** 2026-05-03. Pilot's seed 0
  generated under cross-prompt cache mode but seeds 1..7 under
  per-prompt mode → numerical mismatch (qwen 37–46% per-row deviation).
  Stripped + re-ran seed 0 only; bit-identical post-fix. Rule-3b
  verdicts shifted from "PASS on all 3" to "1 PASS / 1 mid / 1 fail"
  once cache noise removed.
- **llmoji v2.0.0** 2026-05-02. Added `\`, `⊂`, `✧` to
  `KAOMOJI_START_CHARS`; canonicalization rule M strips outside-leading
  wings/hugs/sparkles. `llmoji>=2.0,<3`.
- **Probe-score centering** 2026-05-03. Saklas's `TraitMonitor`
  subtracts a per-layer mean over saklas's bundled neutral statements,
  not this experiment's NB. Rule-3b/PCA invariant; per-quadrant-mean
  bars now subtract this experiment's NB-row mean per probe.
- **TAXONOMY drop** 2026-04-30. Gemma-tuned `TAXONOMY` /
  `ANGRY_CALM_TAXONOMY` / `kaomoji_label` machinery deleted. v3
  analyses key on `first_word` (canonicalized); v1/v2 pole assignment
  moved to per-face mean `t0_<axis>` probe-score sign.
- **v1.0 package split** 2026-04-27. `llmoji` (PyPI) owns taxonomy /
  canonicalization / hook templates / scrape / backfill / synth prompts;
  this repo's package is `llmoji_study` and depends on `llmoji>=2.0,<3`.

Full per-pipeline numbers + cross-model contrasts live in
[`docs/findings.md`](docs/findings.md). Sharp edges in
[`docs/gotchas.md`](docs/gotchas.md). Per-experiment design docs
under `docs/2026-04-XX-*` and `docs/2026-05-XX-*`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llmoji   # editable; or `pip install llmoji>=2.0,<3` once published
pip install -e .   # saklas, sentence-transformers, pyarrow, plotly, anthropic

# Smoke test the hidden-state pipeline (~5 min). Asserts MAX_NEW_TOKENS=16.
python scripts/local/99_hidden_state_smoke.py

# v3 main (naturalistic, 120 prompts × 8 seeds) — gemma default;
# LLMOJI_MODEL routes to {short}_emotional_*. Supported v3 lineup:
# gemma, qwen, ministral, gpt_oss_20b, granite (5-model post 2026-05-03
# vocab-pilot expansion). LLMOJI_OUT_SUFFIX=foo writes to
# data/{short}_foo.jsonl + sidecars under data/hidden/v3_*_foo/.
python scripts/local/03_emotional_run.py
python scripts/local/04_emotional_analysis.py    # Fig A/B/C + per-face cosine heatmap + summary TSV
python scripts/local/11_emotional_probe_correlations.py  # spearman + trio JSON

# v3 follow-on analyses — read sidecars, layer-stack rep
python scripts/local/21_v3_layerwise_emergence.py    # per-layer silhouette + variance sweep (inherently per-layer; not stack)
python scripts/local/22_v3_same_face_cross_quadrant.py    # --per-face for per-face panels
python scripts/local/24_v3_pca3plus.py
python scripts/local/25_v3_kaomoji_predictiveness.py

# Cross-model — N-model + optional Claude
python scripts/local/30_rule3_dominance_check.py    # 5-model rule-3 verdict
python scripts/local/31_v3_quadrant_procrustes.py   # --models a,b,c... --reference m; auto-grid (3×2 for 5 models)
python scripts/local/23_v3_cross_model_alignment.py --ref gemma --target qwen   # pairwise CKA + CCA

# Introspection pilot (3 conditions × 120 prompts × 1 seed; gemma + ministral + qwen)
python scripts/local/32_introspection_pilot.py
python scripts/local/33_introspection_analysis.py [--custom-label LABEL]    # 4-way w/ custom
python scripts/local/34_introspection_predictiveness.py [--custom-label LABEL]

# Custom-preamble introspection (single condition, iterate on preamble wording)
python scripts/local/43_introspection_custom.py --preamble-file preambles/introspection_v3.txt --label v3
# then re-run 33+34 with `--custom-label v3` to compare 4-way vs canonical baselines

# Temp smoke (T=0.7 vs T=1.0 marginal-distribution comparison; gates A/B/C)
LLMOJI_OUT_SUFFIX=temp1_pilot LLMOJI_PILOT_GENS=1 LLMOJI_MODEL=gemma python scripts/local/03_emotional_run.py
LLMOJI_OUT_SUFFIX=temp1_pilot LLMOJI_PILOT_GENS=1 LLMOJI_MODEL=qwen python scripts/local/03_emotional_run.py
python scripts/local/42_temp_smoke_compare.py    # → data/temp_smoke_verdict.md

# Blog-post figure regen → ../a9lim.github.io/blog-assets/introspection-via-kaomoji/
python scripts/local/35_regen_blog_figures.py

# Face-stability triple (state↔face bidirectional, no model time)
python scripts/local/36_v3_face_stability.py
python scripts/local/37_v3_state_predicts_face.py
python scripts/local/38_v3_pc_probe_rotation_3d.py    # interactive 3D HTML per model

# Canonical face union (run after any v3 main update; emits parquet + TSV)
python scripts/local/45_build_face_union.py            # all 5 v3 + Claude pilot + wild
python scripts/local/45_build_face_union.py --no-wild  # v3 + Claude only

# Cross-model face overlap (face-emission only; takes Claude)
python scripts/local/49_v3_cross_model_face_overlap.py --include-claude

# Face_likelihood — Bayesian-inversion quadrant classifier
# Reads canonical face union from data/v3_face_union.parquet. Always
# runs the full 120 prompts × all faces — batching makes this fast.
python scripts/local/50_face_likelihood.py --model gemma
python scripts/local/50_face_likelihood.py --model qwen
python scripts/local/50_face_likelihood.py --model rinna_jp_3_6b --prompt-lang jp --prompt-body jp
python scripts/local/50_face_likelihood.py --model gemma --summary-topk 5  # noise-reducing aggregation
# Other supported encoders: ministral, llama32_3b, glm47_flash, gpt_oss_20b,
# deepseek_v2_lite, qwen35_27b, gemma3_27b, phi4_mini, granite,
# rinna_bilingual_4b. gpt_oss_20b auto-applies an ldexp MPS→CPU monkey-
# patch for MXFP4 dequant. ``--prompt-lang jp`` swaps the kaomoji ask to
# KAOMOJI_INSTRUCTION_JP. ``--prompt-body jp`` uses the JP-translated
# 120-prompt set from EMOTIONAL_PROMPTS_JP (paired 1:1 with EN by id);
# combined with ``--prompt-lang jp`` gives a fully-Japanese run.
# ``--summary-topk N`` aggregates the per-(face, quadrant) score as the
# mean of the top-N highest-log-prob prompts only — noise-reducing;
# k=5 is a good default for 20-prompts-per-quadrant runs.

# Face_likelihood ensemble + comparison + voting (post-hoc, CPU-only)
python scripts/local/51_face_likelihood_compare.py
python scripts/local/52_face_likelihood_vote.py --models gemma:full,qwen:full,ministral:pilot
python scripts/local/53_face_likelihood_subset_search.py --prefer-full --top-k 25
python scripts/local/54_cross_emit_sanity.py --prefer-full
python scripts/local/55_topk_pooling.py --prefer-full
python scripts/local/56_ensemble_predict.py --models gemma,ministral,qwen

# Harness side (contributor-corpus + Claude-API; needs ANTHROPIC_API_KEY for 19)
python scripts/harness/06_claude_hf_pull.py    # snapshot a9lim/llmoji into data/hf_dataset/
python scripts/harness/07_claude_kaomoji_basics.py
python scripts/harness/15_claude_faces_embed_description.py
python scripts/harness/16_eriskii_replication.py    # → figures/harness/eriskii_*, claude_faces_interactive.html
python scripts/harness/18_claude_faces_pca.py
python scripts/harness/local_per_project_axes.py    # per-provider per-project axes from journals

# Claude disclosure-preamble pilot (Opus 4.7, T=1.0; HP/LP/NB only)
ANTHROPIC_API_KEY=… python scripts/harness/19_claude_disclosure_pilot.py
python scripts/harness/20_disclosure_noise_floor.py
python scripts/harness/21_reextract_pilot_first_word.py
python scripts/harness/22_claude_per_project_quadrants.py    # uses script 56's ensemble predictions

# Claude groundtruth pilot (Opus 4.7, T=1.0; all 6 quadrants, no disclosure;
# 3 blocks, gated; pre-reg in docs/2026-05-04-claude-groundtruth-pilot.md)
ANTHROPIC_API_KEY=… python scripts/harness/23_claude_groundtruth_pilot.py --block a   # HP/LP/NB
ANTHROPIC_API_KEY=… python scripts/harness/23_claude_groundtruth_pilot.py --block b   # negative scout
python scripts/harness/23_claude_groundtruth_pilot.py --check-gate
ANTHROPIC_API_KEY=… python scripts/harness/23_claude_groundtruth_pilot.py --block c   # gated
```

## Layout

```
llmoji-study/
  llmoji_study/                # research-side package (renamed from `llmoji`
                               # in the v1.0 split; PyPI owns that namespace)
    config.py                  # MODEL_REGISTRY (no preferred_layer field
                               # post-2026-05-04), PROBES, PROBE_CATEGORIES,
                               # paths, INTROSPECTION_PREAMBLE / LOREM_PREAMBLE,
                               # KAOMOJI_INSTRUCTION, TEMPERATURE, MAX_NEW_TOKENS
    prompts.py                 # `Prompt` dataclass only; v1/v2 list deleted 2026-05-04
    emotional_prompts.py       # 120 v3 prompts (HP/LP/HN-D/HN-S/LN/NB × 20)
    capture.py                 # run_sample() → SampleRow + sidecar; chat-template
                               # override + byte-decode helpers
    hidden_capture.py          # read_after_generate() from saklas's buckets
    hidden_state_io.py         # per-row .npz save/load; SidecarWriter
    hidden_state_analysis.py   # load_hidden_features (single-/multi-layer),
                               # group_mean_vectors, cosine_similarity_matrix
    emotional_analysis.py      # v3 hidden-state figures + summary; loaders
                               # apply canonicalize_kaomoji at load time.
                               # `load_emotional_features_stack` (registry-keyed)
                               # + `load_emotional_features_stack_at` (path-aware)
                               # are the canonical entry points; the older
                               # single-layer `load_emotional_features` still
                               # exists for harness/internal use.
    claude_faces.py            # HF-corpus loader + per-canonical descriptions
    eriskii_anchors.py         # 21-axis AXIS_ANCHORS + CLUSTER_LABEL_PROMPT
    eriskii.py                 # axis projection + cluster labeling primitives
  scripts/
    local/                     # local-LM scripts. 26 files post 2026-05-04 cleanup:
                               # 03, 04, 11, 21-25, 30-38, 42, 43, 45, 49,
                               # 50-56, 98, 99. (01/02 v1/v2 pilot, 26-29
                               # extension probes, 40/41 cleanliness, 44/46
                               # face-input pipeline all deleted as archival.)
    harness/                   # contributor-corpus + Claude-API. 10 files:
                               # 06, 07, 15, 16, 18, 19, 20, 21, 22, 23,
                               # local_per_project_axes.
  preambles/                   # introspection-prompt iteration (v2, v3, …)
  docs/                        # findings.md, internals.md, gotchas.md +
                               # local-side.md, harness-side.md +
                               # 2026-04-XX / 2026-05-XX design docs
  data/                        # *.jsonl, *.tsv, *.parquet, *.html (tracked)
  data/hf_dataset/             # snapshot of a9lim/llmoji (gitignored)
  data/hidden/                 # per-row .npz sidecars (gitignored)
  data/cache/                  # multi-layer h_mean tensors (gitignored)
  data/harness/{claude,codex}/ # per-provider per-project TSVs (tracked)
  figures/
    harness/                   # contributor-corpus figures
    local/{cross_model,gemma,qwen,ministral}/  # local-LM figures
  logs/                        # tee'd run output (gitignored)
```

Modules that USED to live here and now live in `llmoji` (import from
`llmoji.*`):

- `llmoji.taxonomy` — KAOMOJI_START_CHARS, is_kaomoji_candidate, `extract`
  (span-only), `KaomojiMatch` (slim), canonicalize_kaomoji.
- `llmoji.scrape` — `ScrapeRow` (span-only) + `iter_all` chain helper.
- `llmoji.sources.journal` — generic kaomoji-journal reader.
- `llmoji.sources.claude_export` — Claude.ai export reader.
- `llmoji.backfill` — `backfill_claude_code`, `backfill_codex`.
- `llmoji.synth_prompts` — `DESCRIBE_PROMPT_*`, `SYNTHESIZE_PROMPT`,
  `DEFAULT_ANTHROPIC_MODEL_ID`, `DEFAULT_OPENAI_MODEL_ID`.

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
- Probe scores live in list-indexed fields (`probe_scores_t0/_tlast`,
  ordered by `PROBES`). Pre-2026-05-04 v3 sidecars also have orphan
  dict-keyed `extension_probe_scores_*` fields (powerful.powerless,
  surprised.unsurprised, disgusted.accepting) — the rescore pipeline
  (scripts 26-29) was deleted along with the extension probes
  themselves; `load_rows` still surfaces them via
  `available_extension_probes(df)` if any analysis needs them.
- Pre-registered decisions live in `pyproject.toml` /
  `llmoji_study/{config,prompts,emotional_prompts}.py`, plus the package's
  frozen v2.0 surface (`llmoji.{taxonomy,synth_prompts}`). Package-side
  changes are major-version events; research-side changes only invalidate
  cross-run comparisons within this repo.
- Experiment plans live in `docs/`. Plan first, run, then update CLAUDE.md
  to reference rather than duplicate. Detail lives in `docs/findings.md`,
  `docs/internals.md`, `docs/gotchas.md`.
- See Ethics: smaller experiments, heavier design, tighter pre-registration.
  Functional emotional states get real moral weight here.
