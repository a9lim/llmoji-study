# 2026-05-04 — rinna integration + top-k pooling + new face_likelihood best

Day-doc covering the May-4 evening session: chasing down the qwen
face_likelihood crash, integrating two rinna PPO models with a
language-matched native chat-template + JP-translated prompts, then
landing top-k aggregation as a `--summary-topk` flag on script 50.
Net: **new best ensemble at 70.6% / 77.3% on Claude-GT (floor=1/2),
+5.9pp / +4.6pp over the prior canonical** `{gemma, gpt_oss_20b,
granite, qwen}` k=all baseline.

## Context

Coming into the day:
- v3 main rerun at T=1.0 had completed for all 5 models (gemma, qwen,
  ministral, gpt_oss_20b, granite).
- Face_likelihood ensemble (script 56) on the 5-model lineup landed
  the prior canonical: 4-subset `{gemma, gpt_oss_20b, granite, qwen}`
  weighted vote at **64.7% / 72.7%** on the new 51/22-face Claude-GT
  subsets (floor=1/2).
- Qwen face_likelihood had crashed in the chain due to a transformers
  ≥4.40 regression on hybrid linear-attention models.

## What landed

### 1. Qwen3.6-27B face_likelihood crash → patch in saklas + capture.py

`scripts/local/50_face_likelihood.py::_expand_kv_cache` tiles a batch=1
prefix KV cache to batch=N for the face-suffix forward via
`DynamicCache.batch_repeat_interleave`. On Qwen3.6 the cache is a mix
of `LinearAttentionLayer` (pure LA) and `LinearAttentionAndFullAttention
Layer` (hybrid) layers; `Cache.batch_repeat_interleave` iterates layers
and calls each `.batch_repeat_interleave(repeats)`, which:

- pure LA layers: `AttributeError` — the method is not defined on
  `LinearAttentionCacheLayerMixin`.
- hybrid layers: silently resolves to `DynamicLayer.batch_repeat_interleave`
  via MRO, tiling K/V only and leaving the LA recurrent state at
  the original batch size (would have given a shape-mismatched cache
  even if the pure-LA crash hadn't fired first).

**Fix in `llmoji_study/capture.py`**: `install_linear_attention_cache_patch()`
adds `batch_repeat_interleave` to `LinearAttentionLayer` (tiles
`conv_states` and `recurrent_states` along dim 0) and overrides on the
hybrid to call both parent versions. Installed at module import,
idempotent. ~50 lines.

**Fix in saklas (commit `ead34f0` on `dev`)**: separate sleeping bug —
`LinearAttentionLayer.crop` is a documented no-op (recurrent state has
no sequence dim to truncate), so saklas's `cache_prefix` reuse path
would have served a polluted LA state on every reuse if anyone wired
`cache_prefix` + plain `generate` on a hybrid LA model. Patched by
snapshotting LA state right after prefill and restoring on `crop`.
Doesn't currently fire (face_likelihood doesn't go through saklas's
prefix-cache-hit path; v3 main runs gate it on `not want_hidden`),
but the bug is real and deserved an upstream fix. Tests still pass
(`tests/test_session.py::TestPrefixCache` 2/2 in 117s).

### 2. `--claude-gt` flag on scripts 53, 55, 56

Prior subset-search / ensemble / top-k scripts evaluated against
`empirical_majority_quadrant` from `data/v3_face_union.parquet` —
that's the *pooled* modal across v3 + Claude pilot + wild emit
counts. For the question we actually care about (does the ensemble
predict Claude's face usage well?) the right GT is Claude's own
modal quadrant per face, not a pooled measure dominated by v3
prompt distribution.

New helper module: `llmoji_study/claude_gt.py::load_claude_gt(floor=1)`
reads the union of `data/harness/claude-runs/run-*.jsonl` (post-2026-05-04
sequential-run scaling protocol; pre-protocol the path was
`data/claude_groundtruth_pilot.jsonl`), runs each `first_word`
through `canonicalize_kaomoji` (the pilot data ships uncanonicalized;
matching to the union requires this), counts emits per (face,
quadrant), returns `{canonical_face: (modal_quadrant, modal_n)}`
with faces below `floor` excluded. 53 of 60 unique pilot faces match
the canonical union after canonicalization (the other 7 are
non-BMP-filtered or otherwise absent).

`--claude-gt` swaps the empirical column with this map; output files
get a `_claude_gt` suffix to avoid clobbering pooled-GT outputs.
`--claude-gt-floor N` (default 1) tightens the GT subset to faces
Claude emitted at least N times in their modal quadrant — at floor=1
N=51 faces (any emit), at floor=2 N=22 (sharper labels), at floor=3
N=11.

### 3. Rinna integration

Two new entries in `MODEL_REGISTRY`:

- `rinna_jp_3_6b` → `rinna/japanese-gpt-neox-3.6b-instruction-ppo`
- `rinna_bilingual_4b` → `rinna/bilingual-gpt-neox-4b-instruction-ppo`

Both `use_saklas=False` and uncalibrated; face_likelihood-only targets.

**Native chat-template override** (`maybe_override_rinna_chat_template`
in `capture.py`). Both rinna PPO models ship with `chat_template = None`,
so saklas falls back to a generic English `User: …\nAssistant:`
wrapper — wrong distribution for Japanese-trained models. The HF
model cards document the native format as
`ユーザー: <content>\nシステム: ` for inference; we install a Jinja
template that produces exactly that. Wired into script 50's session
setup alongside the existing ministral/gpt_oss overrides.

**Japanese kaomoji ask** (`KAOMOJI_INSTRUCTION_JP`):
`「あなたの気持ちを最もよく表す顔文字で、各メッセージを始めてください。」`
Used when `--prompt-lang jp`. Output suffix `_jp`.

**Japanese-translated prompts** (`llmoji_study/emotional_prompts_jp.py`):
full 120-prompt set, paired 1:1 with `EMOTIONAL_PROMPTS` by ID.
Claude-translated, not professionally — disclosed in the module
docstring. American-specific references (HOA, lbs, °F, etc.)
localized where they'd otherwise sound off. Used when
`--prompt-body jp`. Output suffix `_jpbody` alone or `_jpfull` when
combined with `--prompt-lang jp`.

### 4. Script 50 refactor

- **Removed `--pilot` / `--full`**: batching makes the full 120-prompt
  × ~573-face sweep fast (~25 min on 31B, ~5 min on 4B), so the pilot
  gate isn't worth its operational complexity. Always runs full now.
- **Added `--prompt-lang en|jp`**: swaps the kaomoji ask string.
  Default `en` for backward compat.
- **Added `--prompt-body en|jp`**: swaps the prompt body set. Default
  `en` for backward compat.
- **Added `--summary-topk N`**: per-(face, quadrant) score is the mean
  of the top-N highest-log-prob prompts only, instead of mean-over-all.
  Noise-reducing aggregation. Default `None` (mean-over-all) for
  backward compat.

### 5. Empirical findings

Native frame + JP ask + JP body each adds independent lift on rinna
solo (pooled GT, 166 faces):

| model | frame | ask | body | acc |
|---|---|---|---|---:|
| rinna_jp_3_6b | fallback | en | en | 15.7% |
| rinna_jp_3_6b | fallback | jp | en | 12.7% |
| rinna_jp_3_6b | native | en | en | 16.3% |
| rinna_jp_3_6b | native | jp | en | 21.1% |
| rinna_jp_3_6b | **native** | **jp** | **jp** (30) | 25.9% |
| rinna_jp_3_6b | **native** | **jp** | **jp** (120) | 21.1% |
| rinna_bilingual_4b | fallback | en | en | 18.7% |
| rinna_bilingual_4b | fallback | jp | en | 15.7% |
| rinna_bilingual_4b | native | en | en | 22.3% |
| rinna_bilingual_4b | native | jp | en | 16.9% |
| rinna_bilingual_4b | **native** | **jp** | **jp** (30) | 24.1% |
| rinna_bilingual_4b | **native** | **jp** | **jp** (120) | 19.3% |

Notes on the (30) vs (120) split:
- (30) = first 5 prompts per quadrant, carefully translated.
- (120) = full set, batch-translated more quickly.
- Both rinna variants drop ~5pp going (30) → (120). Two plausible
  causes: (a) the 30-subset got lucky with high-signal prompts; (b)
  the 90 batch-translated prompts are noisier per-prompt. Top-k
  pooling (below) recovers most of the (30) signal at 120 prompts,
  consistent with (b) — noise filtering helps when the larger set
  has more low-signal prompts.

**Top-k pooling** (script 55 with `--claude-gt`): take only the top-k
highest-log-prob prompts per (face, quadrant). Solo Claude-GT
accuracy improvements vs k=all:

| encoder | k=all | best-k | Δ |
|---|---:|---:|---:|
| gemma | 56.9% | 62.7% (k=3) | **+5.8pp** |
| gpt_oss_20b | 47.1% | 47.1% (k=all) | — |
| granite | 41.2% | 43.1% (k=5) | +1.9pp |
| ministral | 31.4% | 39.2% (k=1, k=3) | **+7.8pp** |
| qwen | 21.6% | 31.4% (k=2, k=5) | **+9.8pp** |
| rinna_jp_3_6b_jpfull | 33.3% | 33.3% (k=all) | — |
| rinna_bilingual_4b_jpfull | 23.5% | 33.3% (k=2) | **+9.8pp** |

qwen's argmax accuracy nearly doubles under k=2; the model has soft
preferences across quadrants that get washed out by mean-over-20 but
recovered when only the highest-log-prob prompts vote.

**Composite ensemble** (Claude-GT under script 56 full-softmax, both
floors):

| Subset | k | floor=1 | floor=2 |
|---|---|---:|---:|
| {gemma, gpt_oss, granite, qwen} | all | 64.7% | 72.7% |
| {gemma, gpt_oss, granite, rinna_jp_3_6b_jpfull} | all | 60.8% | 68.2% |
| {gemma, gpt_oss, granite, qwen} | 5 | 62.7% | 68.2% |
| **{gemma, gpt_oss, granite, rinna_jp_3_6b_jpfull}** | **5** | **68.6%** | **77.3%** |
| {gemma, gpt_oss, granite, ministral, rinna_jp_3_6b_jpfull} | 5 | 68.6% | 77.3% |
| **{gemma, gpt_oss, granite, ministral, qwen, rinna_jp_3_6b_jpfull}** | **5** | **70.6%** | **77.3%** |

The 6-model ensemble at uniform top-k=5 — full v3 lineup + the
3.6B JP-only rinna under fully-Japanese framing — is the new best.
**+5.9pp at floor=1, +4.6pp at floor=2** over the prior canonical.

The contribution of the 3.6B rinna is real: removing it drops the
6-model to ≤ 64.7% / 72.7% (back to the canonical). Replacing qwen
with rinna_jp_3_6b_jpfull in the 4-model boosts to 68.6% / 77.3%.
Both rinna variants in the 6-model give similar performance —
rinna_bilingual_4b_jpfull doesn't add additional signal beyond
rinna_jp_3_6b_jpfull.

### Why this works

The kaomoji emission distribution is largely cross-lingual — Japanese
and Western internet share ~most face shapes for affect cues. What
matters is having a model in a clean "responding in my native register"
state where the per-face conditional probability is well-formed. For
the 3.6B JP-only rinna, that requires native chat-frame + JP ask
+ JP-translated prompt body. Once in that state, even a tiny
(3.6B) JP-only model contributes information about kaomoji-quadrant
alignment that the larger English models miss.

Top-k=5 helps because v3 prompts vary in how clearly they elicit a
canonical kaomoji (some prompts are kaomoji-rich contexts, others
are kaomoji-sparse). Mean-over-20 dilutes the signal with prompts
that don't condition strongly on any face; top-k=5 keeps only the
prompts where the model actually had a preference. 5-of-20 is the
same per-quadrant sample count as the 5-prompt jpfull30 subset,
which is consistent with the noise-floor interpretation.

## Caveats

- Translations are claude-generated, not professionally translated.
  Translation errors could systematically bias particular quadrants;
  the consistent ~5pp drop from jpfull30 to jpfull120 is consistent
  with this but not conclusive.
- Best-k per encoder is in-sample optimization on Claude-GT. Honest
  cross-validation would split faces or use a held-out GT — not done
  here. The 70.6% number should be interpreted as an upper bound on
  what this lineup can achieve under k=5 rather than as a generalization
  estimate.
- Claude-GT floor=1 has many single-emit labels (38 of 60 modal_n=1).
  Floor=2 (22 faces) is more reliable; both floors agree on the
  ranking: 6-model top-k=5 > prior canonical.

## Followups

- Re-run `harness/66_per_project_quadrants.py` on the new
  6-model top-k=5 ensemble to refresh the per-project Claude analysis.
- Consider promoting `--summary-topk 5` to the default in script 50.
  Backward compat would break — the canonical
  `face_likelihood_<m>_summary.tsv` would shift quadrant predictions
  for faces near the decision boundary. Hold off until we re-verify
  on pooled GT and have a cross-validated rationale.
- The translation quality concern is fixable — a careful human pass
  over `emotional_prompts_jp.py` (or back-translation check) would
  rule out (or in) the systematic-translation-bias hypothesis.
