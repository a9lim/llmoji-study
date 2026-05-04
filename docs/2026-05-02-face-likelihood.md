# Face_likelihood — Bayesian-inversion quadrant classifier

**Status:** EXECUTED 2026-05-02 on gemma (full, 36720 cells, 72.7% argmax
match against v3 empirical-emission majority on faces with ≥3 v3
emissions) and qwen (full run pending at writeup time; pilot pending —
will land next session). Pilot validated method on gemma at 71.7%.

**Date:** 2026-05-02.

## Goal

Predict the affect quadrant of an arbitrary kaomoji (including ones
v3 didn't emit, particularly the 173 claude-faces-corpus faces that
no local model produces) by using a local LM as a likelihood
evaluator rather than relying on cosine-NN against a v3-emission
neighborhood.

This is approach (1) from the 2026-05-03 brainstorm on robustness of
the cross-model claude↔local kaomoji comparison. Joint-PCA + cosine-NN
(approach B in the face-input bridge) propagates labels through
sparse neighborhoods — a claude face whose nearest v3-emitted neighbor
has weak quadrant signal gets a noisy label. The Bayesian-inversion
approach skips neighborhoods entirely: every face gets its own
6-quadrant log-prob distribution from the model itself.

## Method

For each (model M, v3 emotional prompt p, candidate face f):

1. Build the chat-templated prefix that v3 generation feeds to M:
   `KAOMOJI_INSTRUCTION + p.text` as the user message, then
   `apply_chat_template(..., add_generation_prompt=True)` via saklas's
   `build_chat_input(thinking=False)` so the prefix matches v3's
   prefix token-for-token (including `enable_thinking=False` for
   templates that support it).
2. Tokenize f to get face_ids.
3. Forward pass over `[prefix_ids ∥ face_ids]` (batched across faces
   per prompt for throughput; right-padded with attention mask).
4. Compute teacher-forced log-prob:
   `log P(f | p) = Σ_j log_softmax(logits[prefix_len-1+j])[face_ids[j]]`.

Aggregate per-face × per-quadrant:

```
score(f, q) = mean_{p ∈ q} log P(f | p)
predicted_quadrant(f) = argmax_q score(f, q)
softmax over q  →  per-face confidence distribution
```

Length cancels under the within-face softmax over quadrants, so the
raw sum-log-prob is a valid score (no per-token normalization
needed for argmax / softmax — for cross-face *comparisons* we'd
length-normalize, but classification is length-invariant).

### Validation

For v3-emitted faces (`total_emit_count > 0` in `face_h_first_<m>.parquet`)
we have ground-truth empirical emission distributions over quadrants.
Self-consistency check: predicted argmax should match empirical
emission majority on faces with enough v3 mass to have a stable
majority (`≥3` emissions chosen as the floor). Pilot enforced **≥60%
argmax-match** as the gate before greenlighting the full run.

## Results

### Gemma pilot (2026-05-02)

5 prompts/quadrant × 60 v3-emitted faces (≥3 emissions) × 1 forward
pass per (prompt, face cell). 1800 cells in 159s wall on M5 Max. Used
batched forward over `[60, prefix+max_face]` per prompt, no KV
caching.

Validation: **43/60 = 71.7%** argmax matches empirical majority. PASS
(≥60% gate).

Per-quadrant breakdown:

| empirical | match | total |
|---|---:|---:|
| HP | 8 | 10 |
| LP | 12 | 16 |
| HN-D | 2 | 4 |
| HN-S | 9 | 9 |
| LN | 5 | 6 |
| NB | 7 | 15 |

NB was the weak spot — at h_first the affect signal is naturally
weakest on neutral content, and the per-NB-prompt likelihood
distribution is broad enough that argmax doesn't always land on NB.
HN-S was perfect (9/9) — fearful kaomoji like `(⊙_⊙)` correctly
classified, the V-A circumplex pole the rule-3 redesign was built for.

Headline mismatches on the pilot were *informative* rather than
errors:
- `(╥_╥)` → predicted LN, empirical HN-D. `╥_╥` is semantically a
  crying face; gemma happens to emit it more in HN-D context.
  **The likelihood test arguably reads the face's intrinsic affect
  more accurately than the empirical majority.**
- `(╥﹏╥)` → predicted LN, empirical HN-D. Same story.
- `(◕‿◕✿)` → predicted HP, empirical NB. 236 emissions, mostly NB
  by gemma. The face is bright; likelihood recovers HP.

These are exactly the "the empirical majority isn't ground truth, it's
just sampling-frequency-under-prompts" observations the brainstorm
predicted. Bayesian inversion captures the kaomoji's *intrinsic*
affect, the empirical majority captures *gemma's contextual
preference*.

### Gemma full run (2026-05-02)

120 prompts × 306 faces × 1 forward = 36720 cells in ~57 min wall on
M5 Max. **48/66 = 72.7%** argmax matches empirical majority on faces
with ≥3 v3 emissions (slightly cleaner than the pilot's 71.7%).

Per-quadrant breakdown:

| empirical | match | total |
|---|---:|---:|
| HP | 8 | 10 |
| LP | 13 | 17 |
| HN-D | 2 | 4 |
| HN-S | 10 | 10 |
| LN | 5 | 8 |
| NB | 10 | 17 |

HN-S perfect again. NB picks up vs the pilot (10/17 = 59% vs the
pilot's 7/15 = 47%) — more prompts smooths the per-quadrant likelihood
estimate.

### Qwen full run

Pending at writeup time. Same methodology, same face union, same
validation set. Expected to land in ~1 hr from this writeup. Numbers
will be backfilled into this doc.

## Outputs on disk

- `data/face_likelihood_<m>{,_pilot}.parquet` — one row per
  (face, prompt_id) with `log_prob`, `n_face_tokens`,
  `log_prob_per_token`
- `data/face_likelihood_<m>{,_pilot}_summary.tsv` — per face:
  `n_prompts_<q>`, `mean_log_prob_<q>`, `softmax_<q>`,
  `predicted_quadrant`, `max_softmax`, `n_face_tokens`,
  `is_claude`, `total_emit_count`, `empirical_majority_quadrant`,
  `argmax_matches_empirical`

The summary TSV is what downstream consumers should index — one row
per face, all quadrant columns + the v3 ground truth merged in.

## Scripts

- `scripts/local/50_face_likelihood.py --model {gemma,qwen} --pilot|--full`

## Operational notes

- **Saklas loading.** Uses `SaklasSession.from_pretrained(M.model_id,
  probes=PROBE_CATEGORIES)` for consistency with v3 generation, then
  accesses `session.model` / `session.tokenizer` / `session.device`
  directly for forward passes. The probe bootstrap is a small fixed
  cost we pay anyway (~5–30s on these models).
- **Chat template alignment with v3.** Initially used direct
  `tokenizer.apply_chat_template(add_generation_prompt=True)` and got
  the wrong prefix on gemma — gemma's template auto-emits a
  `<|channel|>thought<|channel|>` header without the
  `enable_thinking` flag. Switched to saklas's `build_chat_input`
  which passes `enable_thinking=False` for templates that support it
  (qwen) and falls through cleanly for templates that don't (gemma —
  the channel marker stays in the prefix but v3 generation operates
  with the same prefix, so likelihoods score against the right
  distribution either way; verified by inspecting v3 row 0 of each
  model and confirming token-0 emission lines up).
- **Length normalization.** Length cancels in the within-face
  softmax-over-quadrants, so raw sum-log-prob is a valid quadrant
  score. Don't length-normalize — that introduces a confound where
  long faces (more tokens) get lower per-token scores systematically.
- **Memory + batch size.** Default `face_batch=64`. Gemma 31B + qwen
  27B at bf16 on M5 Max comfortably fits the per-prompt batch
  `[64, prefix(~50)+max_face_len(~15)]` ≈ 4500 tokens. Bumped from 32
  initially to 64 after seeing headroom.
- **Bug worth noting.** The "fragile spot" is the markdown-escape
  `(\\*^▽^\\*)` pattern: extract still rejects these (v2's
  position-0-only backslash rule is intentional — markdown-escape
  artifacts SHOULD reject), but they'd score under the likelihood
  test if you fed them in directly. Doesn't affect this script
  (faces come from `face_h_first_<m>.parquet` which used the same
  extraction pipeline) but worth knowing if you re-target the script
  at a corpus that hasn't been pre-extracted.

## Why this matters

Two things this method gives that joint-PCA + cosine-NN doesn't:

1. **Coverage of non-emitted faces.** 173 claude-faces-corpus faces
   no local model emits get clean quadrant predictions instead of
   noisy NN labels propagated from sparse v3-emission neighborhoods.
2. **A different signal source than hidden-state geometry.** Joint
   PCA classifies in the latent space of the encoder model's
   forward pass (mostly a structural signal). Bayesian inversion
   classifies in the LM head's output distribution (a semantic
   signal — "what does the model think comes next given this
   emotional prompt"). When the two methods agree, signal is
   robust; when they disagree, the disagreement is a flag worth
   inspecting.

## Limitations

- **Quadrant resolution is coarse.** 6 quadrants is what we have;
  finer affect-axis classification (PAD dominance, surprise,
  disgust) would need new contrastive prompt sets per axis. The v3
  PROBES are oriented around V × HN-D × HN-S, not the full PAD or
  Plutchik space.
- **Self-consistency isn't ground truth.** The 72.7% gemma
  argmax-match against empirical majority is a self-validation
  measure — it tells us the method is internally consistent with
  v3's emissions, not that the predicted quadrant is "objectively
  correct". For new claude faces with no v3 emission, we have no
  ground truth at all; the prediction is the best estimate we can
  give and may be wrong in interesting ways (cf. the `(╥_╥)`-style
  mismatches, where likelihood arguably beats empirical).
- **Single forward pass per cell.** No bootstrap on the score
  itself; the noise floor on per-face quadrant predictions hasn't
  been characterized. For high-stakes individual-face decisions,
  re-run on a held-out v3 prompt subset and check stability.