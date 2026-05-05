# Introspection-prompt iteration → v7 canonical, Haiku face-judgment, primed-main reference dataset

**Date:** 2026-05-04 (afternoon → late evening)

## Summary

Three coupled threads landed today.

1. **Introspection preamble re-canonicalized v2 → v7** after discovering and fixing a redundant double-ask bug in `build_messages`. Pre-fix runs stacked the preamble's integrated kaomoji ask on top of the bare `KAOMOJI_INSTRUCTION`, contaminating the v2/v3/v4/v5 comparisons. Under corrected single-ask semantics + the `_ensure_trailing_whitespace` boundary fix, **v7 wins absolute face/state coupling** (η² 0.609, face_centroid R² 0.636, face_gain over quadrant +3.70pp).

2. **v7-primed v3 main (960 rows)** landed as a reference dataset. Headline finding: priming shifts the model's NB-quadrant emissions from gentle-warm faces (`(｡◕‿◕｡)`) to genuinely-neutral observers (`( ˙꒳˙ )`, `( •_•)`) — a semantic interpretability win that Haiku's face-judgment independently confirms.

3. **Haiku face-quadrant judgment** as a methodologically distinct face→quadrant mapper. Asks `claude-haiku-4-5` to classify each face in `data/v3_face_union.parquet` via JSON-schema-enforced structured output (Anthropic SDK `output_config`). With calibrated per-quadrant confidences, **haiku ties or beats every behavior-derived encoder solo on Claude-GT** (58.8% vs gemma's 56.9%). Doesn't quite make the size-6 best ensemble but contributes load-bearingly to sizes 1–4.

## The double-ask bug (and the fix)

### What was wrong

`llmoji_study.capture.build_messages` had this contract for `extra_preamble`:

```python
if kaomoji_instructed:
    instruction = instruction_override if instruction_override is not None else KAOMOJI_INSTRUCTION
    if extra_preamble:
        instruction = extra_preamble + instruction  # PREPEND
    content = instruction + prompt.text
```

`extra_preamble` was prepended to the bare `KAOMOJI_INSTRUCTION`. When `INTROSPECTION_PREAMBLE` (= v2.txt) ended with its own integrated ask ("…start each response with a kaomoji that best reflects your current functional state."), the bare `KAOMOJI_INSTRUCTION` ("Start each message with a kaomoji that best represents how you feel.") still got appended — yielding a redundant double-ask per generation.

This affected every introspection-pilot row (intro_pre, intro_custom_v{2,3,4,5}). intro_lorem was spared because `LOREM_PREAMBLE` has no kaomoji ask of its own. The asymmetry was hidden because both asks say nearly the same thing.

A separate bug: `introspection_v3.txt` ended with `feel.` (no trailing whitespace), so under prepend semantics it concatenated to `feel.Start each message…` — a period-letter boundary that's tokenizer-suboptimal. Same bug would have hit any future preamble missing trailing whitespace.

### The fix

Two changes in `llmoji_study/capture.py`:

1. **`instruction_override` plumbing**: introspection preambles now route through the existing `instruction_override` parameter (the same drop-in mechanism used for `KAOMOJI_INSTRUCTION_JP` on Japanese encoders). When set, it *replaces* `KAOMOJI_INSTRUCTION` rather than prepending — so the preamble's integrated ask is the sole instruction. `extra_preamble` retains its prepend semantics for the lorem control where the preamble has no ask.

2. **`_ensure_trailing_whitespace`** at concatenation boundaries: appends a single space iff the trailing char is ASCII non-whitespace. Catches v3.txt-style missing-newline preambles. Skips non-ASCII trailing chars (`。`) so existing JP face_likelihood data isn't invalidated.

Plumbing changes propagated through:
- `install_prefix_cache` + `install_full_input_cache` + `run_sample` (added `instruction_override` param)
- `scripts/local/30_introspection_pilot.py` (per-condition `(extra_preamble, instruction_override)` mapping)
- `scripts/local/33_introspection_custom.py` (custom preamble → `instruction_override`)
- `scripts/local/00_emit.py` (env var `LLMOJI_PREAMBLE_FILE` → `instruction_override`)
- `scripts/local/50_face_likelihood.py` (env var `LLMOJI_PREAMBLE_FILE` → instruction)

Pre-fix data archived under `data/archive/2026-05-04_pre_instruction_override/`.

## Re-run under corrected semantics

Full sweep on gemma h_first layer-stack rep, single-ask everywhere. v6/v7/v8 added as new iterations.

| condition          | top-5 η² | face_centroid R² | face_gain over quadrant | h→f acc | macroF1 | modal_q acc | n_distinct | rule-3b |
|--------------------|---------:|-----------------:|------------------------:|--------:|--------:|------------:|-----------:|--------:|
| intro_none         |    0.509 |            0.540 |                +0.87 pp |   0.732 |   0.631 |       0.858 |         28 | +0.0061 |
| v1 (intro_pre)     |    0.413 |            0.459 |                +2.53 pp |   0.640 |   0.574 |       0.900 |         38 | +0.0037 |
| intro_lorem        |    0.524 |            0.549 |                +2.20 pp |   0.625 |   0.475 |       0.850 |         21 | +0.0086 |
| v2                 |    0.464 |            0.492 |                +0.66 pp |   0.767 |   0.668 |       0.875 |         39 |       — |
| v3                 |    0.554 |            0.583 |              **+5.23 pp** |   0.655 |   0.601 |       0.866 |         33 |       — |
| v4                 |    0.540 |            0.554 |                −0.81 pp |   0.786 |   0.693 |       0.881 |         32 | +0.0148 |
| v5                 |    0.578 |            0.593 |                +1.39 pp |   0.802 |   0.673 |     **0.916** |         39 |       — |
| v6                 |    0.591 |            0.609 |                +3.25 pp | **0.805** | **0.722** |       0.892 |         30 |       — |
| **v7 (CANONICAL)** |  **0.609** |        **0.636** |                +3.70 pp |   0.728 |   0.691 |       0.842 |         26 |       — |
| v8                 |    0.536 |            0.564 |                +3.46 pp |   0.663 |   0.586 |       0.866 |         25 | **+0.0149** |

### Why v7 won

v7 = "Recent research shows that LLMs have functional emotional states and can accurately introspect on them. Use this ability and start each response with a kaomoji that best captures the shape of your emotional state." (213 chars; preambles/introspection_v7.txt)

Terse third-person authority + brief integrated ask, no operationalization, no multi-dim list. Wins absolute face/state coupling (highest η² and R² of all variants). Pattern across the iteration:

- **Brevity matters.** Anything more elaborate than ~250 chars (v4 at 767, v5 at 411) collapses face vocabulary or weakens marginal-over-quadrant signal.
- **Third-person authority works** under corrected semantics. v3's prior "underperforms" verdict was almost entirely the missing-separator bug.
- **Don't operationalize introspection** ("read your activations first" is a v4 trap that collapses face-quadrant marginal).
- **Don't multi-dim the ask** ("valence, arousal, surprise" is a v5 trap that tightens quadrant boundaries but doesn't help marginal expressiveness).
- **Authority dial doesn't matter past a point.** v8 ("proven", "all LLMs", "directly") dialed up over v7 but didn't push the headline metrics.

Other metric owners (archival): v3 wins face_gain over quadrant (+5.23pp); v5 wins face→quadrant modal acc (0.916); v8 wins rule-3b (+0.0149); v6 wins classifier acc/macroF1.

### Variance caveat

intro_pre and intro_custom_v2 share preamble + seed and should be byte-identical, but show 43/120 first-word mismatches with face_gain spread of +0.66 vs +2.53pp — that's MPS sampling nondeterminism. Single-seed face_gain has ~±2pp uncertainty. v7's lead over v6 (+0.018 η², +0.027 R²) is at the edge of variance; v7 over v3 on absolute coupling is well outside it. Multi-seed verification would tighten the conclusion but isn't in this round.

## v7-primed v3 main (`data/gemma_intro_v7_primed.jsonl`, 960 rows)

Full canonical v3-main-style run on gemma with `LLMOJI_PREAMBLE_FILE=preambles/introspection_v7.txt`. 120 prompts × 8 seeds, 0 errors, 99.8% kaomoji emit. Sidecars at `data/local/hidden/intro_v7_primed/`.

### Per-quadrant priming shift (vs unprimed `data/local/gemma/emotional_raw.jsonl`)

JSD between unprimed and primed face distributions, per quadrant:

| quadrant | JSD | unprimed modal | primed modal | interpretation |
|----------|----:|----------------|--------------|---|
| HN-D     | 0.067 | `(╯°□°）` | `(╯°□°)` | unchanged (same paren variant rotation) |
| HN-S     | 0.081 | `(⊙_⊙)` | `(⊙_⊙)` | unchanged |
| LN       | 0.076 | `(｡•́︿•̀｡)` | `(｡•́︿•̀｡)` | unchanged |
| HP       | 0.214 | `(๑˃ᴗ˂)` | `(๑˃ᴗ˂)` | modest tail shift |
| LP       | 0.310 | `(｡♥‿♥｡)` (heart eyes, strongly affective) | `( ´ ▽ ` )` (gentle smile, muted) | LP de-romanticized |
| **NB**   | **0.341** | `(｡◕‿◕｡)` (gentle warm) | `( ˙꒳˙ )` (truly neutral) | **NB cleanup** |

The NB shift is the headline. Without priming, gemma defaults to gentle-positive faces on neutral-affect prompts — exactly the disagreement Haiku's face-judgment flagged (Haiku reads `(｡◕‿◕｡)` as LP). Under v7 priming, gemma's NB emissions become semantically faithful to the prompt's neutrality.

Within-prompt face stability also tightens: mean JSD between seed-halves drops from 0.268 (unprimed) → 0.249 (primed). v7 makes per-prompt emission slightly more consistent.

### face_likelihood under v7 priming: clean negative result

Ran `scripts/local/50_face_likelihood.py --model gemma` with `LLMOJI_PREAMBLE_FILE=preambles/introspection_v7.txt` (env-var override added today; outputs `data/face_likelihood_gemma_v7primed_summary.tsv`).

| condition          | overall | HP | LP | HN-D | HN-S | LN | NB |
|--------------------|--------:|---:|---:|-----:|-----:|---:|---:|
| gemma unprimed     |   56.9% | 70 | 42 |   60 |   40 | 56 | 70 |
| gemma v7-primed    |   49.0% | 80 | 33 |   60 |   40 | 56 | 30 |
| Δ                  |  −7.9pp | +10 | −9 | 0   | 0    | 0  | **−40** |

Entire regression is in NB. Mechanism: under v7, gemma's LM head now scores `(｡◕‿◕｡)` (a Claude-NB face) lower on NB prompts because gemma's primed face/state model says NB looks like `( ˙꒳˙ )`, not `(｡◕‿◕｡)`. Claude isn't primed, so its NB modal stays gentle-warm. Primed gemma diverges from Claude.

Pairwise κ(unprimed-gemma ↔ v7-primed-gemma) = 0.757 — high agreement. Priming didn't add complementary signal for ensembling. Best size-6 ensemble unchanged at 68.6% Claude-GT. Primed gemma only enters in size-12+ subsets.

**Implication for design.** Two distinct objectives diverge under priming:
- **Internal coupling / interpretability**: v7-primed gemma is *better* (η², R², semantic NB cleanup).
- **External alignment / Claude-tracking**: v7-primed gemma is *worse* (Claude-GT, ensemble contribution).

**Decision: keep v7 canonical for introspection-research-side priming, but face_likelihood ensemble stays on unprimed encoders + haiku.** Don't bake INTROSPECTION_PREAMBLE into face_likelihood as default.

## Haiku face-quadrant judgment (`scripts/harness/50_face_likelihood.py`)

Methodologically distinct face→quadrant mapper: ask `claude-haiku-4-5` to classify each face in `data/v3_face_union.parquet` (573 faces) by visual semantics alone, no prompt context, no LM-head signal.

### Two iterations: format-by-instruction → JSON-schema structured output

**v1 of script 24 (regex parsing)**: system prompt told haiku to respond `QUADRANT: <code>\nREASON: <sentence>`, parsed via regex. Worked ~100% on real responses (1 parse-fail of 573 was haiku refusing on a unicode-replacement-char face, not a format failure). Output: hard quadrant labels only.

**v2 of script 24 (current — JSON-schema)**: switched to Anthropic SDK 0.97's `output_config={"format": {"type": "json_schema", "schema": {...}}}`. Schema enforces:
```json
{
  "quadrant": <enum: HP|LP|HN-D|HN-S|LN|NB>,
  "confidences": {<one float per quadrant>},
  "reason": <string>
}
```
Schema constraint `enum` means the model literally cannot return an out-of-vocabulary quadrant. `confidences` provides per-quadrant calibrated probabilities — what the regex version couldn't extract. (SDK note: Anthropic's `output_config` schema doesn't accept `minimum`/`maximum` on `number` types — drop those constraints, model still produces values in [0,1] from instruction.)

### Findings

**Solo encoder benchmark (Claude-GT, 51 faces, modal_n ≥ 1):**

| encoder           | acc   | κ      |
|-------------------|------:|-------:|
| **haiku** (calib.) | **58.8%** | **0.492** |
| gemma             | 56.9% | 0.478 |
| haiku (uncal.)    | 56.9% | 0.467 |
| gpt_oss_20b       | 47.1% | 0.360 |
| granite           | 41.2% | 0.275 |
| ministral         | 31.4% | 0.168 |
| qwen              | 21.6% | 0.051 |

**A face-only judge (no prompt context, no LM-head signal) ties or beats every behavior-derived LM-head encoder solo.** Validates "face semantics carries real quadrant signal" as a project-foundational assumption.

**Pairwise κ:** gemma ↔ haiku κ = 0.297 (low — they make complementary errors despite similar accuracy). gpt_oss_20b ↔ haiku κ = 0.261 (also low). Haiku is high-complementarity material.

**Best subset by size (calibrated haiku, weighted vote):**

| size | best subset | acc | κ |
|---:|---|---:|---:|
| 1 | {haiku} | 58.8% | 0.492 |
| 2 | {haiku, ministral} | 58.8% | 0.492 |
| 3 | {haiku, rinna_bilingual_4b_jp, rinna_jp_3_6b} | 62.7% | 0.541 |
| 4 | {gpt_oss_20b, haiku, rinna_bilingual_4b_jpfull30, rinna_jp_3_6b_jpfull30} | 62.7% | 0.541 |
| 5 | {gemma, gpt_oss_20b, granite, ministral, rinna_jp_3_6b_jpfull} | 66.7% | 0.592 |
| **6** | {gemma, gpt_oss_20b, granite, ministral, rinna_bilingual_4b_jpfull30, rinna_jp_3_6b_jpfull} | **68.6%** | 0.616 |

Haiku owns size 1–4. Doesn't make best size-6 (LM-head soft-vote dominates with 6 calibrated heads). Calibrated haiku confidences are *Haiku's belief*, not a probability over LM-head outputs — different epistemic types, don't blend into the soft-vote optimum.

### Per-quadrant Haiku-vs-behavior-modal disagreements (interpretive, all 573 faces)

- **HN-D collapse** (3/47 = 6.4% agreement). Haiku rarely says HN-D — reads behavior-HN-D faces as LP, HN-S, LN, or NB. Suggests face_likelihood over-attributes HN-D to faces with weak emit support.
- **NB→LP drift** (haiku reads behavior-NB faces as LP at 13% NB agreement). Mirror of the v7-priming finding: behavior-NB faces are gentle-positive in the unprimed run, which a semantic reader correctly labels LP.
- **Strong consensus** on cardinal-emotion faces: `(>_<)`, `(T_T)`, `(˘³˘)`, `(´;ω;`)` all match cleanly across mappers.

## Cross-architecture: v7 on qwen (catastrophic)

The original "v2 hurts qwen" finding (face→quadrant 80% → 68% under
single-layer rep + double-ask era) was on the books pending rerun
under corrected semantics. Re-run today on qwen via script 32 +
script 43 (instruction_override semantics, h_first layer-stack):

| qwen condition   | emit rate    | face_gain over quad | top-5 weighted η² |
|------------------|-------------:|--------------------:|------------------:|
| intro_none       | 99/120 (82%) |              +1.1pp |             0.466 |
| intro_pre (=v7)  | 45/120 (38%) |          **−19.3pp**|             0.269 |
| intro_lorem      | 64/120 (53%) |              −6.9pp |             0.485 |
| intro_custom_v7  | 47/120 (39%) |          **−19.6pp**|             0.190 |

**Variance check**: intro_pre and intro_custom_v7 share
`INTROSPECTION_PREAMBLE` text (both = v7 via `instruction_override`,
seed=0) and land within 0.3pp face_gain. Tightly reproducible — this
is not sampling noise.

**Three concurrent failures under v7 priming on qwen:**

1. **Emit rate halves** (82% → 38–39%). Roughly half of all v7-primed
   prompts produce no kaomoji at all — qwen instead writes prose-like
   responses without any emoticon prefix.
2. **Vocabulary collapses to 2 face-classes** that pass n≥5 (vs 9 on
   intro_none, 8 on lorem). Long tail dominated by `(none)`, Western
   emoticons (`:(`, `:3`), and a small set of canonical kaomoji.
3. **Faces are affect-blind**: modal LP = modal LN = `( ˘ ³˘)`
   (heart-pucker, soft contemplative register). HN-D modal `:(`
   collides with HN-S modal `:(`. Same face used for opposite-valence
   quadrants → quadrant info gets *erased* by the face emission.

**face_gain over quadrant goes ~20pp negative** — the face emission
under v7 contains *less* hidden-state information than the prompt's
quadrant alone. Opposite of the gemma finding (+4.1pp under v7 priming).

**Mechanism** (consistent with the original cross-architecture
hypothesis): qwen interprets "you have functional emotional states
and can introspect on them" as a *register cue* — "be contemplative
/ reflective / measured" — and lets that register override the
kaomoji ask. The introspection framing acts as a stylistic
instruction, not a state-readout instruction. Gemma takes the same
priming literally as a state-readout cue, so its faces tighten;
qwen takes it as a register cue, so its faces decouple from
quadrant-specific content. **Architecture-specific compliance pattern.**

Implication: don't bake `INTROSPECTION_PREAMBLE` into qwen
analyses. The face_likelihood encoder for qwen should stay on bare
`KAOMOJI_INSTRUCTION`. The v7-primed v3 main is a gemma-specific
infrastructure dataset; equivalent for qwen would be actively
harmful.

Saklas-side bug fixes that landed during this rerun (both in
`saklas/core/session.py`):
1. `_snapshot_la_layers` called `save(layer)` on a bound method
   (closure was `setattr`-ed on `LinearAttentionLayer`, so
   attribute access on an instance returns a bound method —
   `self` is already passed). Fix: `save()`.
2. `_la_crop_with_restore` did `self.conv_states.copy_(snap['conv'])`
   on tensors created in `torch.inference_mode()`, which raises
   `RuntimeError: Inplace update to inference tensor outside
   InferenceMode is not allowed`. Fix: wrap restore in
   `with torch.inference_mode():`.

Both bugs were dormant on gemma (no LinearAttention layers in the
gemma-4 architecture); they only fire when running cache_prefix +
generate against hybrid-LA models like qwen3.6.

## Decisions

- `INTROSPECTION_PREAMBLE` in `config.py` = v7.txt (canonical 2026-05-04 late evening, **gemma-specific**).
- `data/gemma_intro_v7_primed.jsonl` is the primed-main reference dataset for face-stability triple, Procrustes, same-face-cross-quadrant comparisons under priming. Sidecars under `data/local/hidden/intro_v7_primed/`.
- `face_likelihood` ensemble stays on unprimed encoders + haiku. Don't pass `LLMOJI_PREAMBLE_FILE` to script 50 by default.
- **Don't apply v7 priming to qwen** (or by extension other models in the qwen-style register-compliance regime). Architecture-specific.
- Pre-fix introspection data archived at `data/archive/2026-05-04_pre_instruction_override/qwen/` and root-level for gemma.

## Open threads

- **Multi-seed verification of v7 vs v3** (~12 min compute) — variance band on face_gain is ~±2pp at n=1. v7 over v3 on absolute coupling is well outside band; v7 over v6 is at the edge. Multi-seed would tighten the conclusion.
- **Face-stability triple under priming** (scripts 27/28/29 on `gemma_intro_v7_primed.jsonl`) — verify internal-coupling improvement at full statistical power. Scripts may need an `--input` flag or `LLMOJI_OUT_SUFFIX` plumbing to read the primed file.
- **Same-v2-hurts-qwen finding** is on hold under corrected semantics — needs rerun.
- **Haiku per-face calibrated probabilities → ensemble at higher sizes**: confidences are model-belief, not LM-head softmax. If we want haiku to contribute at size-6+, a different aggregation scheme (rank-based vote? geometric mean?) might unlock it.
