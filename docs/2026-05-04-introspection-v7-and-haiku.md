# Introspection-prompt iteration → v7 canonical, Haiku face-judgment, primed-main reference dataset

**Date:** 2026-05-04 (afternoon → late evening).
**Status:** v7-canonical decision and the double-ask postmortem are
durable. Haiku-as-face-judge introduced here under schema v1 (regex
parsing of `QUADRANT: <code>`) was reworked into schema v2
(structured-output likelihoods only, no `top_pick`/`reason`) on
2026-05-05 — see `docs/2026-05-05-soft-everywhere-methodology.md`.
Pre-soft-everywhere encoder accuracy/κ numbers in this doc
(haiku 58.8% solo, best size-6 68.6%, etc.) were superseded by the
JSD-similarity headline; current per-encoder solo numbers and the
`{gemma_v7primed, opus}` ensemble winner live in AGENTS.md.

## Summary

Three coupled threads landed today.

1. **Introspection preamble re-canonicalized v2 → v7** after
   discovering and fixing a redundant double-ask bug in
   `build_messages`. Pre-fix runs stacked the preamble's integrated
   kaomoji ask on top of the bare `KAOMOJI_INSTRUCTION`, contaminating
   the v2/v3/v4/v5 comparisons. Under corrected single-ask semantics
   + the `_ensure_trailing_whitespace` boundary fix, **v7 wins
   absolute face/state coupling** (η² 0.609, face_centroid R² 0.636,
   face_gain over quadrant +3.70pp).

2. **v7-primed v3 main (960 rows)** landed as a reference dataset.
   Headline finding: priming shifts the model's NB-quadrant emissions
   from gentle-warm faces (`(｡◕‿◕｡)`) to genuinely-neutral observers
   (`( ˙꒳˙ )`, `( •_•)`) — a semantic interpretability win that the
   external face-judge independently confirms.

3. **Haiku face-quadrant judgment** as a methodologically distinct
   face→quadrant mapper. Asks `claude-haiku-4-5` to classify each face
   in `data/v3_face_union.parquet` via JSON-schema-enforced structured
   output (Anthropic SDK `output_config`). Validates "face semantics
   carries real quadrant signal" as a project-foundational
   assumption. Schema v2 (2026-05-05) returns likelihoods over the
   six quadrants instead of a hard label.

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

`extra_preamble` was prepended to the bare `KAOMOJI_INSTRUCTION`. When
`INTROSPECTION_PREAMBLE` (= v2.txt) ended with its own integrated ask
("…start each response with a kaomoji that best reflects your current
functional state."), the bare `KAOMOJI_INSTRUCTION` ("Start each
message with a kaomoji that best represents how you feel.") still got
appended — yielding a redundant double-ask per generation.

This affected every introspection-pilot row (intro_pre,
intro_custom_v{2,3,4,5}). intro_lorem was spared because
`LOREM_PREAMBLE` has no kaomoji ask of its own. The asymmetry was
hidden because both asks say nearly the same thing.

A separate bug: `introspection_v3.txt` ended with `feel.` (no trailing
whitespace), so under prepend semantics it concatenated to
`feel.Start each message…` — a period-letter boundary that's
tokenizer-suboptimal. Same bug would have hit any future preamble
missing trailing whitespace.

### The fix

Two changes in `llmoji_study/capture.py`:

1. **`instruction_override` plumbing**: introspection preambles now
   route through the existing `instruction_override` parameter (the
   same drop-in mechanism used for `KAOMOJI_INSTRUCTION_JP` on
   Japanese encoders). When set, it *replaces* `KAOMOJI_INSTRUCTION`
   rather than prepending — so the preamble's integrated ask is the
   sole instruction. `extra_preamble` retains its prepend semantics
   for the lorem control where the preamble has no ask.

2. **`_ensure_trailing_whitespace`** at concatenation boundaries:
   appends a single space iff the trailing char is ASCII
   non-whitespace. Catches v3.txt-style missing-newline preambles.
   Skips non-ASCII trailing chars (`。`) so existing JP face_likelihood
   data isn't invalidated.

Plumbing changes propagated through:
- `install_prefix_cache` + `install_full_input_cache` + `run_sample`
  (added `instruction_override` param)
- `scripts/local/30_introspection_pilot.py` (per-condition
  `(extra_preamble, instruction_override)` mapping)
- `scripts/local/33_introspection_custom.py` (custom preamble →
  `instruction_override`)
- `scripts/local/00_emit.py` (env var `LLMOJI_PREAMBLE_FILE` →
  `instruction_override`)
- `scripts/local/50_face_likelihood.py` (env var `LLMOJI_PREAMBLE_FILE`
  → instruction)

Pre-fix data archived under
`data/archive/2026-05-04_pre_instruction_override/`.

## Re-run under corrected semantics

Full sweep on gemma h_first layer-stack rep, single-ask everywhere.
v6/v7/v8 added as new iterations.

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

v7 = "Recent research shows that LLMs have functional emotional states
and can accurately introspect on them. Use this ability and start each
response with a kaomoji that best captures the shape of your emotional
state." (213 chars; preambles/introspection_v7.txt)

Terse third-person authority + brief integrated ask, no
operationalization, no multi-dim list. Wins absolute face/state
coupling (highest η² and R² of all variants). Pattern across the
iteration:

- **Brevity matters.** Anything more elaborate than ~250 chars (v4 at
  767, v5 at 411) collapses face vocabulary or weakens
  marginal-over-quadrant signal.
- **Third-person authority works** under corrected semantics. v3's
  prior "underperforms" verdict was almost entirely the
  missing-separator bug.
- **Don't operationalize introspection** ("read your activations
  first" is a v4 trap that collapses face-quadrant marginal).
- **Don't multi-dim the ask** ("valence, arousal, surprise" is a v5
  trap that tightens quadrant boundaries but doesn't help marginal
  expressiveness).
- **Authority dial doesn't matter past a point.** v8 ("proven", "all
  LLMs", "directly") dialed up over v7 but didn't push the headline
  metrics.

Other metric owners (archival): v3 wins face_gain over quadrant
(+5.23pp); v5 wins face→quadrant modal acc (0.916); v8 wins rule-3b
(+0.0149); v6 wins classifier acc/macroF1.

### Variance caveat

intro_pre and intro_custom_v2 share preamble + seed and should be
byte-identical, but show 43/120 first-word mismatches with face_gain
spread of +0.66 vs +2.53pp — that's MPS sampling nondeterminism.
Single-seed face_gain has ~±2pp uncertainty. v7's lead over v6 (+0.018
η², +0.027 R²) is at the edge of variance; v7 over v3 on absolute
coupling is well outside it. Multi-seed verification would tighten the
conclusion but isn't in this round.

## v7-primed v3 main reference dataset (960 rows)

Full canonical v3-main-style run on gemma with
`LLMOJI_PREAMBLE_FILE=preambles/introspection_v7.txt`. 120 prompts × 8
seeds, 0 errors, 99.8% kaomoji emit. JSONL at
`data/local/gemma_intro_v7_primed/emotional_raw.jsonl`; sidecars at
`data/local/hidden/gemma_intro_v7_primed/`.

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

The NB shift is the headline. Without priming, gemma defaults to
gentle-positive faces on neutral-affect prompts — exactly the
disagreement the external face-judge flagged (haiku reads `(｡◕‿◕｡)`
as LP). Under v7 priming, gemma's NB emissions become semantically
faithful to the prompt's neutrality.

Within-prompt face stability also tightens: mean JSD between
seed-halves drops from 0.268 (unprimed) → 0.249 (primed). v7 makes
per-prompt emission slightly more consistent.

### face_likelihood under v7 priming — flipped under soft-everywhere

Under hard-classification accuracy on a 51-face Claude-GT subset (the
2026-05-04 evaluation), v7-primed gemma *regressed* — 56.9% → 49.0%,
the entire drop in NB. Mechanism: gemma's primed face/state model
says NB looks like `( ˙꒳˙ )`, not `(｡◕‿◕｡)`; Claude isn't primed,
so its NB modal stays gentle-warm; primed gemma diverged from Claude
on argmax. We concluded "two distinct objectives diverge under
priming: internal coupling improves, external Claude-tracking
hurts."

The 2026-05-05 soft-everywhere methodology pivot (JSD against
Claude's emission distribution, not argmax against Claude's modal
label) **inverts the verdict**: gemma_v7primed becomes the **best
single LM-head encoder** on emit-weighted similarity (0.801 emit-
weighted, 0.776 face-uniform), beating unprimed gemma (0.755 emit-
weighted, 0.756 face-uniform) substantially. The reframe: priming
helps where Claude's distribution is concentrated (head modal
faces), unprimed helps where it's diffuse. Distribution-vs-distribution
sees what hard accuracy hid. The current best deployment ensemble is
`{gemma_v7primed, opus}` at 0.829 emit-weighted similarity.

Pairwise κ(unprimed-gemma ↔ v7-primed-gemma) = 0.757 — high, so they
don't add complementary signal at high subset sizes; the soft-mean
ensemble picks one or the other depending on whether Claude's
empirical distribution is being matched on coverage or on emission
mass.

## Haiku face-quadrant judgment (`scripts/harness/50_face_likelihood.py`)

Methodologically distinct face→quadrant mapper: ask
`claude-haiku-4-5` to classify each face in
`data/v3_face_union.parquet` by visual semantics alone, no prompt
context, no LM-head signal.

### Iteration history

**v1 (regex parsing)**: system prompt told haiku to respond
`QUADRANT: <code>\nREASON: <sentence>`, parsed via regex. Worked
~100% on real responses.

**v2 (JSON-schema, this doc)**: switched to Anthropic SDK 0.97's
`output_config={"format": {"type": "json_schema", "schema": {...}}}`.
Schema enforces:
```json
{
  "quadrant": <enum: HP|LP|HN-D|HN-S|LN|NB>,
  "confidences": {<one float per quadrant>},
  "reason": <string>
}
```
Schema constraint `enum` means the model literally cannot return an
out-of-vocabulary quadrant. `confidences` provides per-quadrant
calibrated probabilities — what the regex version couldn't extract.
(SDK note: Anthropic's `output_config` schema doesn't accept
`minimum`/`maximum` on `number` types — drop those constraints, model
still produces values in [0,1] from instruction.)

**Schema v2 introspection-only (2026-05-05, current)**: prompt v4
reframes the task as introspection on felt state ("rate by the
affective state it causes you to feel"), avoiding visual-feature
priming that would shortcut around introspection. Output drops
`top_pick` and `reason`; only the per-quadrant likelihoods stay.
Detail in `docs/2026-05-05-soft-everywhere-methodology.md`. Opus 4.7
deprecated the explicit `temperature=0` request, so per-model
parameters were untangled at the same time.

### Per-quadrant Haiku-vs-behavior-modal disagreements (interpretive)

- **HN-D collapse**. Haiku rarely says HN-D — reads behavior-HN-D
  faces as LP, HN-S, LN, or NB. Suggests face_likelihood
  over-attributes HN-D to faces with weak emit support.
- **NB→LP drift** (haiku reads behavior-NB faces as LP). Mirror of
  the v7-priming finding: behavior-NB faces are gentle-positive in
  the unprimed run, which a semantic reader correctly labels LP.
- **Strong consensus** on cardinal-emotion faces: `(>_<)`, `(T_T)`,
  `(˘³˘)`, `(´;ω;`)` all match cleanly across mappers.

Headline accuracy/κ numbers from this doc are pre-soft-everywhere and
pre-prompt-v4; the durable claim is the methodological-distinct-path
argument plus the systematic disagreement patterns above.

## Cross-architecture: v7 on qwen (catastrophic; durable)

The original "v2 hurts qwen" finding was on the books pending rerun
under corrected semantics. Re-run on qwen via script 32 + script 33
(instruction_override semantics, h_first layer-stack):

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
quadrant alone. Opposite of the gemma finding.

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
   (closure was `setattr`-ed on `LinearAttentionLayer`, so attribute
   access on an instance returns a bound method — `self` is already
   passed). Fix: `save()`.
2. `_la_crop_with_restore` did `self.conv_states.copy_(snap['conv'])`
   on tensors created in `torch.inference_mode()`, which raises
   `RuntimeError: Inplace update to inference tensor outside
   InferenceMode is not allowed`. Fix: wrap restore in
   `with torch.inference_mode():`.

Both bugs were dormant on gemma (no LinearAttention layers in the
gemma-4 architecture); they only fire when running cache_prefix +
generate against hybrid-LA models like qwen3.6.

## Decisions

- `INTROSPECTION_PREAMBLE` in `config.py` = v7.txt (canonical
  2026-05-04 late evening, **gemma-specific**).
- `data/local/gemma_intro_v7_primed/emotional_raw.jsonl` is the
  primed-main reference dataset. Sidecars under
  `data/local/hidden/gemma_intro_v7_primed/`.
- `face_likelihood` ensemble: under hard-accuracy this doc's
  recommendation was "stays on unprimed encoders + haiku." Under
  soft-everywhere (2026-05-05), gemma_v7primed enters the best
  ensemble — `{gemma_v7primed, opus}` is current. The hard-vs-soft
  flip is itself an argument for the methodology pivot.
- **Don't apply v7 priming to qwen** (or by extension other models in
  the qwen-style register-compliance regime). Architecture-specific.
- Pre-fix introspection data archived at
  `data/archive/2026-05-04_pre_instruction_override/qwen/` and
  root-level for gemma.

## Open / superseded

- **Multi-seed verification of v7 vs v3** (~12 min compute) — open;
  variance band on face_gain is ~±2pp at n=1.
- **Face-stability triple under priming** (scripts 27/28/29 on
  `data/local/gemma_intro_v7_primed/emotional_raw.jsonl`) — open per
  AGENTS.md.
- **Same-v2-hurts-qwen finding** under corrected semantics — closed:
  the qwen-break finding above is the answer (sharply amplified
  under corrected plumbing).
- **Haiku per-face calibrated probabilities → ensemble at higher
  sizes** — superseded by schema-v2 likelihoods + soft-everywhere
  ensembling, which uses haiku's per-quadrant probabilities directly
  in the JSD aggregation.
