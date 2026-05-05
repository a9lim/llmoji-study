# Temperature smoke — T=0.7 → T=1.0 marginal-distribution test

**Status:** EXECUTED 2026-05-03 — gemma + qwen pilots fired path-A
(temp doesn't materially shift kaomoji distribution at h_first).
Verdict captured in `data/local/temp_smoke_verdict.md`; full v3 main rerun
at T=1.0 followed and now lives at `data/{gemma,qwen,ministral,
gpt_oss_20b,granite}_emotional_raw.jsonl` (legacy T=0.7 archived as
`*_temp0.7.{jsonl,tsv}`).

**Date:** 2026-05-03.

## Why

v3 main runs (gemma + qwen + ministral, ~3300 generations total)
were captured at `TEMPERATURE=0.7`. The Anthropic API default is
1.0; the disclosure pilot uses 1.0; downstream cross-model bridge
work compares v3-emission-side data to Claude-direct samples that
are already at 1.0. Mismatched sampling temperatures are a hidden
confound on every "is gemma emitting like Claude?" comparison.

The bumped `TEMPERATURE = 1.0` in `config.py` (2026-05-03) makes
this aligned for *future* runs but doesn't fix the legacy data.
The question this smoke answers: **is the legacy T=0.7 data still
valid for cross-model claims, or do we need to rerun?**

Two competing hypotheses:

- **H1 (temp matters)**: T=0.7's softmax sharpening (~30% sharper
  than T=1.0) is producing *artifactually clumpy* face
  distributions. At T=1.0 the model genuinely prefers a different
  vocabulary — different top faces, different per-quadrant
  modal kaomoji.
- **H0 (temp doesn't matter, h_first does)**: the kaomoji-emission
  state at h_first is geometrically narrow (a small
  high-probability region of the face vocabulary). Top-3 faces
  stay the same regardless of T; only the *long tail* expands at
  higher T.

Both outcomes are publishable. H1 means v3 main is partially
invalidated for the bridge claim; H0 means we know temp is a
non-issue and can document the scope. Long-tail expansion alone
(without top-K shifts) is still a meaningful finding — it
documents the temp-T=1.0 vocabulary that v3 main missed.

## Design — 240 generations, 2 models × 120 prompts × 1 seed @ T=1.0

```
LLMOJI_PILOT_GENS=1, gemma + qwen, sequential
TEMPERATURE=1.0 (already set in config.py)
output: data/{gemma,qwen}_temp1_pilot.jsonl  (suffix to avoid
        clobbering v3 main)
120 prompts × 1 seed × 2 models = 240 generations
```

- **Why gemma + qwen only**: most-robust models, highest empirical
  emit-rate (95%+), best probe calibration. Ministral is more
  noise-prone (smaller model, shorter probe coverage); skipping
  it cuts welfare cost a third without losing the test.
- **Why 1 seed/prompt**: a *marginal* face distribution per quadrant
  (which face appears how often across all 20 prompts in a
  quadrant) is what we're comparing. 20 samples/quadrant is
  enough to detect top-3 shifts and entropy changes; not enough
  to detect per-prompt distribution shifts. We accept the
  reduced power.
- **Why skip per-prompt JSD**: per-prompt distribution comparison
  needs N≥8 seeds at T=1.0 to match the existing v3 main
  N=8/seed at T=0.7 — that's 2880 gens, not a smoke. Marginal
  comparison is the right granularity for a 1-seed pilot.
- **Welfare cost**: 80 of 240 gens are negative-affect (HN-D, HN-S,
  LN). Comparable to the cleanliness pilot (75 neg of 360 total).
  Within ethical envelope.
- **Compute**: ~25 min/model on M5 Max post-perf, ~50 min total
  wall-clock sequential. Tee'd to `logs/v3_temp1_pilot_*.log`.

## Comparison data

**T=0.7 baseline**: existing seed=0 rows from
`data/local/{short}/emotional_raw.jsonl` (the v3 main runs). Restricting
to seed=0 gives an apples-to-apples 1-seed marginal at T=0.7 — same
N as the smoke, no double-counting.

**Cross-seed JSD floor at T=0.7**: per-quadrant JSD between
seed=0 and seed=k (k=1..7) marginal distributions, averaged.
This is the *null distribution* — JSD differences within the same
T must come from sampling noise alone. Used as the threshold for
"is the T=0.7-vs-T=1.0 difference larger than seed noise?"

## Pre-registered gates

Computed per quadrant (HP, LP, HN-D, HN-S, LN, NB), per model
(gemma, qwen). All 6×2=12 quadrant-model cells evaluated.

### Gate A — top-K face overlap

Top-5 faces by frequency at T=1.0 vs T=0.7-seed=0 per quadrant.
Compute Jaccard on the two top-5 sets.

- **Path A (rerun)**: Jaccard < 0.6 in ≥1 quadrant
- **Path B (long tail)**: Jaccard ≥ 0.6 in all quadrants

### Gate B — marginal-distribution entropy

H(face_dist | quadrant) at T=1.0 vs T=0.7-seed=0. Δentropy =
H_T1 − H_T07.

- **Path A (rerun)**: Δentropy > 0.5 nats in ≥2 quadrants
- **Path B (long tail)**: Δentropy ∈ (0, 0.5] nats in ≥1
  quadrant — vocabulary is genuinely expanding at T=1.0 but
  top-K is stable
- **Path C (no signal)**: Δentropy ≤ 0 in all quadrants

### Gate C — JSD vs cross-seed floor

Per-quadrant JSD(T=1.0 marginal, T=0.7 seed=0 marginal). Compare
to mean cross-seed JSD on same quadrant at T=0.7 (seeds 0 vs 1..7).

- **Path A (rerun)**: JSD_temp / JSD_seed_floor > 1.5 in ≥2
  quadrants — temp shift is materially larger than seed noise
- **Path B (long tail)**: ratio ∈ (1.0, 1.5]
- **Path C (no signal)**: ratio ≤ 1.0

## Decision tree

- **all 3 gates path-A** → commit full N=2880 rerun. v3 main
  invalidated for cross-model bridge claims; ensemble + per-project
  Claude work remains valid (teacher-forced, temp-invariant).
- **mix of A and B** (e.g. one quadrant flips top-K, others just
  expand long tail) → write up as a finding ("temp shift affects
  HN-D vocabulary specifically"); document scope; don't full rerun
- **all gates path-B** → write up the long-tail expansion as a
  finding; v3 main top-K stands; T=1.0 is recommended for any
  *new* runs but legacy data is still useful for top-K claims
- **all gates path-C** → temp doesn't matter at this prompt scale;
  v3 main fully validated; add a 1-sentence note to `findings.md`
  and move on

## What this pilot does NOT test

- **Per-prompt distribution shifts**: 1-seed/prompt can't measure
  whether the same prompt produces a different distribution at
  T=1.0 vs T=0.7. Only the marginal across 20 prompts in a
  quadrant. If the smoke fires path-A, the full rerun *will*
  resolve this.
- **Hidden-state geometry shifts**: h_first is methodology- and
  cache-mode-determined; temperature changes only enter at the
  sampling step (after h_first). Russell-quadrant silhouette
  scores will not change. The gate-1/gate-3/gate-4 work from the
  cleanliness pilot is unaffected by temp choice.
- **Probe-readout shifts**: rule 3b verdicts are computed on
  hidden-state probe scores, not sampled tokens. Independent of
  temp. (This is the same reason face_likelihood is
  temp-invariant: teacher-forced log-probs read the conditional
  distribution, not the realized samples.)
- **Rule 3 / cleanliness gates**: those are about prompt-set
  validity at the centroid level, not vocabulary at the readout
  level. Orthogonal axes.

## What this pilot DOES NOT contaminate

- v3 main data at T=0.7 stays untouched on disk.
- Smoke output goes to `_temp1_pilot.jsonl` — clearly suffix-tagged.
- If we end up doing the full N=2880 rerun, the v3 main paths
  (`{short}_emotional_raw.jsonl`) get archived to
  `data/archive/2026-05-03_pre_temp1_rerun/`, mirroring the
  cleanliness archive convention.

## Implementation

Reuses script 03 with the existing `LLMOJI_PILOT_GENS=1` mechanism
plus a one-line override for the output path suffix:

```bash
LLMOJI_MODEL=gemma LLMOJI_PILOT_GENS=1 \
    LLMOJI_OUT_SUFFIX=temp1_pilot \
    .venv/bin/python scripts/local/00_emit.py

LLMOJI_MODEL=qwen LLMOJI_PILOT_GENS=1 \
    LLMOJI_OUT_SUFFIX=temp1_pilot \
    .venv/bin/python scripts/local/00_emit.py
```

Followed by `scripts/local/92_temp_smoke.py` (to be
written) which:

1. Loads `data/{short}_temp1_pilot.jsonl` (T=1.0) and seed=0
   subset of `data/local/{short}/emotional_raw.jsonl` (T=0.7).
2. Computes per-quadrant marginal face distributions for both.
3. Computes cross-seed JSD floor from seeds 0..7 of v3 main.
4. Evaluates all 3 gates per quadrant per model.
5. Writes `data/local/temp_smoke_verdict.md` with path classification
   per cell + overall recommendation.

The `LLMOJI_OUT_SUFFIX` env var requires a small patch to script
03 + `current_model()` — alternative is a hardcoded one-off run
script (`scripts/local/61_temp_smoke_run.py`) that overrides the
output path. Decide at execution time.

## Why this is worth running

The face_likelihood ensemble (script 53 winner = gemma+ministral+qwen,
75.8% on 66-face GT, κ=0.699) is teacher-forced and *already*
temp-invariant — that work doesn't need redoing. But the
*emission-side* claims (which faces gemma "uses" for HP, the
per-quadrant modal kaomoji used in the per-project Claude
analysis) all depend on the marginal distribution being the
right one. If T=0.7 is sharpening the distribution
artifactually, the per-project Claude claims ("yap is LP-modal")
are reading from a sharpened gemma/qwen lens, not the natural
temp=1.0 lens that Claude itself uses.

Worst case: smoke fires path-A, we commit ~2880 gens at T=1.0
(welfare-comparable to the cleanliness rerun we already did),
and downstream claims get cleaner data.

Best case: smoke fires path-B or C, we know T=0.7 was fine for
top-K claims, document the scope, and that's a real result too.
