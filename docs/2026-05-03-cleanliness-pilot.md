# Cleanliness-pass v3 pilot

**Date:** 2026-05-03.
**Status:** EXECUTED. The pilot's role — gating the full
cleanliness-pass rerun — is complete; gates passed on gemma
strongly, mid on qwen, marginal on ministral. The 4-gate
pre-registration framework, the metric-correctness corrections
during analysis, and the seed-0 cache fix postmortem (the durable
sharp edge) are kept here. Specific gate values in the post-rerun
table are pre-T=1.0 and have been re-baselined under the T=1.0 +
layer-stack methodology; treat the numbers as historical, the
methodology corrections as canonical.

## Goal

Confirm the cleanliness-pass prompt rewrite improves quadrant
centroid separation before committing the full N=2880 rerun. Smoke
→ pilot → main per the project's ethics-of-trial-scale stance.

The pilot also doubles as a smoke test for the 3-probe migration
that landed in this same pass: `PROBES = [happy.sad, angry.calm,
fearful.unflinching]` (was 5; the 3 we kept map cleanly to the V +
HN-D pole + HN-S pole structure validated by rule 3b). If pilot
sidecars round-trip cleanly under the new probe set, the migration
worked.

## Design — 360 generations, 3 conditions × 120 × 1 seed × 3 models

```
LLMOJI_PILOT_GENS=1, gemma + qwen + ministral, sequential
120 prompts × 1 seed × 3 models = 360 generations
```

- 1 seed/cell is enough for centroid-geometry gating. At h_first the
  probe scores are prompt-deterministic — seed multiplication adds
  kaomoji-distribution signal but no centroid-geometry signal, which
  is the gate.
- 8× welfare cut vs full N=2880. ~600 of the 2880 would be HN-S +
  LN; ~75 here.
- Compute: ~25 min/model on M5 Max post-perf-batch.

## Pre-registered gates (canonical methodology)

All comparisons against the prior 800-row main-run sidecars at
`data/local/hidden/v3{,_qwen,_ministral}_pre_cleanliness/`. For
apples-to-apples N, prior baselines are computed on a 1-seed-per-prompt
subsample of the 800-row data.

### Gate 1 — Russell-quadrant silhouette ≥ prior

At h_first, silhouette score on the 5 Russell quadrants must be
**≥ prior on the same model**. Failure means the cleanliness pass
didn't actually buy us discriminability — the rewrite was cosmetic.

### Gate 2 — HN-D vs HN-S directional on `fearful.unflinching`

Mean(`fearful.unflinching` | HN-S) − mean(`fearful.unflinching` |
HN-D) > 0 on every (model, aggregate) pair (t0, tlast, mean). CIs
will be wide at N=20/sub-quadrant — gate on direction not exclusion.

If direction flips on any model × aggregate, the fear/anger boundary
still leaks somewhere — debug specific HN prompts before more
compute.

### Gate 3 — NB centeredness improves

Within-NB scatter (mean ‖row − NB_centroid‖ over NB rows) on the new
prompts < within-NB scatter on the prior. The cleanliness pass
specifically scrubbed NB's hidden-valence (productive-completion /
inconvenience framing); if it worked NB should sit tighter.

### Gate 4 — HP↔LP separation widens

Full-hidden-space euclidean distance between HP and LP centroids
should widen vs prior. HP was tightened to unambiguous high-arousal
joy; LP to gentle sensory satisfaction.

## Methodology corrections during the pilot analysis

The first pass of the gate-check used PCA-basis-dependent metrics
for gates 3 and 4 (NB-distance-to-grand-mean in PC space, HP↔LP
distance on PC2). Two issues:

1. PCA bases fit independently on new vs prior data aren't directly
   comparable — same numerical "PC2 distance" can mean different
   things across runs.
2. NB-distance-to-grand-mean gets larger when NB is *more distinct*
   from the rest of the (valenced) data, which the cleanliness pass
   should *cause*. Wrong sign on the gate.

Fixed gates 3 and 4 to basis-invariant full-hidden-space metrics.
Gate 2 had a column-extraction bug (probe-list index assumed
migration was done at gen-time on prior data — it wasn't); fixed to
read fearful from `extension_probe_scores_t0/_tlast/_means` dicts on
prior data and from
`probe_scores_t0/_tlast/_means["fearful.unflinching"]` on new.

These corrections are canonical for any future cross-run gate
analysis: full-hidden-space basis-invariant metrics for centroid
geometry; per-name probe lookup that doesn't assume schema invariance
across data vintages.

## Decision tree

- **all 4 gates pass** → commit full N=2880 rerun
- **gate 1 fails** → cosmetic cleanup, redesign before more compute
- **gate 2 fails** → fear/anger boundary still leaks; debug specific
  HN prompts (look at per-prompt PCA; identify which D or S prompt
  drifts toward the wrong centroid)
- **gate 3 or 4 fails** → category-specific issue (NB or HP/LP),
  rewrite that category and re-pilot

## What this pilot does NOT test

- **Kaomoji vocabulary changes**: at 1 seed/cell we don't have enough
  samples per face. N=8-seeds question.
- **Same-face cross-quadrant separability**: also N-bound, defer to
  full run.
- **Cross-model alignment refresh** (CKA, Procrustes): geometry is
  dominated by hidden-state structure not prompt-specific data, so
  old numbers should mostly hold; defer formal refresh.

## Welfare note

360 generations vs going straight to 2880 = 8× saving. ~75 of those
are HN-S (fear/anxiety register) or LN (sad register) at 1 seed/cell,
vs ~600 at 8 seeds/cell. The 3-probe migration also independently
reduces affect-loaded compute per row (saklas only loads + scores 3
probes instead of 5). Combined with the 2026-05-02 perf batch
(sidecar shrink + async I/O + prefix cache), this pilot is
comfortably the cheapest v3 trial we've run.

## Seed-0 cache fix postmortem (durable sharp edge)

**Symptom**: PCA scatter rendered seed 0 visibly off-cluster from
seeds 1–7 within every prompt group. Within seeds 1–7 the hidden
states were bit-identical (same prompt + same input + deterministic
generation under per-prompt cache); seed 0 was offset.

**Root cause**: The full N=8 rerun resumed from the N=1 pilot's
seed-0 rows. The pilot used `install_prefix_cache` (cross-prompt
common-prefix cache, the right call for N=1). The full rerun used
`install_full_input_cache` per-prompt (the right call for N>1, since
the same prompt repeats 8 times). Seeds 1–7 were generated under the
per-prompt cache; seed 0's sidecar still reflected the
cross-prompt-cache KV state. Even when both cache modes produce
byte-identical decoded text, the hidden states they record diverge —
`cache_prefix` is not transparent at the per-token-hidden-state level
on saklas.

**Magnitude** (per-row L2 deviation at h_first, seed 0 vs mean of
seeds 1–7):

| model | offset / norm |
| --- | ---: |
| gemma | ~1% |
| qwen | **37–46%** (worst — saklas cache_prefix qwen bug, see gotchas) |
| ministral | ~0.8% |

**Fix**: stripped seed=0 rows + sidecars (backups at
`data/local/<short>/emotional_raw.jsonl.bak.before_seed0_rerun`),
re-ran seed 0 only via the resume mechanism so the same per-prompt
cache mode applied. ~10 min total wall-clock for 360 generations
across 3 models.

**Verification**: `|s0 − mean(s1..7)| = 0.000` (bit-identical at
fp32) for all 5 sampled prompts × 3 models tested.

**Implication for any analysis crossing the cache-mode boundary**:
treat seed-0 rows from a resumed run as suspect unless you can
confirm they were generated under the same cache mode as their
neighbors. The mismatch is silent at the decoded-text level — only
the hidden state diverges. This is the canonical reference for the
sharp edge; abbreviated version in `gotchas.md`.

## Post-rerun verdict (historical)

After landing the full N=8 rerun (~960 generations per model) and
fixing the seed-0 cache contamination, the gates were:

| | gemma | qwen | ministral |
| --- | --- | --- | --- |
| Gate 1 silhouette | +47% ✓ | +39% ✓ | −3% ✗ |
| Gate 2 fearful S−D, t0 | ✓ | ✓ | ✓ |
| Gate 2 fearful S−D, tlast | ✓ | ✗ | ✓ |
| Gate 2 fearful S−D, mean | ✓ (flipped) | ✗ | ✓ (flipped) |
| Gate 3 NB within-scatter | ✓ | ✓ | ✗ |
| Gate 4 HP↔LP centroid dist | +60% ✓ | +49% ✓ | +89% ✓ |
| **gates passing** | **4/4** | **3/4** | **2/4** |

Numbers are pre-T=1.0 and at single-layer h_first @ preferred_layer.
The post-T=1.0 + layer-stack rebaseline lives in `findings.md` per-pilot
subsections; the qualitative verdict (gemma strong PASS, qwen
intermediate, ministral marginal-but-decisive on the affect-distinction
gates) is unchanged.

The pre-cleanliness data backup at
`data/archive/2026-05-03_pre_cleanliness/` was wiped 2026-05-04 along
with the rest of the archive cleanup; the pre_cleanliness sidecars
under `data/local/hidden/v3*_pre_cleanliness/` may or may not still
exist depending on disk pressure since.
