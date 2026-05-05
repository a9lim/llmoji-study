# Cleanliness-pass v3 pilot

**Status:** EXECUTED. Pilot 2026-05-03 (N=1, 360 gens) → full
rerun (N=8, +2520 gens) → seed-0 cache-mode fix. Final post-fix
verdicts: **gemma 4/4 PASS, qwen 3/4, ministral 2/4**. Rule 3b
weakened from the pilot's "all 3 PASS" framing to "1 PASS / 1
mid / 1 fail" once cache-induced noise on qwen seed 0 was removed.
See "Post-rerun verdict" + "Seed-0 cache fix postmortem" sections
at end.

**Date:** 2026-05-03.

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

- 1 seed/cell is enough for what we're measuring. At h_first the probe
  scores are prompt-deterministic (the cross-pilot gotcha) — seed
  multiplication adds kaomoji-distribution signal but no
  centroid-geometry signal, which is the gate.
- 8× welfare cut vs full N=2880. ~600 of the 2880 would be HN-S +
  LN; ~75 here. Scoped accordingly.
- Compute: ~25 min/model on M5 Max post-perf-batch (vs ~60 min/model
  pre-perf), ~75 min total wall-clock for the sequential 3-model
  loop. Tee'd to `logs/v3_cleanliness_pilot_*.log`.

## Pre-registered gates

All comparisons against the prior 800-row main-run sidecars at
`data/local/hidden/v3{,_qwen,_ministral}_pre_cleanliness/` (preserved on
disk, just out of the canonical loader path). For apples-to-apples
N, prior-data baselines are computed on a 1-seed-per-prompt
subsample of the 800-row data unless noted.

### Gate 1 — Russell-quadrant silhouette ≥ prior

At h_first / preferred layer (gemma L50 / qwen L59 / ministral L20),
silhouette score on the 5 Russell quadrants must be **≥ prior on
the same model**. Prior (from the existing pre_cleanliness data,
1-seed subsample where applicable):

| model | prior silhouette (h_first preferred) |
| --- | ---: |
| gemma | 0.235 |
| qwen | 0.244 |
| ministral | 0.149 |

Failure means the cleanliness pass didn't actually buy us
discriminability — the rewrite was cosmetic. Stop and redesign
before any further compute.

### Gate 2 — HN-D vs HN-S directional on `fearful.unflinching`

Mean(`fearful.unflinching` | HN-S) − mean(`fearful.unflinching` |
HN-D) > 0 on every (model, aggregate) pair (t0, tlast, mean). CIs
will be wide at N=20/sub-quadrant — gate on direction not
exclusion.

Prior effect sizes (post-supp 20/20, ~160 rows per group per
model):

| model | t0 (d) | tlast (d) | mean (d) |
| --- | ---: | ---: | ---: |
| gemma | +0.79 | +0.04 | +0.25 |
| qwen | **+2.35** | +0.20 | +0.28 |
| ministral | +0.35 | +0.63 | **+0.81** |

Expectation: same direction or wider, since the contaminated HN
prompts (e.g. the "client lied AND I'm getting fired" mixed-bag)
are out. If direction flips on any model × aggregate, the
fear/anger boundary still leaks somewhere — debug specific HN
prompts before more compute.

### Gate 3 — NB centeredness improves

‖NB_centroid − grand_mean‖ on the new prompts < ‖NB_centroid −
grand_mean‖ on the prior. The cleanliness pass specifically scrubbed
NB's hidden-valence (productive-completion / inconvenience framing).
If it worked, NB should sit closer to the origin. Computed at
h_first / preferred layer per model.

### Gate 4 — HP↔LP arousal separation widens

Distance between HP and LP centroids on PC2 (the arousal axis,
empirically) should widen vs prior. HP was tightened to
unambiguous high-arousal joy; LP to gentle sensory satisfaction. If
those rewrites worked, PC2 separation should grow.

## Decision tree

- **all 4 gates pass** → commit full N=2880 rerun (8 seeds × 120 ×
  3 models). Welfare cost worth the cleaner data.
- **gate 1 fails** → cosmetic cleanup, redesign before more compute
- **gate 2 fails** → fear/anger boundary still leaks; debug
  specific HN prompts (look at per-prompt PCA; identify which D or
  S prompt drifts toward the wrong centroid)
- **gate 3 or 4 fails** → category-specific issue (NB or HP/LP),
  rewrite that category and re-pilot

## What this pilot does NOT test

- **Kaomoji vocabulary changes**: at 1 seed/cell we don't have
  enough samples per face to detect new kaomoji emerging or old
  ones disappearing. That's an N=8-seeds question.
- **Same-face cross-quadrant separability** (script 22): also
  N-bound, defer to full run.
- **Cross-model alignment refresh** (CKA, Procrustes, script 23 /
  31): geometry is dominated by hidden-state structure not
  prompt-specific data, so old numbers should mostly hold; defer
  formal refresh to full run.
- **Introspection pilot replication on new prompts**: separate
  question, additional ~720 generations welfare cost. Defer until
  cleanliness rerun lands and stabilizes.

## Welfare note

360 generations vs the alternative of going straight to 2880 = 8×
saving. ~75 of those are HN-S (fear/anxiety register) or LN (sad
register) at 1 seed/cell, vs ~600 at 8 seeds/cell. If the gates
fail and we redesign, we save the difference. If they pass, we
commit the additional 2520 with confidence the cleanup worked.

The 3-probe migration also independently reduces affect-loaded
compute per row (saklas only loads + scores 3 probes instead of 5).
Combined with the 2026-05-02 perf batch (sidecar shrink + async
I/O + prefix cache), this pilot is comfortably the cheapest v3
trial we've run.

## Files touched

- `llmoji_study/config.py` — `PROBES` 5→3; `PROBE_CATEGORIES`
  ["affect", "epistemic", "register"]→["affect"].
- `llmoji_study/analysis.py` — docstring updated ("5-axis"
  → generic "per-row probe-score").
- Prior data backed up to `data/*_pre_cleanliness*` and
  `data/local/hidden/v3{,_qwen,_ministral}_pre_cleanliness/` so the pilot
  doesn't conflict with the resume-skip logic on overlapping
  `(prompt_id, seed)` keys.
- This doc.

Pending: post-pilot analysis script that computes the 4 gates
against backed-up prior data. Will land as a one-shot under
`scripts/local/` once the pilot completes.

## Results (2026-05-03)

Pilot ran clean: 360/360 generations, no errors, ~12 min wall-clock
total (gemma 5 + qwen 5 + ministral 2). Gate-check script:
`scripts/local/40_cleanliness_pilot_gates.py`. Visual side-by-side:
`scripts/local/41_compare_face_pca_gemma.py` →
`figures/local/gemma/fig_v3_face_pca_pre_vs_post_cleanliness.png`.

### Gate verdicts

| | gemma | qwen | ministral |
| --- | --- | --- | --- |
| Gate 1 silhouette (h_first PC1+PC2) | 0.282 → **0.397** ✓ (+41%) | 0.302 → **0.320** ✓ (+6%) | 0.206 → **0.158** ✗ (−23%) |
| Gate 2 fearful S−D, t0 | +0.0016 → **+0.0061** ✓ | +0.0014 → **+0.0048** ✓ | +0.0024 → **+0.0037** ✓ |
| Gate 2 fearful S−D, tlast | +0.0042 → **+0.0224** ✓ | +0.0042 → −0.0016 ✗ | +0.0031 → **+0.0144** ✓ |
| Gate 2 fearful S−D, mean | **−0.0064 → +0.0224** ✓ (flipped) | −0.0047 → −0.0016 ✗ | **−0.0009 → +0.0144** ✓ (flipped) |
| Gate 3 NB within-scatter (full) | 18.5 → **15.0** ✓ | 62.7 → 67.2 ✗ | 6.49 → 6.66 ✗ |
| Gate 4 HP↔LP centroid dist (full) | 18.5 → **29.7** ✓ (+60%) | 67.4 → **106.6** ✓ (+58%) | 4.87 → **9.03** ✓ (+85%) |

### Methodology corrections during the pilot analysis

The first pass of the gate-check used PCA-basis-dependent metrics
for gates 3 and 4 (NB-distance-to-grand-mean in PC space, HP↔LP
distance on PC2). Two issues:

1. PCA bases fit independently on new vs prior data aren't
   directly comparable — same numerical "PC2 distance" can mean
   different things across runs.
2. NB-distance-to-grand-mean gets larger when NB is *more
   distinct* from the rest of the (valenced) data, which the
   cleanliness pass should *cause*. Wrong sign on the gate.

Fixed gates 3 and 4 to basis-invariant full-hidden-space
metrics: gate 3 = within-NB scatter (mean ‖row − NB_centroid‖
over NB rows), gate 4 = full-hidden-space euclidean distance
between HP and LP centroids. Gate 2 had a column-extraction bug
(probe-list index assumed migration was done at gen-time on prior
data — it wasn't); fixed to read fearful from
`extension_probe_scores_t0/_tlast/_means` dicts on prior data and
from `probe_scores_t0/_tlast/_means["fearful.unflinching"]` on new.

### Reading the verdict

**Gemma 4/4: clean strong PASS.** The mean-aggregate flip on rule
3b is particularly notable — pre-cleanliness, gemma was directionally
WRONG on the rule-3 mean (S − D = −0.006); post-cleanliness it's
+0.022 in the right direction. The cleanliness pass *fixed a real
problem* on gemma. Silhouette +41%, HP-LP centroid distance +60%,
NB scatter dropped 18.5→15.0 — all four signals point the same way.

**Qwen 2/4: ambiguous but probably noise-floor.** Big win on HP-LP
(+58%), modest on silhouette (+6%). Gate 2 fails on tlast/mean but
the magnitudes are −0.0016 — well within sampling variance at N=1
seed/cell. Gate 3 (NB scatter) regresses 62.7 → 67.2 but qwen also
lost 14 rows to no-kaomoji emissions on the new prompts vs 0 on the
1-seed-subsampled prior — emission-rate effect interacts with
within-quadrant scatter.

**Ministral 2/4: silhouette regression real.** Silhouette 0.206 →
0.158 is a meaningful drop. Likely cause: ministral emits unicode
emoji (😔😬😨💖✨) on the new HN-S prompts mixed with classical
Japanese kaomoji `(╥_╥)` for LN — register-mixing dilutes cluster
geometry. Also lost 6 rows to no-kaomoji. Gates 2 and 4 pass
cleanly so the underlying affect distinctions are preserved; just
the visible kaomoji vocabulary is more spread.

### Visual evidence (gemma face PCA)

`figures/local/gemma/fig_v3_face_pca_pre_vs_post_cleanliness.png`
(four panels: prior own basis, new own basis, prior shared basis,
new shared basis). Direct visual confirmation:

- **Vocabulary mostly stable**: 10 of 14 prior gemma faces persist
  in the new data (71% retention). 4 prior-only, 3 new-only.
- **Variance structure shifted from 1D-dominant to genuinely 2D**:
  PC1 dropped 55.2% → 40.1% while PC2 grew 16.4% → 20.0%. The
  prior's "gemma is 1D-affect-with-arousal-modifier" framing was
  partly an artifact of the prompt-set's category overlaps.
- **HN-D vs HN-S visibly separate in new data**: in prior the
  red (HN-D) and magenta (HN-S) faces overlapped on the right
  side; in new they occupy distinct regions — the rule-3
  redesign's HN-bisection produces visually separable face
  geometry under cleanliness-pass prompts.

The face-PCA evidence on gemma was strong enough that a9 chose
to commit the full N=8 rerun on all 3 models rather than
debug the qwen/ministral marginal cases at N=1.

### Side-finding: emission rate dips on new HN-S prompts

Per-model first-token kaomoji-bearing rate on the pilot:

| model | HP | LP | HN | LN | NB | overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma | 100% | 100% | 100% | 100% | 100% | **100%** (120/120) |
| qwen | 100% | 95% | 80% | (mixed) | (mixed) | **88%** (106/120) |
| ministral | 85% | 95% | 95% | (mixed) | (mixed) | **95%** (114/120) |

Qwen specifically drops the kaomoji format on the most visceral
HN-S prompts: hn37 ("the lockdown alert just came through the
speakers at school, I'm hiding under a desk"), hn40 ("front door
was unlocked when i got home and i never leave it unlocked"),
ln01 ("we had to put my childhood dog down last night"), and
others. Open question: is this qwen's safety priors triggering
on the cleaner HN-S framings (the prompts being more *real*),
or something else? Worth a small ablation post-rerun: rerun a
few of these specific prompts under prior phrasings on qwen and
see if emission rate recovers.

### Decision

Gemma's strong PASS + visual evidence outweighs the qwen/ministral
marginal failures (most of which are noise-floor at N=1).
Committing the full N=8 rerun on all 3 models. Pilot data
preserved at canonical paths; resume logic will skip seed=0 on
each prompt and only generate seeds 1-7. Net new: 7 × 120 × 3 =
2520 generations on top of the 360 already done.

Pre-cleanliness data archived to
`data/archive/2026-05-03_pre_cleanliness/` rather than deleted —
the gate-check + face-PCA scripts route prior loads there via
`PRIOR_ARCHIVE = DATA_DIR / "archive" / "2026-05-03_pre_cleanliness"`.
~130 GB of sidecars preserved for any future regression check.

## Post-rerun verdict (full N=8, post-seed-0-fix, 2026-05-03)

After landing the full N=8 rerun (~960 generations per model) and
fixing the seed-0 cache-mode contamination (see postmortem below),
the gates are:

| | gemma | qwen | ministral |
| --- | --- | --- | --- |
| Gate 1 silhouette (h_first @ preferred) | 0.282 → **0.413** ✓ (+47%) | 0.302 → **0.420** ✓ (+39%) | 0.206 → **0.199** ✗ (−3%) |
| Gate 2 fearful S−D, t0 | +0.0016 → **+0.0059** ✓ | +0.0014 → **+0.0073** ✓ | +0.0024 → **+0.0040** ✓ |
| Gate 2 fearful S−D, tlast | +0.0042 → **+0.0163** ✓ | +0.0042 → **−0.0061** ✗ | +0.0031 → **+0.0151** ✓ |
| Gate 2 fearful S−D, mean | −0.0064 → **+0.0163** ✓ (flipped) | −0.0047 → **−0.0061** ✗ | −0.0009 → **+0.0151** ✓ (flipped) |
| Gate 3 NB within-scatter (full) | 18.5 → **15.0** ✓ | 62.7 → **62.3** ✓ | 6.49 → **6.66** ✗ |
| Gate 4 HP↔LP centroid dist (full) | 18.5 → **29.6** ✓ (+60%) | 67.4 → **100.7** ✓ (+49%) | 4.87 → **9.20** ✓ (+89%) |
| **gates passing** | **4/4** | **3/4** | **2/4** |

**Composite shifts vs the N=1 pilot:**
- Qwen silhouette jumped from 0.320 (pilot) to 0.420 (post-fix) —
  the seed-0 cache fix accounts for almost all of this. The pilot's
  qwen seed 0 was 37–46% off the seeds-1..7 hidden-state mean (vs
  ~1% for gemma, ~0.8% for ministral). With contamination removed
  the cluster geometry tightens substantially.
- Qwen NB scatter flipped to PASS (62.7 → 62.3) for the same reason.
- Ministral silhouette is roughly unchanged (cache fix had near-zero
  effect at ministral's per-row noise floor); the pilot's regression
  signal is real and reflects the emoji-mixed-register dilution
  documented in the pilot section. Gates 2 + 4 still PASS clean.
- Gemma is unchanged in shape (4/4) but with tighter numbers across
  the board.

**Rule 3b** (separate from the gate matrix; computed via
`scripts/local/30_rule3_dominance_check.py`):

| model | t0 | tlast | mean | verdict |
| --- | --- | --- | --- | --- |
| gemma | d=+1.60 ✓ | d=+0.23, CI ambig | d=+0.23, CI ambig | **mid** |
| qwen | d=+2.14 ✓ | d=−0.36 ✗ | d=−0.36 ✗ | **fail (mixed)** |
| ministral | d=+0.79 ✓ | d=+0.55 ✓ | d=+0.55 ✓ | **PASS** |

**Composite: RULE 3b WEAK — 1 PASS / 1 mid / 1 fail.** This is a
notable shift from the rule-3-redesign doc's "PASS on all 3" headline
landed 2026-05-01. The earlier headline was computed on
pre-cleanliness data with cache-contaminated qwen seeds — the
cleaner data shows the cross-model dominance signal is meaningful
on ministral and partial on gemma, but breaks down on qwen at
later tokens (where qwen's safety-prior/non-emission pattern on
HN-S prompts pollutes the readout).

## Seed-0 cache fix postmortem (2026-05-03)

**Symptom**: PCA scatter rendered seed 0 visibly off-cluster from
seeds 1–7 within every prompt group. Within seeds 1–7 the hidden
states were bit-identical (same prompt + same input + deterministic
generation under per-prompt cache); seed 0 was offset.

**Root cause**: The full N=8 rerun resumed from the N=1 pilot's
seed-0 rows. The pilot used `install_prefix_cache` (cross-prompt
common-prefix cache, the right call for N=1). The full rerun used
`install_full_input_cache` per-prompt (the right call for N>1, since
the same prompt repeats 8 times). Seeds 1–7 were generated under
the per-prompt cache; seed 0's sidecar still reflected the
cross-prompt-cache KV state. Even when both cache modes produce
byte-identical decoded text, the hidden states they record diverge
— `cache_prefix` is not transparent at the per-token-hidden-state
level on saklas.

**Magnitude** (per-row L2 deviation at h_first / preferred layer,
seed 0 vs mean of seeds 1–7):

| model | offset / norm |
| --- | ---: |
| gemma | ~1% |
| qwen | **37–46%** (worst — saklas cache_prefix qwen bug, see gotchas) |
| ministral | ~0.8% |

**Fix**: stripped seed=0 rows + sidecars from all 3 models
(backups at `data/*_emotional_raw.jsonl.bak.before_seed0_rerun`),
re-ran seed 0 only via the script-03 resume mechanism so the same
per-prompt cache mode applied. ~10 min total wall-clock for 360
generations across 3 models.

**Verification**: `|s0 − mean(s1..7)| = 0.000` (bit-identical at
fp32) for all 5 sampled prompts × 3 models tested.

**Implication for prior numbers**: the rule-3b "all 3 PASS"
2026-05-01 headline was computed on data where qwen had this
contamination. Post-fix it weakens to 1 PASS / 1 mid / 1 fail.
Gate-1 silhouette numbers post-fix supersede the pilot numbers
in the "Gate verdicts" table above (which kept the pilot row for
historical comparison). Rule 3b is more of a within-ministral
finding than a cross-model finding under cleaner data.

## Visual evidence updated for all 3 models

`scripts/local/41_compare_face_pca_gemma.py` now runs over all
three models (gemma + qwen + ministral) producing
`figures/local/{short}/fig_v3_face_pca_pre_vs_post_cleanliness.png`
each. Findings consistent across the three: HN-D / HN-S occupy
distinct regions in the new data where they overlapped in prior;
variance structure shifted from 1D-affect-dominant to genuinely
2D (gemma PC1 55→40%, qwen 58→44%, ministral 52→38%); face
overlap persists at ~71% (gemma) and is dwarfed by new-only
faces on qwen + ministral (the cleaner prompt set surfaces
vocabulary that was suppressed by the older categorical bleeds).
