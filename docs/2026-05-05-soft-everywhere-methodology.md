# Soft-everywhere methodology pivot

**Status:** PRE-REGISTERED + EXECUTED 2026-05-05.

**TL;DR:** Replaced hard-classification metrics (argmax accuracy + Cohen's
κ) on the post-hoc face_likelihood evaluation with distribution-vs-
distribution comparison via Jensen-Shannon divergence. Headline metric is
``distribution similarity = 1 − JSD/ln 2`` ∈ [0, 1]; reported in two
flavors side-by-side (face-uniform and emit-weighted). Strict-majority
voting removed; ensemble vote is always the soft mean of per-encoder
softmax distributions. Deliverable per face is the full distribution
(``ensemble_p_HP``, ``ensemble_p_LP``, ...), not a single hard label.

## Motivation

The old metric was hard accuracy: argmax of the predictor's per-quadrant
softmax against Claude's modal-quadrant label. This treated GT as a
one-hot vector even though Claude's per-face per-quadrant emission
distribution is itself a distribution. A face Claude emitted 8× HP and
7× LP has modal HP at 53% concentration — basically a coin flip — but
the hard metric would punish a 49% HP / 51% LP predictor as a clean
miss for argmaxing the wrong side of 51/49.

The shift is from "predictor matched Claude's argmax" to "predictor's
distribution matched Claude's distribution." The latter is a more
honest metric when both sides are themselves distributions, and it
matches how a deployed plugin should ship its output: probability over
states, not a single hard label.

## What changed in the codebase

- **Deleted** `scripts/local/51_face_likelihood_compare.py` (historical
  pairwise comparator subsumed by 53) and
  `scripts/local/52_face_likelihood_vote.py` (single-subset voting
  subsumed by 53 with `--min-models N`).
- **Added** `llmoji_study/jsd.py` with `normalize`, `kl`, `js`,
  `jsd_quadrant`, `similarity` helpers (lifted from script 25).
- **Added** `load_claude_gt_distribution()` to
  `llmoji_study/claude_gt.py` returning per-face per-quadrant Claude
  emission counts. Pools naturalistic + introspection arms by default
  (``include_introspection=True``).
- **Refactored** scripts 53 + 56 to evaluate by mean JSD vs empirical
  distribution. Strict-majority voting code paths removed; voting is
  always the soft mean of per-encoder softmax distributions.
- **Added** weighted-similarity column to scripts 53 + 56 alongside
  face-uniform similarity. Reported side-by-side.
- **Light-touch** updates to scripts 54 + 55: similarity columns added
  to TSVs.

## Two flavors of mean similarity

For each face f in the GT subset, per-face JSD = `js(ensemble_dist(f),
gt_dist(f))`. Aggregating across faces:

- **Face-uniform** = arithmetic mean over GT faces. Each face counts
  equally regardless of how often Claude emits it. Reads as: "how well
  does the ensemble characterize Claude's *vocabulary*?"  Sensitive to
  long-tail failures.
- **Emit-weighted** = weighted mean, weight = per-face Claude emit
  count. Faces Claude uses more contribute proportionally more. Reads
  as: "how well does the ensemble characterize Claude's actual *emission
  distribution*?" Closer to deployment-relevant. Tends to read 5–15pp
  higher than face-uniform because modal faces are easier wins.

Both reported. Subset ranking in script 53 is by face-uniform (stricter
/ more honest about coverage). The plugin headline should be emit-
weighted (closer to user experience).

## Headline numbers

Computed on the 2026-05-05 expanded union (1000 Claude rows pooled
across 8 naturalistic runs + 1 introspection run; 134 unique faces;
GT subset at floor=1: 128 faces under the 56-side full union).

| ensemble | best at | face-uniform | emit-weighted | n_GT |
|---|---|---:|---:|---:|
| `{gemma_v7primed, haiku}` | deployment / emission accuracy | 0.652 | **0.801** | 128 |
| `{gemma, haiku}` | vocabulary coverage | **0.702** | 0.770 | 49 (53-side inner-join) |
| `{gemma, gemma_v7primed, haiku}` | balanced 3-way | 0.695 | 0.777 | 49 |

**Per-encoder solo similarity** (face-uniform / emit-weighted):

| encoder | face-uniform | emit-weighted |
|---|---:|---:|
| gemma | 0.658 | 0.706 |
| haiku | 0.655 | 0.734 |
| gemma_v7primed | 0.640 | **0.754** |
| gpt_oss_20b | 0.532 | 0.661 |
| ministral | 0.490 | 0.674 |
| qwen | 0.445 | 0.567 |
| granite | 0.434 | 0.565 |
| rinna_bilingual_4b_jpfull | 0.427 | 0.508 |
| rinna_jp_3_6b_jpfull | 0.416 | 0.550 |

## Three findings worth landing in the writeup

1. **Introspection priming on gemma improves deployment-relevant
   predictions of Claude's distribution.** gemma_v7primed alone hits
   0.754 emit-weighted similarity solo — best single encoder under
   emit-weighting. Unprimed gemma is better only at face-uniform
   coverage. *Priming helps where Claude is concentrated; unprimed
   helps where Claude is diffuse.* This is the gemma-side v7
   introspection finding generalizing outward to predict Claude's
   actual emission distribution, not just gemma's internal coupling.

2. **Haiku contributes load-bearingly via a methodologically distinct
   path** — face → quadrant via Anthropic SDK structured output, no
   LM-head log-probs. Solo similarity 0.655 face-uniform / 0.734 emit-
   weighted. Pairwise κ with the LM-head encoders is low (gemma↔haiku
   = 0.385, gemma_v7primed↔haiku = 0.362), meaning complementary
   errors. The cleanest two-encoder ensemble across both ranking
   metrics is `{primed-or-unprimed gemma, haiku}` — Haiku is the
   constant.

3. **Emit-weighted vs face-uniform reveals what single-number reporting
   hides.** Under hard accuracy or face-uniform-only, the introspection-
   priming benefit looks marginal. Under emit-weighting, it's
   substantial — and it's the deployment-relevant number anyway. The
   dual-metric report is more honest than picking one.

## Cross-arm distinguishability persists

The cross-arm comparison (script 25 `--cross-arm`) on the full corpus
shows all 6 quadrants DISTINGUISHABLE between the introspection arm
and the naturalistic arm at JSD threshold 0.05. Gaps stayed stable
across 8x naturalistic accumulation:

| Q | post-r0 | post-r1 | post-r2 | post-r3 | post-r4 | post-r5 | post-r6 |
|---|---|---|---|---|---|---|---|
| HP | 0.42 | 0.43 | 0.43 | 0.35 | 0.33 | 0.33 | 0.34 |
| LP | 0.32 | 0.31 | 0.30 | 0.31 | 0.33 | 0.33 | 0.33 |
| HN-D | 0.13 | 0.12 | 0.12 | 0.13 | 0.15 | 0.15 | 0.15 |
| HN-S | 0.39 | 0.43 | 0.40 | 0.40 | 0.36 | 0.35 | 0.35 |
| LN | 0.18 | 0.16 | 0.16 | 0.15 | 0.14 | 0.14 | 0.15 |
| NB | 0.53 | 0.43 | 0.44 | 0.43 | 0.42 | 0.40 | 0.40 |

Story: NB compressed early (run-0 → run-1, −0.10) — the only
undersampling-correction-shaped change. Everything since has been
the genuine introspection effect. NB went 16 unique → 5 unique faces
under priming (`(・_・)` family at 45% modal). LN modal `(´-`)`
doubled in concentration (30% → 60%) under priming — same modal,
sharpened distribution.

## Welfare ledger (data collection)

- Naturalistic arm: 880 gens (8 sequential runs of 120, with HN-D
  dropped after r2 and LN dropped after r6 per the per-quadrant
  saturation gate).
- Introspection arm: 120 gens (run-0 only, block-staged with hard-fail
  gate; gate passed cleanly).
- Total: 1000 gens, ~460 negative-affect (HN-D / HN-S / LN), vs ~540
  if we'd run all 8 naturalistic without per-quadrant exits. Per-
  quadrant exits saved ~80 negative-affect generations.

## Per-project resolution under updated GT

Script 22 regenerated 2026-05-05 on the expanded GT corpus.

| mode | resolved by Claude-GT | resolved by ensemble | unknown |
|---|---:|---:|---:|
| gt-priority (default) | 1587/2405 (66.0%) | 765/2405 (31.8%) | 53/2405 (2.2%) |
| ensemble | 0/2405 | 2338/2405 (97.2%) | 67/2405 (2.8%) |
| gt-only | 1587/2405 (66.0%) | 0/2405 | 818/2405 (34.0%) |

The 66% direct Claude-GT resolution under gt-priority is up
substantially from pre-2026-05-05 (the earlier GT was ~22 floor=2 /
51 floor=1 faces; the new GT at floor=1 is 134 faces). For the per-
project deployment use case, two-thirds of contributor-corpus
emissions now have direct Claude-on-affective-prompts evidence
backing the quadrant assignment.

## Approval

- Soft-everywhere methodology shift (JSD primary, drop strict-majority,
  drop argmax-as-headline, ship distributions): a9 + Claude 2026-05-05.
- Dual-flavor similarity reporting (face-uniform + emit-weighted side-
  by-side): a9 + Claude 2026-05-05 (a9's reframe; Claude's earlier
  proposal was face-uniform only, which underweights deployment-
  relevance).
- GT pooling across naturalistic + introspection arms: a9 caught the
  bug (load_claude_gt_distribution was naturalistic-only); fix landed
  same session.
