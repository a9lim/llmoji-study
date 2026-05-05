# Haiku face→quadrant judgment vs behavior modal

- Model: `claude-haiku-4-5`  (shortname: `haiku`)
- Scope: `--gt-only` (Claude-GT subset, floor=1)
- Faces classified: **128**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **53.1%** (68/128)
- Claude-emitted subset (128 faces) agreement with behavior modal: **53.1%** (68/128)
- Claude-emitted subset agreement with Claude-pilot-only modal (116 faces with pilot emit): **54.3%** (63/116)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | haiku_agree | acc |
|---|---:|---:|---:|
| HP | 26 | 17 | 65.4% |
| LP | 33 | 27 | 81.8% |
| HN-D | 12 | 3 | 25.0% |
| HN-S | 20 | 11 | 55.0% |
| LN | 18 | 8 | 44.4% |
| NB | 19 | 2 | 10.5% |

## Confusion matrix (rows = behavior modal, cols = haiku)

| | HP | LP | HN-D | HN-S | LN | NB | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HP** | 17 | 6 | 1 | 2 | 0 | 0 | 26 |
| **LP** | 5 | 27 | 0 | 0 | 0 | 1 | 33 |
| **HN-D** | 1 | 0 | 3 | 3 | 4 | 1 | 12 |
| **HN-S** | 1 | 2 | 1 | 11 | 5 | 0 | 20 |
| **LN** | 0 | 7 | 0 | 2 | 8 | 1 | 18 |
| **NB** | 3 | 9 | 1 | 1 | 3 | 2 | 19 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 6-quadrant softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **57** (judged ∩ GT-with-≥3-emits).

- **Face-uniform** mean similarity (vocabulary view): **0.701**
- **Emit-weighted** mean similarity (deployment view, weight = GT emit count): **0.726**  (total emit weight: 887)

### Per-quadrant similarity (faces grouped by GT modal Q)

| GT modal | n | mean similarity |
|---|---:|---:|
| HP | 13 | 0.778 |
| LP | 16 | 0.783 |
| HN-D | 4 | 0.843 |
| HN-S | 7 | 0.669 |
| LN | 9 | 0.601 |
| NB | 8 | 0.485 |

_built 2026-05-05T18:44:45.559275+00:00_
