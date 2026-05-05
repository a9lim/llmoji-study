# Haiku face→quadrant judgment vs behavior modal

- Model: `claude-haiku-4-5`
- Faces classified: **652**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **32.1%** (209/652)
- Claude-emitted subset (94 faces) agreement with behavior modal: **60.6%** (57/94)
- Claude-emitted subset agreement with Claude-pilot-only modal (86 faces with pilot emit): **62.8%** (54/86)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | haiku_agree | acc |
|---|---:|---:|---:|
| HP | 106 | 41 | 38.7% |
| LP | 121 | 83 | 68.6% |
| HN-D | 54 | 4 | 7.4% |
| HN-S | 95 | 36 | 37.9% |
| LN | 90 | 35 | 38.9% |
| NB | 61 | 10 | 16.4% |

## Confusion matrix (rows = behavior modal, cols = haiku)

| | HP | LP | HN-D | HN-S | LN | NB | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HP** | 41 | 41 | 4 | 12 | 6 | 2 | 106 |
| **LP** | 14 | 83 | 2 | 5 | 9 | 8 | 121 |
| **HN-D** | 2 | 15 | 4 | 10 | 14 | 9 | 54 |
| **HN-S** | 8 | 11 | 6 | 36 | 20 | 14 | 95 |
| **LN** | 9 | 29 | 2 | 7 | 35 | 8 | 90 |
| **NB** | 13 | 28 | 1 | 6 | 3 | 10 | 61 |

_built 2026-05-05T06:06:39.932644+00:00_
