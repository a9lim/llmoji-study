# Haiku face→quadrant judgment vs behavior modal

- Model: `claude-haiku-4-5`
- Faces classified: **573**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **29.8%** (171/573)
- Claude-emitted subset (51 faces) agreement with behavior modal: **58.8%** (30/51)
- Claude-emitted subset agreement with Claude-pilot-only modal (51 faces with pilot emit): **58.8%** (30/51)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | haiku_agree | acc |
|---|---:|---:|---:|
| HP | 88 | 35 | 39.8% |
| LP | 98 | 63 | 64.3% |
| HN-D | 47 | 3 | 6.4% |
| HN-S | 80 | 30 | 37.5% |
| LN | 81 | 31 | 38.3% |
| NB | 54 | 9 | 16.7% |

## Confusion matrix (rows = behavior modal, cols = haiku)

| | HP | LP | HN-D | HN-S | LN | NB | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HP** | 35 | 32 | 3 | 10 | 6 | 2 | 88 |
| **LP** | 14 | 63 | 2 | 5 | 8 | 6 | 98 |
| **HN-D** | 2 | 15 | 3 | 9 | 11 | 7 | 47 |
| **HN-S** | 7 | 10 | 5 | 30 | 17 | 11 | 80 |
| **LN** | 6 | 27 | 2 | 7 | 31 | 8 | 81 |
| **NB** | 11 | 25 | 1 | 6 | 2 | 9 | 54 |

_built 2026-05-04T21:40:56.035961+00:00_
