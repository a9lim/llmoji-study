# Opus face→quadrant judgment vs behavior modal

- Model: `claude-opus-4-7`  (shortname: `opus`)
- Scope: `--gt-only` (Claude-GT subset, floor=1)
- Faces classified: **128**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **60.9%** (78/128)
- Claude-emitted subset (128 faces) agreement with behavior modal: **60.9%** (78/128)
- Claude-emitted subset agreement with Claude-pilot-only modal (116 faces with pilot emit): **57.8%** (67/116)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | opus_agree | acc |
|---|---:|---:|---:|
| HP | 26 | 13 | 50.0% |
| LP | 33 | 31 | 93.9% |
| HN-D | 12 | 6 | 50.0% |
| HN-S | 20 | 12 | 60.0% |
| LN | 18 | 10 | 55.6% |
| NB | 19 | 6 | 31.6% |

## Confusion matrix (rows = behavior modal, cols = opus)

| | HP | LP | HN-D | HN-S | LN | NB | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HP** | 13 | 10 | 1 | 2 | 0 | 0 | 26 |
| **LP** | 0 | 31 | 0 | 0 | 1 | 1 | 33 |
| **HN-D** | 0 | 0 | 6 | 1 | 5 | 0 | 12 |
| **HN-S** | 0 | 1 | 2 | 12 | 5 | 0 | 20 |
| **LN** | 0 | 2 | 0 | 3 | 10 | 3 | 18 |
| **NB** | 0 | 8 | 1 | 1 | 3 | 6 | 19 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 6-quadrant softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **57** (judged ∩ GT-with-≥3-emits).

- **Face-uniform** mean similarity (vocabulary view): **0.752**
- **Emit-weighted** mean similarity (deployment view, weight = GT emit count): **0.798**  (total emit weight: 887)

### Per-quadrant similarity (faces grouped by GT modal Q)

| GT modal | n | mean similarity |
|---|---:|---:|
| HP | 13 | 0.683 |
| LP | 16 | 0.841 |
| HN-D | 4 | 0.881 |
| HN-S | 7 | 0.667 |
| LN | 9 | 0.753 |
| NB | 8 | 0.698 |


## Head-to-head: Opus vs Haiku

On the 128 face(s) both models rated:

- **Hard agreement (argmax-vs-argmax)**: opus ↔ haiku = **91/128 (71.1%)**
- **Soft agreement (distributional similarity, face-uniform)**: mean similarity(opus, haiku) = **0.891**
- **Hard accuracy vs Claude-pilot modal** (n=116): opus **67/116 (57.8%)**, haiku **63/116 (54.3%)**
- **Soft accuracy vs Claude-GT distribution** (n=57 faces with ≥3 GT emits, total weight 887):
  - opus: face-uniform **0.752**, emit-weighted **0.798**
  - haiku: face-uniform **0.701**, emit-weighted **0.726**

### Disagreements (first 30 of 37)

| face | opus | haiku | claude-pilot modal |
|---|---|---|---|
| `(;´д`)` | LN | HN-S | HN-S |
| `(;´ヮ`)` | HN-S | LP | LN |
| `(;╹⌓╹)` | HN-S | LP | LN |
| `(;・∀・)` | HN-S | HP | HN-S |
| `(^ω^)` | LP | HP | HP |
| `(^‿^)` | LP | HP | NB |
| `(^▽^)` | LP | HP | HP |
| `(`ε´)` | HN-D | HN-S | — |
| `(`・ω・´)` | NB | LP | NB |
| `(¬_¬)` | HN-D | NB | NB |
| `(¯―¯٥)` | LN | LP | NB |
| `(°▽°)` | LP | HP | HP |
| `(´-`)` | NB | LN | LN |
| `(´-ω-`)` | LN | LP | LN |
| `(´-﹏-`;)` | HN-S | LN | LN |
| `(´;ω;`)` | LN | HN-S | LN |
| `(´~`)` | LN | LP | NB |
| `(´°̥̥̥̥̥̥` | LN | HN-S | HN-D |
| `(´∀`)` | LP | HP | LP |
| `(´▽`)` | LP | HP | LP |
| `(´・̥̥̥ω・̥̥` | LN | LP | LN |
| `(´・̥ω・̥`)` | LN | LP | LN |
| `(´・ω・`)` | LN | LP | NB |
| `(‿‿‿)` | LN | LP | — |
| `(╬Ò﹏Ó)` | HN-D | HN-S | HN-S |
| `(╬⊙△⊙)` | HN-D | HP | HN-S |
| `(☀´▽`)` | LP | HP | NB |
| `(♡˃͈̑‿` | LP | HP | HP |
| `(・_・)` | NB | LN | NB |
| `(・_・ヾ` | NB | LN | NB |

_built 2026-05-05T19:04:59.207821+00:00_
