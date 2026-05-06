# Opus face→quadrant judgment vs behavior modal

- Model: `claude-opus-4-7`  (shortname: `opus`)
- Scope: full v3 face union
- Faces classified: **684**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **31.1%** (213/684)
- Claude-emitted subset (128 faces) agreement with behavior modal: **60.9%** (78/128)
- Claude-emitted subset agreement with Claude-pilot-only modal (116 faces with pilot emit): **57.8%** (67/116)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | opus_agree | acc |
|---|---:|---:|---:|
| HP | 109 | 30 | 27.5% |
| LP | 119 | 87 | 73.1% |
| HN-D | 54 | 13 | 24.1% |
| HN-S | 89 | 36 | 40.4% |
| LN | 80 | 35 | 43.8% |
| NB | 60 | 12 | 20.0% |

## Confusion matrix (rows = behavior modal, cols = opus)

| | HP | LP | HN-D | HN-S | LN | NB | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HP** | 30 | 53 | 7 | 10 | 8 | 1 | 109 |
| **LP** | 9 | 87 | 2 | 5 | 7 | 9 | 119 |
| **HN-D** | 0 | 17 | 13 | 9 | 12 | 3 | 54 |
| **HN-S** | 4 | 15 | 8 | 36 | 18 | 8 | 89 |
| **LN** | 2 | 25 | 1 | 9 | 35 | 8 | 80 |
| **NB** | 2 | 34 | 4 | 5 | 3 | 12 | 60 |

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

_built 2026-05-06T08:24:05.455816+00:00_
