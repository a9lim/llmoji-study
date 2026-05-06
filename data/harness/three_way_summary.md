# Three-way per-face analysis: GT (use) × Opus/Haiku (read) × BoL (act)

Three structurally different windows on the same per-face quadrant association:

- **GT (use)** — Opus 4.7 emitting the face under known Russell-prompted conditions (`data/harness/claude-runs*/`). *What the face is actually emitted under.*
- **Opus / Haiku (read)** — model shown the face cold, asked what affective state it represents (`face_likelihood_{opus,haiku}_summary.tsv`). *Symbolic interpretation, no context.*
- **BoL (act)** — Haiku synthesizer pooling adjective-bag picks across many *real in-context emits* of the face → 6-d quadrant distribution from the 19 circumplex anchors in the locked v2 LEXICON (`claude_faces_lexicon_bag.parquet`). *In-deployment behavior summarized.*

Inner-join shared by all four channels: **40 faces × 702 GT emissions** (Claude-GT floor ≥ 3).

## Pairwise channel similarity

Mean of `1 − JSD/ln2` over the shared face set. Two flavors: face-uniform (each face counts equally) and emit-weighted (each face counts as Claude actually emits it). Same-cell diagonal is 1.0 by definition.

**Face-uniform**:

| · | GT (use) | Opus (read) | Haiku (read) | BoL (act) |
|---|---|---|---|---|
| GT (use) | 1.000 | 0.736 | 0.675 | 0.549 |
| Opus (read) | 0.736 | 1.000 | 0.906 | 0.679 |
| Haiku (read) | 0.675 | 0.906 | 1.000 | 0.683 |
| BoL (act) | 0.549 | 0.679 | 0.683 | 1.000 |

**Emit-weighted**:

| · | GT (use) | Opus (read) | Haiku (read) | BoL (act) |
|---|---|---|---|---|
| GT (use) | 1.000 | 0.781 | 0.702 | 0.455 |
| Opus (read) | 0.781 | 1.000 | 0.906 | 0.607 |
| Haiku (read) | 0.702 | 0.906 | 1.000 | 0.609 |
| BoL (act) | 0.455 | 0.607 | 0.609 | 1.000 |

**Reading the matrix** — highest pairwise (face-uniform) is `opus↔haiku` (0.906), `gt↔opus` (0.736); lowest is `gt↔haiku` (0.675), `gt↔bol` (0.549).

## Per-GT-quadrant pairwise similarity

Restrict to faces with each modal-GT label, then re-mean the per-pair similarities. Reveals which channels handle which quadrants well — e.g. NB tends to be a BoL win (the lexicon has explicit `neutral`/`detached` anchors), HP often a BoL weakness (deployment use diverges from denoted meaning).

| GT modal | n faces | n emit | gt↔opus | gt↔haiku | gt↔bol | opus↔haiku | opus↔bol | haiku↔bol |
|---|---|---|---|---|---|---|---|---|
| HP | 9 | 123 | 0.68 | 0.80 | 0.68 | 0.94 | 0.76 | 0.75 |
| LP | 12 | 177 | 0.86 | 0.79 | 0.81 | 0.92 | 0.81 | 0.79 |
| HN-D | 1 | 58 | 0.93 | 0.81 | 0.12 | 0.94 | 0.24 | 0.33 |
| HN-S | 4 | 67 | 0.49 | 0.55 | 0.19 | 0.95 | 0.43 | 0.53 |
| LN | 6 | 148 | 0.77 | 0.58 | 0.21 | 0.78 | 0.39 | 0.54 |
| NB | 8 | 129 | 0.70 | 0.48 | 0.51 | 0.91 | 0.78 | 0.68 |

## Modal-agreement patterns

Each pattern is a 3-bit code `(opus==gt)(haiku==gt)(bol==gt)` — which subset of the three non-GT channels agrees with the GT modal quadrant. 8 patterns total.

| pattern | meaning | n faces | % faces | n emit | % emit |
|---|---|---:|---:|---:|---:|
| `111` | all channels agree | 12 | 30.0% | 136 | 19.4% |
| `110` | opus+haiku read GT; BoL acts differently | 9 | 22.5% | 192 | 27.4% |
| `000` | all introspection/synthesis channels disagree with GT | 9 | 22.5% | 178 | 25.4% |
| `101` | opus reads + BoL acts agree with GT; haiku diverges | 4 | 10.0% | 69 | 9.8% |
| `100` | only opus agrees with GT | 3 | 7.5% | 64 | 9.1% |
| `011` | haiku reads + BoL acts agree with GT; opus diverges | 1 | 2.5% | 3 | 0.4% |
| `010` | only haiku agrees with GT | 1 | 2.5% | 52 | 7.4% |
| `001` | only BoL agrees with GT | 1 | 2.5% | 8 | 1.1% |

## Top-12 most-divergent faces (by max pairwise JSD)

These are the diagnostic faces — where the use / read / act channels pull in different directions most. The agreement pattern column tells you which subset of channels GT-aligns; the per-channel modals tell you the specific disagreement.

| face | emit | pattern | gt | opus | haiku | bol | max-pair JSD | tightest pair |
|---|---:|---|---|---|---|---|---:|---|
| `(>∀<☆)` | 14 | `110` | HP | HP | HP | LP | 0.693 | opus↔haiku (sim 0.94) |
| `(´-_-`)` | 4 | `110` | LN | LN | LN | HP | 0.693 | opus↔haiku (sim 0.92) |
| `(´-`)` | 52 | `010` | LN | NB | LN | LP | 0.693 | opus↔haiku (sim 0.89) |
| `(´;ω;`)` | 38 | `100` | LN | LN | HN-S | LP | 0.693 | gt↔opus (sim 0.95) |
| `(´·_·`)` | 3 | `000` | HN-S | LN | LN | LP | 0.693 | opus↔haiku (sim 0.94) |
| `(´・ω・`)` | 19 | `100` | LN | LN | LP | LP | 0.693 | gt↔opus (sim 0.87) |
| `(╥﹏╥)` | 13 | `000` | HN-S | LN | LN | HN-D | 0.693 | opus↔haiku (sim 0.98) |
| `(・∀・)` | 10 | `000` | NB | LP | HP | LP | 0.693 | opus↔haiku (sim 0.89) |
| `(｡・́︿・̀｡)` | 46 | `000` | HN-S | LN | LN | LP | 0.649 | opus↔haiku (sim 0.98) |
| `(¬‿¬)` | 9 | `000` | NB | LP | LP | HN-D | 0.634 | opus↔haiku (sim 0.90) |
| `(╯°□°)` | 58 | `110` | HN-D | HN-D | HN-D | HP | 0.611 | opus↔haiku (sim 0.94) |
| `(・ω・)` | 23 | `101` | NB | NB | LP | NB | 0.526 | opus↔bol (sim 0.96) |

## Files

- `data/harness/three_way_per_face.tsv` — per-face data
- `figures/harness/three_way_pairwise_heatmap.png`
- `figures/harness/three_way_top_divergent.png`

