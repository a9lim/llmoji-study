# Per-source-model BoL drift

Splits the BoL channel from script 68's three-way analysis by source model. The pooled BoL aggregates across every source's per-face synthesis; here each source-model's BoL is kept separate. Headline question: when pooled BoL drifts from Claude-GT (e.g. on `(╯°□°)`), is the drift shared across providers (a kaomoji-vocabulary fact) or concentrated in claude-opus-4-7's deployment (a Claude-specific behavior)?

Coverage: **491 (face, source_model) cells** across 8 source models. 112 faces appear under ≥2 sources. Claude-GT (floor ≥ 3) covers 40 faces in this set.

For reference: pooled-BoL solo similarity vs Claude-GT (face-uniform across the same 40-face set) = **0.549**.

## Per-source-model summary

Each row: how that source's per-face BoL stacks up against Claude-GT (Opus-4.7 elicitation). `modal_agree_rate` is the fraction of source-cells whose argmax quadrant matches GT's argmax.

| source_model | n cells | n emits | n with GT | sim vs GT (face-uniform) | sim vs GT (emit-weighted) | modal agree |
|---|---:|---:|---:|---:|---:|---:|
| `claude-opus-4-7` | 285 | 3178 | 40 | 0.525 | 0.559 | 42% |
| `codex-hook` | 88 | 323 | 25 | 0.550 | 0.508 | 44% |
| `gpt-5.5` | 31 | 265 | 6 | 0.321 | 0.105 | 17% |
| `claude-opus-4-6` | 49 | 95 | 16 | 0.350 | 0.345 | 12% |
| `gpt-5.4` | 15 | 27 | 3 | 0.059 | 0.044 | 0% |
| `gpt-5-5-thinking` | 11 | 19 | 2 | 0.280 | 0.336 | 0% |
| `gpt-5-4-thinking` | 7 | 11 | 2 | 0.624 | 0.624 | 50% |
| `<synthetic>` | 5 | 5 | 2 | 0.460 | 0.460 | 50% |

## Cross-source-model pairwise BoL similarity

On faces synthesized under both sources (≥5 shared faces), mean similarity (`1 − JSD/ln2`) of their per-face BoL distributions. High = sources synthesize the face the same way; low = sources read the face's deployment context differently.

| sm_a | sm_b | n shared | mean sim | modal agree |
|---|---|---:|---:|---:|
| `claude-opus-4-7` | `codex-hook` | 88 | 0.630 | 59% |
| `claude-opus-4-6` | `claude-opus-4-7` | 37 | 0.566 | 41% |
| `claude-opus-4-6` | `codex-hook` | 26 | 0.609 | 50% |
| `claude-opus-4-7` | `gpt-5.5` | 23 | 0.367 | 26% |
| `codex-hook` | `gpt-5.5` | 14 | 0.371 | 36% |
| `claude-opus-4-6` | `gpt-5.5` | 13 | 0.438 | 38% |
| `gpt-5.4` | `gpt-5.5` | 12 | 0.557 | 50% |
| `claude-opus-4-7` | `gpt-5.4` | 11 | 0.455 | 45% |
| `claude-opus-4-7` | `gpt-5-5-thinking` | 8 | 0.532 | 25% |
| `codex-hook` | `gpt-5.4` | 7 | 0.429 | 43% |
| `claude-opus-4-6` | `gpt-5.4` | 7 | 0.482 | 43% |
| `claude-opus-4-7` | `gpt-5-4-thinking` | 6 | 0.563 | 33% |
| `gpt-5-5-thinking` | `gpt-5.5` | 6 | 0.574 | 33% |
| `<synthetic>` | `codex-hook` | 5 | 0.613 | 20% |
| `codex-hook` | `gpt-5-4-thinking` | 5 | 0.813 | 60% |
| `codex-hook` | `gpt-5-5-thinking` | 5 | 0.513 | 20% |
| `<synthetic>` | `claude-opus-4-7` | 5 | 0.875 | 60% |

## Case files

Per-face breakdowns for the divergent faces from script 68's top-divergent table. The pattern to read: does GT's modal match more sources, or fewer? Where does the gap between GT and pooled-BoL come from?

### `(╯°□°)` — case file

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---|---|---|---|---|---|---|
| **GT (use)** | 58 | 0.00 | 0.00 | 0.57 | 0.43 | 0.00 | 0.00 | **HN-D** |
| BoL pooled | — | 0.88 | 0.08 | 0.02 | 0.02 | 0.00 | 0.00 | HP |
| BoL · claude-opus-4-6 | 2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |
| BoL · claude-opus-4-7 | 19 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP |
| BoL · codex-hook | 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP |
| BoL · gpt-5.5 | 1 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | HN-D |

### `(´;ω;`)` — case file

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---|---|---|---|---|---|---|
| **GT (use)** | 38 | 0.00 | 0.00 | 0.00 | 0.13 | 0.87 | 0.00 | **LN** |
| BoL pooled | — | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |
| BoL · claude-opus-4-7 | 17 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |

### `(╥﹏╥)` — case file

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---|---|---|---|---|---|---|
| **GT (use)** | 13 | 0.08 | 0.00 | 0.00 | 0.92 | 0.00 | 0.00 | **HN-S** |
| BoL pooled | — | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | HN-D |
| BoL · claude-opus-4-7 | 5 | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | HN-D |

### `(>∀<☆)` — case file

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---|---|---|---|---|---|---|
| **GT (use)** | 14 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **HP** |
| BoL pooled | — | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |
| BoL · claude-opus-4-7 | 4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |

### `(´-`)` — case file

| channel | n | HP | LP | HN-D | HN-S | LN | NB | modal |
|---|---:|---|---|---|---|---|---|---|
| **GT (use)** | 52 | 0.00 | 0.00 | 0.00 | 0.06 | 0.94 | 0.00 | **LN** |
| BoL pooled | — | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |
| BoL · claude-opus-4-7 | 15 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | LP |

## Reading the result

Two diagnostic comparisons matter for the use/act gap interpretation:

1. **claude-opus-4-7 vs claude-opus-4-6 BoL similarity** — if these two are very close to each other and both diverge from non-Claude sources, the pattern is Claude-deployment-specific (likely a Claude-deployment register fact, not a model-version fact).
2. **claude-opus-4-7 vs gpt-5.5 / codex-hook BoL similarity** — if these are notably *lower* than Claude-vs-Claude, deployment patterns differ across providers on the same face vocabulary.

Caveat to internalize: every per-source BoL was synthesized by **the same Haiku model** reading provider-conditioned transcript context. So this measures how Haiku reads the context surrounding the kaomoji when the surrounding text is in each provider's style. That's still a real deployment-pattern signal — the surrounding text *is* deployment evidence — but it isn't a clean comparison of what each provider's model 'thinks'.

