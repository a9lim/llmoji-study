# Cross-emit sanity check

**Ground-truth floor:** total_emit_count ≥ 3
**Encoders compared:** bol, gemma, gemma_intro_v7_primed, gpt_oss_20b, granite, haiku, ministral, opus, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_jp_3_6b, rinna_jp_3_6b_jp

## Partition counts (in GT subset)

| origin | n |
|---|---:|
| gemma_only | 30 |
| qwen_only | 43 |
| ministral_only | 23 |
| shared_2 | 16 |
| shared_3 | 9 |

## Accuracy by encoder × origin

Each cell: accuracy (n_correct/n) | κ.  **Bold cells** are cross-prediction (encoder predicting on faces only EMITTED by other v3 models).

| encoder | gemma_only | qwen_only | ministral_only | shared_2 | shared_3 |
|---|---|---|---|---|---|
| bol | 62% (5/8) | κ=0.52 | 29% (7/24) | κ=0.17 | 17% (1/6) | κ=0.03 | 40% (6/15) | κ=-0.05 | 38% (3/8) | κ=0.25 |
| gemma | 80% (24/30) | κ=0.76 | **70% (30/43) | κ=0.63** | **26% (6/23) | κ=0.13** | 56% (9/16) | κ=0.38 | 78% (7/9) | κ=0.71 |
| gemma_intro_v7_primed | 67% (20/30) | κ=0.59 | 60% (26/43) | κ=0.52 | 22% (5/23) | κ=0.08 | 50% (8/16) | κ=0.29 | 78% (7/9) | κ=0.71 |
| gpt_oss_20b | 43% (13/30) | κ=0.28 | 44% (19/43) | κ=0.33 | 30% (7/23) | κ=0.15 | 44% (7/16) | κ=0.29 | 78% (7/9) | κ=0.71 |
| granite | 30% (9/30) | κ=0.12 | 44% (19/43) | κ=0.31 | 22% (5/23) | κ=0.04 | 56% (9/16) | κ=0.36 | 33% (3/9) | κ=0.25 |
| haiku | 38% (3/8) | κ=0.17 | 42% (8/19) | κ=0.31 | 0% (0/1) | κ=0.00 | 60% (6/10) | κ=0.29 | 62% (5/8) | κ=0.50 |
| ministral | **30% (9/30) | κ=0.18** | **51% (22/43) | κ=0.40** | 30% (7/23) | κ=0.16 | 31% (5/16) | κ=0.08 | 22% (2/9) | κ=0.12 |
| opus | 50% (15/30) | κ=0.37 | 65% (28/43) | κ=0.58 | 26% (6/23) | κ=0.13 | 62% (10/16) | κ=0.40 | 78% (7/9) | κ=0.71 |
| qwen | **30% (9/30) | κ=0.12** | 35% (15/43) | κ=0.23 | **30% (7/23) | κ=0.14** | 38% (6/16) | κ=0.09 | 0% (0/9) | κ=-0.09 |
| rinna_bilingual_4b | 33% (10/30) | κ=0.17 | 21% (9/43) | κ=0.09 | 9% (2/23) | κ=-0.01 | 6% (1/16) | κ=-0.06 | 44% (4/9) | κ=0.00 |
| rinna_bilingual_4b_jp | 23% (7/30) | κ=0.08 | 16% (7/43) | κ=0.04 | 10% (2/21) | κ=0.00 | 6% (1/16) | κ=0.00 | 44% (4/9) | κ=0.06 |
| rinna_jp_3_6b | 13% (4/30) | κ=-0.02 | 16% (7/43) | κ=-0.02 | 26% (6/23) | κ=0.03 | 0% (0/16) | κ=-0.10 | 22% (2/9) | κ=0.02 |
| rinna_jp_3_6b_jp | 20% (6/30) | κ=0.05 | 26% (11/43) | κ=0.13 | 10% (2/21) | κ=-0.00 | 12% (2/16) | κ=-0.02 | 56% (5/9) | κ=0.29 |

## Headline cross-predictions

| encoder | origin | accuracy | κ | reading |
|---|---|---:|---:|---|
| gemma | qwen_only | 70% (30/43) | 0.63 | ✓ converging |
| qwen | gemma_only | 30% (9/30) | 0.12 | ~ ambiguous |
| gemma | ministral_only | 26% (6/23) | 0.13 | ✗ encoder-specific |
| ministral | gemma_only | 30% (9/30) | 0.18 | ~ ambiguous |
| qwen | ministral_only | 30% (7/23) | 0.14 | ~ ambiguous |
| ministral | qwen_only | 51% (22/43) | 0.40 | ✓ converging |

**Threshold heuristic** (per user's request): >50% =  encoders converge on shared intrinsic affect; <30% = the empirical-majority signal is too tied to the emitting model's sampling preference (would mean we need broader v3 coverage and/or a Claude-direct baseline).
