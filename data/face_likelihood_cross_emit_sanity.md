# Cross-emit sanity check

**Ground-truth floor:** total_emit_count ≥ 3
**Encoders compared:** gemma, gpt_oss_20b, granite, ministral, qwen

## Partition counts (in GT subset)

| origin | n |
|---|---:|
| gemma_only | 14 |
| qwen_only | 24 |
| ministral_only | 2 |
| shared_2 | 18 |
| shared_3 | 8 |

## Accuracy by encoder × origin

Each cell: accuracy (n_correct/n) | κ.  **Bold cells** are cross-prediction (encoder predicting on faces only EMITTED by other v3 models).

| encoder | gemma_only | qwen_only | ministral_only | shared_2 | shared_3 |
|---|---|---|---|---|---|
| gemma | 86% (12/14) | κ=0.82 | **67% (16/24) | κ=0.57** | **100% (2/2) | κ=1.00** | 59% (10/17) | κ=0.48 | 88% (7/8) | κ=0.84 |
| gpt_oss_20b | 43% (6/14) | κ=0.27 | 33% (8/24) | κ=0.21 | 100% (2/2) | κ=1.00 | 65% (11/17) | κ=0.55 | 75% (6/8) | κ=0.68 |
| granite | 43% (6/14) | κ=0.25 | 33% (8/24) | κ=0.19 | 100% (2/2) | κ=1.00 | 41% (7/17) | κ=0.27 | 38% (3/8) | κ=0.22 |
| ministral | **29% (4/14) | κ=0.08** | **42% (10/24) | κ=0.30** | 100% (2/2) | κ=1.00 | 41% (7/17) | κ=0.21 | 25% (2/8) | κ=0.11 |
| qwen | **29% (4/14) | κ=0.10** | 42% (10/24) | κ=0.31 | **0% (0/2) | κ=0.00** | 24% (4/17) | κ=0.12 | 25% (2/8) | κ=0.08 |

## Headline cross-predictions

| encoder | origin | accuracy | κ | reading |
|---|---|---:|---:|---|
| gemma | qwen_only | 67% (16/24) | 0.57 | ✓ converging |
| qwen | gemma_only | 29% (4/14) | 0.10 | ✗ encoder-specific |
| gemma | ministral_only | 100% (2/2) | 1.00 | ✓ converging |
| ministral | gemma_only | 29% (4/14) | 0.08 | ✗ encoder-specific |
| qwen | ministral_only | 0% (0/2) | 0.00 | ✗ encoder-specific |
| ministral | qwen_only | 42% (10/24) | 0.30 | ~ ambiguous |

**Threshold heuristic** (per user's request): >50% =  encoders converge on shared intrinsic affect; <30% = the empirical-majority signal is too tied to the emitting model's sampling preference (would mean we need broader v3 coverage and/or a Claude-direct baseline).
