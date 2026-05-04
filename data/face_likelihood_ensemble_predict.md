# Ensemble per-face predictions

**Encoders:** gemma, ministral, qwen, gpt_oss_20b, granite (sources: {'gemma': 'full', 'ministral': 'full', 'qwen': 'full', 'gpt_oss_20b': 'full', 'granite': 'full'})
**Faces predicted:** 573
  - Full overlap (all encoders predict): 573
  - Partial overlap (subset of encoders): 0
  - With empirical metadata: 573

## Aggregate validation against empirical (total_emit_count ≥ 3, n=166)

- Ensemble accuracy: **51.8%** (86/166)

| empirical | n | correct | accuracy |
|---|---:|---:|---:|
| HP | 28 | 12 | 42.9% |
| LP | 30 | 12 | 40.0% |
| HN-D | 19 | 5 | 26.3% |
| HN-S | 28 | 11 | 39.3% |
| LN | 32 | 26 | 81.2% |
| NB | 29 | 20 | 69.0% |

## Predicted quadrant distribution (all faces)

| quadrant | n | share |
|---|---:|---:|
| HP | 63 | 11.0% |
| LP | 92 | 16.1% |
| HN-D | 40 | 7.0% |
| HN-S | 65 | 11.3% |
| LN | 176 | 30.7% |
| NB | 137 | 23.9% |
