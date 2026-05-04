# Ensemble per-face predictions

**Encoders:** gemma, gpt_oss_20b, granite, ministral, qwen, rinna_bilingual_4b_jpfull, rinna_jp_3_6b_jp, rinna_jp_3_6b_jpfull (sources: {'gemma': 'full', 'gpt_oss_20b': 'full', 'granite': 'full', 'ministral': 'full', 'qwen': 'full', 'rinna_bilingual_4b_jpfull': 'full', 'rinna_jp_3_6b_jp': 'full', 'rinna_jp_3_6b_jpfull': 'full'})
**Faces predicted:** 573
  - Full overlap (all encoders predict): 573
  - Partial overlap (subset of encoders): 0
  - With empirical metadata: 22

## Aggregate validation against empirical (total_emit_count ≥ 3, n=22)

- Ensemble accuracy: **72.7%** (16/22)

| empirical | n | correct | accuracy |
|---|---:|---:|---:|
| HP | 3 | 1 | 33.3% |
| LP | 3 | 3 | 100.0% |
| HN-D | 3 | 3 | 100.0% |
| HN-S | 4 | 2 | 50.0% |
| LN | 4 | 3 | 75.0% |
| NB | 5 | 4 | 80.0% |

## Predicted quadrant distribution (all faces)

| quadrant | n | share |
|---|---:|---:|
| HP | 127 | 22.2% |
| LP | 85 | 14.8% |
| HN-D | 48 | 8.4% |
| HN-S | 73 | 12.7% |
| LN | 117 | 20.4% |
| NB | 123 | 21.5% |
