# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** full per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=2 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|---|
| gemma | 48% / 0.37 | 50% / 0.39 | 53% / 0.43 | 54% / 0.45 | **57% / 0.48** | k=all |
| gemma_intro_v7_primed | 49% / 0.39 | **54% / 0.44** | 52% / 0.42 | 52% / 0.42 | 51% / 0.41 | k=2 |
| gpt_oss_20b | 44% / 0.32 | 45% / 0.33 | 46% / 0.34 | **47% / 0.36** | 46% / 0.35 | k=5 |
| granite | 34% / 0.20 | 35% / 0.20 | 35% / 0.21 | **36% / 0.22** | 33% / 0.19 | k=5 |
| ministral | 33% / 0.19 | **35% / 0.22** | 34% / 0.20 | 33% / 0.20 | 30% / 0.17 | k=2 |
| qwen | 31% / 0.16 | 35% / 0.21 | **37% / 0.23** | 34% / 0.20 | 33% / 0.19 | k=3 |
| rinna_bilingual_4b | 20% / 0.02 | 21% / 0.04 | 20% / 0.02 | 20% / 0.03 | **23% / 0.06** | k=all |
| rinna_bilingual_4b_jp | 15% / -0.02 | 16% / -0.01 | 18% / 0.02 | **19% / 0.02** | 17% / 0.00 | k=5 |
| rinna_jp_3_6b | 12% / -0.07 | 16% / -0.02 | 13% / -0.05 | **18% / 0.01** | 17% / -0.01 | k=5 |
| rinna_jp_3_6b_jp | 21% / 0.06 | 23% / 0.07 | 23% / 0.07 | **24% / 0.08** | 23% / 0.06 | k=5 |

## Best-k per encoder

- **gemma**: no meaningful difference (baseline-all 57%, best 57%)
- **gemma_intro_v7_primed**: **+3.0pp lift** at k=2 (baseline-all 51%, best 54%)
- **gpt_oss_20b**: **+1.0pp lift** at k=5 (baseline-all 46%, best 47%)
- **granite**: **+3.0pp lift** at k=5 (baseline-all 33%, best 36%)
- **ministral**: **+4.9pp lift** at k=2 (baseline-all 30%, best 35%)
- **qwen**: **+4.4pp lift** at k=3 (baseline-all 33%, best 37%)
- **rinna_bilingual_4b**: no meaningful difference (baseline-all 23%, best 23%)
- **rinna_bilingual_4b_jp**: **+1.6pp lift** at k=5 (baseline-all 17%, best 19%)
- **rinna_jp_3_6b**: **+1.6pp lift** at k=5 (baseline-all 17%, best 18%)
- **rinna_jp_3_6b_jp**: **+1.0pp lift** at k=5 (baseline-all 23%, best 24%)
