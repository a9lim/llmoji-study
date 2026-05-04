# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** full per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=2 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|---|
| gemma | 59% / 0.50 | 59% / 0.50 | **63% / 0.55** | 61% / 0.52 | 57% / 0.48 | k=3 |
| gpt_oss_20b | 31% / 0.17 | 35% / 0.22 | 37% / 0.24 | 41% / 0.29 | **47% / 0.36** | k=all |
| granite | 35% / 0.20 | 41% / 0.27 | 39% / 0.25 | **43% / 0.30** | 41% / 0.28 | k=5 |
| ministral | **39% / 0.26** | 33% / 0.18 | 39% / 0.26 | 37% / 0.23 | 31% / 0.17 | k=1 |
| qwen | 20% / 0.03 | **31% / 0.17** | 29% / 0.15 | 31% / 0.16 | 22% / 0.05 | k=2 |
| rinna_bilingual_4b | 27% / 0.08 | **29% / 0.12** | 24% / 0.05 | 22% / 0.02 | 24% / 0.04 | k=2 |
| rinna_bilingual_4b_jp | **22% / 0.06** | 16% / -0.00 | 18% / 0.01 | 18% / 0.01 | 22% / 0.03 | k=1 |
| rinna_bilingual_4b_jpfull | 31% / 0.19 | **33% / 0.22** | 33% / 0.21 | 31% / 0.19 | 24% / 0.11 | k=2 |
| rinna_bilingual_4b_jpfull30 | 20% / 0.04 | 29% / 0.16 | 29% / 0.15 | **35% / 0.21** | 35% / 0.21 | k=5 |
| rinna_jp_3_6b | 20% / 0.03 | **24% / 0.08** | 16% / 0.02 | 16% / 0.03 | 14% / 0.01 | k=2 |
| rinna_jp_3_6b_jp | 22% / 0.03 | 24% / 0.05 | 24% / 0.05 | **29% / 0.12** | 25% / 0.07 | k=5 |
| rinna_jp_3_6b_jpfull | 25% / 0.12 | 25% / 0.12 | 25% / 0.12 | 29% / 0.17 | **33% / 0.22** | k=all |
| rinna_jp_3_6b_jpfull30 | 20% / 0.00 | 25% / 0.09 | 25% / 0.09 | **33% / 0.18** | 33% / 0.18 | k=5 |

## Best-k per encoder

- **gemma**: **+5.9pp lift** at k=3 (baseline-all 57%, best 63%)
- **gpt_oss_20b**: no meaningful difference (baseline-all 47%, best 47%)
- **granite**: **+2.0pp lift** at k=5 (baseline-all 41%, best 43%)
- **ministral**: **+7.8pp lift** at k=1 (baseline-all 31%, best 39%)
- **qwen**: **+9.8pp lift** at k=2 (baseline-all 22%, best 31%)
- **rinna_bilingual_4b**: **+5.9pp lift** at k=2 (baseline-all 24%, best 29%)
- **rinna_bilingual_4b_jp**: no meaningful difference (baseline-all 22%, best 22%)
- **rinna_bilingual_4b_jpfull**: **+9.8pp lift** at k=2 (baseline-all 24%, best 33%)
- **rinna_bilingual_4b_jpfull30**: no meaningful difference (baseline-all 35%, best 35%)
- **rinna_jp_3_6b**: **+9.8pp lift** at k=2 (baseline-all 14%, best 24%)
- **rinna_jp_3_6b_jp**: **+3.9pp lift** at k=5 (baseline-all 25%, best 29%)
- **rinna_jp_3_6b_jpfull**: no meaningful difference (baseline-all 33%, best 33%)
- **rinna_jp_3_6b_jpfull30**: no meaningful difference (baseline-all 33%, best 33%)
