# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** pilot per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|
| gemma | 62% / 0.55 | **69% / 0.62** | 65% / 0.57 | 65% / 0.57 | k=3 |
| gemma_v7primed | 60% / 0.52 | 65% / 0.57 | 65% / 0.57 | **67% / 0.59** | k=all |
| gpt_oss_20b | 35% / 0.22 | 44% / 0.32 | 46% / 0.34 | **50% / 0.40** | k=all |
| granite | 23% / 0.04 | 35% / 0.22 | **46% / 0.34** | 46% / 0.34 | k=5 |
| ministral | **38% / 0.24** | 33% / 0.19 | 31% / 0.17 | 23% / 0.09 | k=1 |
| qwen | 31% / 0.14 | **46% / 0.32** | 46% / 0.31 | 31% / 0.15 | k=3 |
| rinna_bilingual_4b | **33% / 0.13** | 28% / 0.09 | 26% / 0.06 | 28% / 0.09 | k=1 |
| rinna_bilingual_4b_jp | **23% / 0.07** | 23% / 0.05 | 21% / 0.02 | 21% / 0.02 | k=1 |
| rinna_bilingual_4b_jpfull | **35% / 0.23** | 33% / 0.20 | 30% / 0.18 | 21% / 0.11 | k=1 |
| rinna_bilingual_4b_jpfull30 | 21% / 0.05 | 28% / 0.14 | **35% / 0.20** | 35% / 0.20 | k=5 |
| rinna_jp_3_6b | 16% / -0.03 | 14% / -0.04 | **21% / 0.04** | 16% / -0.03 | k=5 |
| rinna_jp_3_6b_jp | 26% / 0.07 | 30% / 0.12 | **35% / 0.19** | 35% / 0.17 | k=5 |
| rinna_jp_3_6b_jpfull | 26% / 0.12 | 30% / 0.17 | 35% / 0.24 | **37% / 0.27** | k=all |
| rinna_jp_3_6b_jpfull30 | 30% / 0.13 | 35% / 0.21 | **40% / 0.27** | 40% / 0.27 | k=5 |

## Best-k per encoder

- **gemma**: **+4.2pp lift** at k=3 (baseline-all 65%, best 69%)
- **gemma_v7primed**: no meaningful difference (baseline-all 67%, best 67%)
- **gpt_oss_20b**: no meaningful difference (baseline-all 50%, best 50%)
- **granite**: no meaningful difference (baseline-all 46%, best 46%)
- **ministral**: **+14.6pp lift** at k=1 (baseline-all 23%, best 38%)
- **qwen**: **+14.6pp lift** at k=3 (baseline-all 31%, best 46%)
- **rinna_bilingual_4b**: **+4.7pp lift** at k=1 (baseline-all 28%, best 33%)
- **rinna_bilingual_4b_jp**: **+2.3pp lift** at k=1 (baseline-all 21%, best 23%)
- **rinna_bilingual_4b_jpfull**: **+14.0pp lift** at k=1 (baseline-all 21%, best 35%)
- **rinna_bilingual_4b_jpfull30**: no meaningful difference (baseline-all 35%, best 35%)
- **rinna_jp_3_6b**: **+4.7pp lift** at k=5 (baseline-all 16%, best 21%)
- **rinna_jp_3_6b_jp**: no meaningful difference (baseline-all 35%, best 35%)
- **rinna_jp_3_6b_jpfull**: no meaningful difference (baseline-all 37%, best 37%)
- **rinna_jp_3_6b_jpfull30**: no meaningful difference (baseline-all 40%, best 40%)
