# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** pilot per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=2 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|---|
| gemma | 64% / 0.57 | 62% / 0.54 | 70% / 0.63 | **72% / 0.65** | 72% / 0.65 | k=5 |
| gemma_intro_v7_primed | 62% / 0.54 | **70% / 0.63** | 68% / 0.61 | 68% / 0.61 | 70% / 0.63 | k=2 |
| gpt_oss_20b | 40% / 0.26 | 51% / 0.39 | 51% / 0.40 | 53% / 0.42 | **58% / 0.49** | k=all |
| granite | 38% / 0.23 | 40% / 0.26 | 42% / 0.28 | **45% / 0.33** | 43% / 0.31 | k=5 |
| ministral | **40% / 0.27** | 36% / 0.23 | 38% / 0.25 | 36% / 0.23 | 26% / 0.14 | k=1 |
| qwen | 34% / 0.17 | **45% / 0.32** | 45% / 0.31 | 45% / 0.31 | 36% / 0.21 | k=2 |
| rinna_bilingual_4b | 35% / 0.14 | **37% / 0.18** | 28% / 0.07 | 28% / 0.07 | 30% / 0.09 | k=2 |
| rinna_bilingual_4b_jp | 24% / 0.06 | 22% / 0.04 | **26% / 0.07** | 24% / 0.04 | 24% / 0.02 | k=3 |
| rinna_jp_3_6b | 17% / -0.03 | **24% / 0.05** | 15% / -0.02 | 20% / 0.04 | 13% / -0.04 | k=2 |
| rinna_jp_3_6b_jp | 24% / 0.04 | 26% / 0.07 | 28% / 0.09 | 33% / 0.14 | **37% / 0.18** | k=all |

## Best-k per encoder

- **gemma**: no meaningful difference (baseline-all 72%, best 72%)
- **gemma_intro_v7_primed**: no meaningful difference (baseline-all 70%, best 70%)
- **gpt_oss_20b**: no meaningful difference (baseline-all 58%, best 58%)
- **granite**: **+1.9pp lift** at k=5 (baseline-all 43%, best 45%)
- **ministral**: **+13.2pp lift** at k=1 (baseline-all 26%, best 40%)
- **qwen**: **+9.4pp lift** at k=2 (baseline-all 36%, best 45%)
- **rinna_bilingual_4b**: **+6.5pp lift** at k=2 (baseline-all 30%, best 37%)
- **rinna_bilingual_4b_jp**: **+2.2pp lift** at k=3 (baseline-all 24%, best 26%)
- **rinna_jp_3_6b**: **+10.9pp lift** at k=2 (baseline-all 13%, best 24%)
- **rinna_jp_3_6b_jp**: no meaningful difference (baseline-all 37%, best 37%)
