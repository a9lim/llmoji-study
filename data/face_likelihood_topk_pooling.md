# Top-k per-prompt pooling — face_likelihood

**GT floor:** total_emit_count ≥ 3
**Source:** full per-cell parquets

Each cell: accuracy / κ. Bold = best k for the encoder. **'all' = mean over all prompts** (current default in script 50 / 52 / 53).

| encoder | k=1 | k=2 | k=3 | k=5 | k=all | best |
|---|---|---|---|---|---|---|
| gemma | 63% / 0.54 | 65% / 0.56 | 66% / 0.58 | 68% / 0.60 | **72% / 0.66** | k=all |
| gemma_v7primed | **68% / 0.60** | 66% / 0.59 | 66% / 0.59 | 68% / 0.61 | 65% / 0.57 | k=1 |
| gpt_oss_20b | 42% / 0.29 | 45% / 0.33 | 48% / 0.37 | 49% / 0.38 | **51% / 0.40** | k=all |
| granite | 35% / 0.20 | 38% / 0.25 | 38% / 0.24 | **40% / 0.26** | 40% / 0.27 | k=5 |
| ministral | 38% / 0.24 | **42% / 0.28** | 38% / 0.25 | 42% / 0.28 | 38% / 0.24 | k=2 |
| qwen | 32% / 0.16 | 38% / 0.23 | **40% / 0.25** | 35% / 0.19 | 31% / 0.16 | k=3 |
| rinna_bilingual_4b | **22% / 0.05** | 20% / 0.03 | 20% / 0.04 | 20% / 0.04 | 20% / 0.04 | k=1 |
| rinna_bilingual_4b_jp | 17% / 0.04 | 17% / 0.04 | **18% / 0.05** | 18% / 0.05 | 17% / 0.03 | k=3 |
| rinna_bilingual_4b_jpfull | **25% / 0.13** | 25% / 0.13 | 20% / 0.08 | 23% / 0.13 | 15% / 0.06 | k=1 |
| rinna_bilingual_4b_jpfull30 | 26% / 0.15 | **34% / 0.23** | 29% / 0.18 | 25% / 0.12 | 25% / 0.12 | k=2 |
| rinna_jp_3_6b | **14% / -0.02** | 14% / -0.01 | 9% / -0.07 | 12% / -0.04 | 11% / -0.06 | k=1 |
| rinna_jp_3_6b_jp | 22% / 0.05 | 25% / 0.08 | **28% / 0.12** | 28% / 0.12 | 26% / 0.11 | k=3 |
| rinna_jp_3_6b_jpfull | 18% / 0.07 | 18% / 0.08 | 22% / 0.11 | **23% / 0.13** | 18% / 0.08 | k=5 |
| rinna_jp_3_6b_jpfull30 | 23% / 0.06 | **31% / 0.18** | 28% / 0.14 | 25% / 0.11 | 25% / 0.11 | k=2 |

## Best-k per encoder

- **gemma**: no meaningful difference (baseline-all 72%, best 72%)
- **gemma_v7primed**: **+3.1pp lift** at k=1 (baseline-all 65%, best 68%)
- **gpt_oss_20b**: no meaningful difference (baseline-all 51%, best 51%)
- **granite**: no meaningful difference (baseline-all 40%, best 40%)
- **ministral**: **+3.1pp lift** at k=2 (baseline-all 38%, best 42%)
- **qwen**: **+9.2pp lift** at k=3 (baseline-all 31%, best 40%)
- **rinna_bilingual_4b**: **+1.5pp lift** at k=1 (baseline-all 20%, best 22%)
- **rinna_bilingual_4b_jp**: **+1.5pp lift** at k=3 (baseline-all 17%, best 18%)
- **rinna_bilingual_4b_jpfull**: **+9.2pp lift** at k=1 (baseline-all 15%, best 25%)
- **rinna_bilingual_4b_jpfull30**: **+9.2pp lift** at k=2 (baseline-all 25%, best 34%)
- **rinna_jp_3_6b**: **+3.1pp lift** at k=1 (baseline-all 11%, best 14%)
- **rinna_jp_3_6b_jp**: **+1.5pp lift** at k=3 (baseline-all 26%, best 28%)
- **rinna_jp_3_6b_jpfull**: **+4.6pp lift** at k=5 (baseline-all 18%, best 23%)
- **rinna_jp_3_6b_jpfull30**: **+6.2pp lift** at k=2 (baseline-all 25%, best 31%)
