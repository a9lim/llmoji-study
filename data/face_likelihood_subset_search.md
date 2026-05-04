# Face_likelihood — exhaustive subset search

**Encoders:** 14  (gemma, gpt_oss_20b, granite, haiku, ministral, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_bilingual_4b_jpfull, rinna_bilingual_4b_jpfull30, rinna_jp_3_6b, rinna_jp_3_6b_jp, rinna_jp_3_6b_jpfull, rinna_jp_3_6b_jpfull30)
**Faces (overlap):** 572
**GT subset (≥3 emits, pooled v3+Claude+wild):** 166
**Subsets evaluated:** 16383

## Headline

- Best single encoder by accuracy: **gemma** at 51.8% (86/166); Cohen's κ = 0.418
- Best weighted-vote subset by accuracy: **{gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b}** at **56.0%** (93/166) — size 5, +4.2pp over best single; κ = 0.466
- Best weighted-vote subset by κ: **{gemma,gpt_oss_20b,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jpfull}** at κ = **0.469** (accuracy 56.0%, size 6)
- Best strict-majority subset: **{gemma}** at 51.8% on 166 resolved (abstains on 0 all-distinct); κ = 0.418

**Reading κ:** Cohen's kappa corrects agreement for chance. 0.0 = no signal beyond random, 0.4–0.6 = moderate, 0.6–0.8 = substantial, >0.8 = near-perfect. Penalizes encoders that always predict the majority class — useful given GLM's 100%-LN bias. Voting models often have lower κ than accuracy because the vote concentrates predictions on common quadrants.

## Per-encoder solo accuracy + Cohen's κ vs empirical

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 51.8% (86/166) | 0.418 |
| gpt_oss_20b | 44.6% (74/166) | 0.334 |
| haiku | 40.4% (67/166) | 0.276 |
| granite | 34.9% (58/166) | 0.205 |
| ministral | 31.9% (52/166) | 0.178 |
| qwen | 30.1% (50/166) | 0.167 |
| rinna_jp_3_6b_jpfull30 | 25.9% (43/166) | 0.105 |
| rinna_bilingual_4b_jpfull30 | 24.1% (40/166) | 0.100 |
| rinna_bilingual_4b | 22.3% (37/166) | 0.063 |
| rinna_jp_3_6b_jp | 21.1% (35/166) | 0.049 |
| rinna_jp_3_6b_jpfull | 21.1% (35/166) | 0.070 |
| rinna_bilingual_4b_jpfull | 19.3% (32/166) | 0.064 |
| rinna_bilingual_4b_jp | 16.9% (27/166) | 0.000 |
| rinna_jp_3_6b | 16.3% (27/166) | -0.007 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Useful for ensemble design: encoder pairs with low κ make complementary errors and are more useful to combine than encoder pairs with high κ.

| pair | κ |
|---|---:|
| rinna_bilingual_4b_jpfull ↔ rinna_bilingual_4b_jpfull30 | 0.493 |
| rinna_jp_3_6b_jpfull ↔ rinna_jp_3_6b_jpfull30 | 0.322 |
| gemma ↔ haiku | 0.297 |
| gpt_oss_20b ↔ haiku | 0.261 |
| gemma ↔ gpt_oss_20b | 0.249 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.223 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jp | 0.191 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull | 0.189 |
| granite ↔ ministral | 0.184 |
| granite ↔ haiku | 0.153 |
| gpt_oss_20b ↔ granite | 0.148 |
| gemma ↔ ministral | 0.141 |
| haiku ↔ ministral | 0.129 |
| gpt_oss_20b ↔ ministral | 0.125 |
| gemma ↔ granite | 0.124 |
| gpt_oss_20b ↔ qwen | 0.112 |
| haiku ↔ qwen | 0.106 |
| qwen ↔ rinna_bilingual_4b_jpfull | 0.102 |
| haiku ↔ rinna_jp_3_6b_jpfull30 | 0.096 |
| gemma ↔ qwen | 0.089 |
| ministral ↔ qwen | 0.087 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull | 0.078 |
| gemma ↔ rinna_jp_3_6b_jpfull30 | 0.076 |
| granite ↔ rinna_jp_3_6b_jpfull | 0.076 |
| haiku ↔ rinna_jp_3_6b_jpfull | 0.071 |
| haiku ↔ rinna_bilingual_4b_jpfull30 | 0.071 |
| granite ↔ rinna_jp_3_6b_jpfull30 | 0.069 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull | 0.065 |
| granite ↔ qwen | 0.064 |
| granite ↔ rinna_bilingual_4b_jpfull30 | 0.064 |
| haiku ↔ rinna_bilingual_4b_jpfull | 0.063 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull30 | 0.062 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jp | 0.059 |
| qwen ↔ rinna_bilingual_4b_jpfull30 | 0.059 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jp | 0.058 |
| gemma ↔ rinna_bilingual_4b_jpfull30 | 0.055 |
| haiku ↔ rinna_bilingual_4b | 0.054 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull30 | 0.050 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull | 0.050 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.048 |
| ministral ↔ rinna_jp_3_6b_jpfull30 | 0.046 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull30 | 0.045 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jp | 0.045 |
| gemma ↔ rinna_jp_3_6b_jp | 0.043 |
| gemma ↔ rinna_bilingual_4b_jpfull | 0.043 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull30 | 0.042 |
| haiku ↔ rinna_jp_3_6b_jp | 0.040 |
| gemma ↔ rinna_bilingual_4b | 0.039 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull | 0.035 |
| haiku ↔ rinna_bilingual_4b_jp | 0.035 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull | 0.034 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull30 | 0.034 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jp | 0.033 |
| qwen ↔ rinna_jp_3_6b_jpfull30 | 0.031 |
| granite ↔ rinna_bilingual_4b | 0.031 |
| granite ↔ rinna_jp_3_6b_jp | 0.028 |
| gemma ↔ rinna_jp_3_6b_jpfull | 0.028 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull30 | 0.027 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull | 0.027 |
| gpt_oss_20b ↔ rinna_bilingual_4b | 0.026 |
| qwen ↔ rinna_jp_3_6b_jp | 0.026 |
| qwen ↔ rinna_jp_3_6b_jpfull | 0.025 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b | 0.024 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull | 0.019 |
| gemma ↔ rinna_bilingual_4b_jp | 0.019 |
| qwen ↔ rinna_bilingual_4b_jp | 0.019 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jp | 0.018 |
| ministral ↔ rinna_bilingual_4b_jpfull | 0.017 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull | 0.017 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b | 0.014 |
| ministral ↔ rinna_bilingual_4b_jp | 0.012 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull | 0.011 |
| gemma ↔ rinna_jp_3_6b | 0.011 |
| granite ↔ rinna_bilingual_4b_jp | 0.008 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b | 0.008 |
| ministral ↔ rinna_jp_3_6b_jp | 0.008 |
| granite ↔ rinna_bilingual_4b_jpfull | 0.007 |
| ministral ↔ rinna_bilingual_4b_jpfull30 | 0.005 |
| granite ↔ rinna_jp_3_6b | 0.004 |
| ministral ↔ rinna_jp_3_6b_jpfull | 0.003 |
| haiku ↔ rinna_jp_3_6b | 0.003 |
| ministral ↔ rinna_bilingual_4b | 0.003 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b | 0.003 |
| ministral ↔ rinna_jp_3_6b | -0.002 |
| qwen ↔ rinna_jp_3_6b | -0.003 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull30 | -0.007 |
| gpt_oss_20b ↔ rinna_jp_3_6b | -0.009 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jp | -0.011 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull30 | -0.013 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jp | -0.014 |
| qwen ↔ rinna_bilingual_4b | -0.034 |

## Top 15 subsets by weighted-vote accuracy

| rank | size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 5 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b} | 56.0% (93/166) | 0.466 | 62.7% (47/75) | 54.8% |
| 2 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 56.0% (93/166) | 0.467 | 80.0% (20/25) | 84.9% |
| 3 | 5 | {gemma,gpt_oss_20b,ministral,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 56.0% (93/166) | 0.469 | 57.1% (36/63) | 62.0% |
| 4 | 6 | {gemma,gpt_oss_20b,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 56.0% (93/166) | 0.469 | 71.4% (20/28) | 83.1% |
| 5 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_jp_3_6b} | 55.4% (92/166) | 0.459 | 67.9% (19/28) | 83.1% |
| 6 | 5 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b,rinna_jp_3_6b_jpfull} | 55.4% (92/166) | 0.458 | 60.5% (46/76) | 54.2% |
| 7 | 6 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 55.4% (92/166) | 0.459 | 71.4% (20/28) | 83.1% |
| 8 | 6 | {gemma,gpt_oss_20b,ministral,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 55.4% (92/166) | 0.462 | 64.3% (18/28) | 83.1% |
| 9 | 7 | {gemma,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 54.8% (91/166) | 0.453 | 60.7% (37/61) | 63.3% |
| 10 | 8 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 54.8% (91/166) | 0.453 | 72.4% (21/29) | 82.5% |
| 11 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 54.8% (91/166) | 0.452 | 77.4% (24/31) | 81.3% |
| 12 | 7 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 54.8% (91/166) | 0.453 | 71.2% (42/59) | 64.5% |
| 13 | 6 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 54.8% (91/166) | 0.453 | 73.2% (30/41) | 75.3% |
| 14 | 7 | {gemma,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 54.8% (91/166) | 0.453 | 65.0% (39/60) | 63.9% |
| 15 | 7 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 54.8% (91/166) | 0.453 | 62.9% (39/62) | 62.7% |

## Top 15 subsets by weighted-vote Cohen's κ

(Class-imbalanced subsets that ride the empirical majority-class base rate score lower here than under raw accuracy.)

| rank | size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6 | {gemma,gpt_oss_20b,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.469 | 56.0% (93/166) | 71.4% (20/28) | 83.1% |
| 2 | 5 | {gemma,gpt_oss_20b,ministral,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.469 | 56.0% (93/166) | 57.1% (36/63) | 62.0% |
| 3 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.467 | 56.0% (93/166) | 80.0% (20/25) | 84.9% |
| 4 | 5 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b} | 0.466 | 56.0% (93/166) | 62.7% (47/75) | 54.8% |
| 5 | 6 | {gemma,gpt_oss_20b,ministral,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.462 | 55.4% (92/166) | 64.3% (18/28) | 83.1% |
| 6 | 6 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.459 | 55.4% (92/166) | 71.4% (20/28) | 83.1% |
| 7 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_jp_3_6b} | 0.459 | 55.4% (92/166) | 67.9% (19/28) | 83.1% |
| 8 | 5 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b,rinna_jp_3_6b_jpfull} | 0.458 | 55.4% (92/166) | 60.5% (46/76) | 54.2% |
| 9 | 7 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.453 | 54.8% (91/166) | 71.2% (42/59) | 64.5% |
| 10 | 8 | {gemma,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.453 | 54.8% (91/166) | 73.3% (22/30) | 81.9% |
| 11 | 8 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.453 | 54.8% (91/166) | 72.4% (21/29) | 82.5% |
| 12 | 4 | {gemma,ministral,qwen,rinna_jp_3_6b} | 0.453 | 54.8% (91/166) | 67.9% (19/28) | 83.1% |
| 13 | 7 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.453 | 54.8% (91/166) | 62.9% (39/62) | 62.7% |
| 14 | 7 | {gemma,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.453 | 54.8% (91/166) | 60.7% (37/61) | 63.3% |
| 15 | 8 | {gemma,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.453 | 54.8% (91/166) | 74.4% (32/43) | 74.1% |

## Top 15 subsets by strict-majority accuracy

(ties broken by larger n_resolved, i.e. more decisive)

| rank | size | encoders | majority(resolved) | weighted | abstain |
|---:|---:|---|---:|---:|---:|
| 1 | 8 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 94.1% (16/17) | 42.8% (71/166) | 89.8% |
| 2 | 6 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 93.8% (15/16) | 47.0% (78/166) | 90.4% |
| 3 | 6 | {granite,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 91.7% (11/12) | 44.0% (73/166) | 92.8% |
| 4 | 8 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 90.5% (19/21) | 47.6% (79/166) | 87.3% |
| 5 | 8 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 90.5% (19/21) | 46.4% (77/166) | 87.3% |
| 6 | 6 | {gemma,granite,ministral,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 90.0% (9/10) | 42.8% (71/166) | 94.0% |
| 7 | 6 | {granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b} | 90.0% (9/10) | 39.8% (66/166) | 94.0% |
| 8 | 8 | {gpt_oss_20b,granite,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b} | 89.5% (17/19) | 45.8% (76/166) | 88.6% |
| 9 | 8 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 89.5% (17/19) | 41.6% (69/166) | 88.6% |
| 10 | 8 | {gemma,granite,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 88.9% (16/18) | 50.6% (84/166) | 89.2% |
| 11 | 8 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b} | 88.9% (16/18) | 47.6% (79/166) | 89.2% |
| 12 | 8 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 88.9% (16/18) | 46.4% (77/166) | 89.2% |
| 13 | 10 | {gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 88.9% (16/18) | 44.6% (74/166) | 89.2% |
| 14 | 10 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 88.9% (16/18) | 40.4% (67/166) | 89.2% |
| 15 | 4 | {gemma,granite,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 88.9% (8/9) | 49.4% (82/166) | 94.6% |

## Best subset per size (by weighted accuracy)

| size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {gemma} | 51.8% (86/166) | 0.418 | 51.8% (86/166) | 0.0% |
| 2 | {gemma,ministral} | 52.4% (87/166) | 0.425 | 57.7% (30/52) | 68.7% |
| 3 | {gemma,ministral,qwen} | 54.2% (90/166) | 0.446 | 52.5% (53/101) | 39.2% |
| 4 | {gemma,ministral,qwen,rinna_jp_3_6b} | 54.8% (91/166) | 0.453 | 67.9% (19/28) | 83.1% |
| 5 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b} | 56.0% (93/166) | 0.466 | 62.7% (47/75) | 54.8% |
| 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 56.0% (93/166) | 0.467 | 80.0% (20/25) | 84.9% |
| 7 | {gemma,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 54.8% (91/166) | 0.453 | 60.7% (37/61) | 63.3% |
| 8 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 54.8% (91/166) | 0.453 | 72.4% (21/29) | 82.5% |
| 9 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 54.2% (90/166) | 0.445 | 75.0% (33/44) | 73.5% |
| 10 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 53.6% (89/166) | 0.438 | 80.0% (20/25) | 84.9% |
| 11 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 53.0% (88/166) | 0.431 | 75.7% (28/37) | 77.7% |
| 12 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 51.8% (86/166) | 0.417 | 77.8% (21/27) | 83.7% |
| 13 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 50.6% (84/166) | 0.402 | 66.7% (24/36) | 78.3% |
| 14 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 49.4% (82/166) | 0.388 | 72.0% (18/25) | 84.9% |

## Best subset per size (by κ)

| size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {gemma} | 0.418 | 51.8% (86/166) | 51.8% (86/166) | 0.0% |
| 2 | {gemma,ministral} | 0.425 | 52.4% (87/166) | 57.7% (30/52) | 68.7% |
| 3 | {gemma,ministral,qwen} | 0.446 | 54.2% (90/166) | 52.5% (53/101) | 39.2% |
| 4 | {gemma,ministral,qwen,rinna_jp_3_6b} | 0.453 | 54.8% (91/166) | 67.9% (19/28) | 83.1% |
| 5 | {gemma,gpt_oss_20b,ministral,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.469 | 56.0% (93/166) | 57.1% (36/63) | 62.0% |
| 6 | {gemma,gpt_oss_20b,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.469 | 56.0% (93/166) | 71.4% (20/28) | 83.1% |
| 7 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.453 | 54.8% (91/166) | 71.2% (42/59) | 64.5% |
| 8 | {gemma,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.453 | 54.8% (91/166) | 73.3% (22/30) | 81.9% |
| 9 | {gemma,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.448 | 54.2% (90/166) | 62.3% (33/53) | 68.1% |
| 10 | {gemma,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.441 | 53.6% (89/166) | 77.8% (21/27) | 83.7% |
| 11 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 0.431 | 53.0% (88/166) | 73.7% (28/38) | 77.1% |
| 12 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.417 | 51.8% (86/166) | 77.8% (21/27) | 83.7% |
| 13 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.402 | 50.6% (84/166) | 66.7% (24/36) | 78.3% |
| 14 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.388 | 49.4% (82/166) | 72.0% (18/25) | 84.9% |

## Encoder inclusion frequency in top-15 weighted-acc

Heuristic: encoders that appear in nearly all top subsets are ensemble-load-bearing; encoders that rarely appear are individually weak AND fail to add complementary signal.

| encoder | top-K acc | top-K κ | solo acc | solo κ |
|---|---:|---:|---:|---:|
| gemma | 15/15 | 15/15 | 51.8% | 0.418 |
| gpt_oss_20b | 14/15 | 13/15 | 44.6% | 0.334 |
| rinna_jp_3_6b_jpfull | 12/15 | 11/15 | 21.1% | 0.070 |
| rinna_jp_3_6b | 10/15 | 11/15 | 16.3% | -0.007 |
| ministral | 9/15 | 10/15 | 31.9% | 0.178 |
| qwen | 7/15 | 8/15 | 30.1% | 0.167 |
| haiku | 6/15 | 6/15 | 40.4% | 0.276 |
| rinna_bilingual_4b_jpfull30 | 6/15 | 6/15 | 24.1% | 0.100 |
| granite | 6/15 | 5/15 | 34.9% | 0.205 |
| rinna_bilingual_4b | 4/15 | 4/15 | 22.3% | 0.063 |
| rinna_jp_3_6b_jpfull30 | 3/15 | 4/15 | 25.9% | 0.105 |
| rinna_jp_3_6b_jp | 1/15 | 1/15 | 21.1% | 0.049 |
| rinna_bilingual_4b_jp | 0/15 | 0/15 | 16.9% | 0.000 |
| rinna_bilingual_4b_jpfull | 0/15 | 0/15 | 19.3% | 0.064 |
