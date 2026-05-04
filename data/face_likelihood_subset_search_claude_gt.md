# Face_likelihood — exhaustive subset search

**Encoders:** 13  (gemma, gpt_oss_20b, granite, ministral, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_bilingual_4b_jpfull, rinna_bilingual_4b_jpfull30, rinna_jp_3_6b, rinna_jp_3_6b_jp, rinna_jp_3_6b_jpfull, rinna_jp_3_6b_jpfull30)
**Faces (overlap):** 573
**GT subset (Claude pilot modal, floor=1):** 51
**Subsets evaluated:** 8191

## Headline

- Best single encoder by accuracy: **gemma** at 62.7% (32/51); Cohen's κ = 0.548
- Best weighted-vote subset by accuracy: **{gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp}** at **72.5%** (37/51) — size 6, +9.8pp over best single; κ = 0.663
- Best weighted-vote subset by κ: **{gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp}** at κ = **0.663** (accuracy 72.5%, size 6)
- Best strict-majority subset: **{gemma}** at 62.7% on 51 resolved (abstains on 0 all-distinct); κ = 0.548

**Reading κ:** Cohen's kappa corrects agreement for chance. 0.0 = no signal beyond random, 0.4–0.6 = moderate, 0.6–0.8 = substantial, >0.8 = near-perfect. Penalizes encoders that always predict the majority class — useful given GLM's 100%-LN bias. Voting models often have lower κ than accuracy because the vote concentrates predictions on common quadrants.

## Per-encoder solo accuracy + Cohen's κ vs empirical

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 62.7% (32/51) | 0.548 |
| gpt_oss_20b | 47.1% (24/51) | 0.360 |
| granite | 43.1% (22/51) | 0.298 |
| ministral | 39.2% (20/51) | 0.257 |
| rinna_bilingual_4b_jpfull30 | 35.3% (18/51) | 0.212 |
| rinna_bilingual_4b_jpfull | 33.3% (17/51) | 0.217 |
| rinna_jp_3_6b_jpfull | 33.3% (17/51) | 0.216 |
| rinna_jp_3_6b_jpfull30 | 33.3% (17/51) | 0.184 |
| qwen | 31.4% (16/51) | 0.174 |
| rinna_bilingual_4b | 29.4% (15/51) | 0.115 |
| rinna_jp_3_6b_jp | 29.4% (15/51) | 0.123 |
| rinna_jp_3_6b | 23.5% (12/51) | 0.083 |
| rinna_bilingual_4b_jp | 21.6% (11/51) | 0.057 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Useful for ensemble design: encoder pairs with low κ make complementary errors and are more useful to combine than encoder pairs with high κ.

| pair | κ |
|---|---:|
| rinna_bilingual_4b_jpfull ↔ rinna_bilingual_4b_jpfull30 | 0.354 |
| rinna_jp_3_6b_jpfull ↔ rinna_jp_3_6b_jpfull30 | 0.323 |
| gemma ↔ gpt_oss_20b | 0.236 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.235 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jp | 0.195 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull | 0.188 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b | 0.167 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jp | 0.144 |
| gpt_oss_20b ↔ granite | 0.140 |
| gemma ↔ ministral | 0.139 |
| granite ↔ ministral | 0.136 |
| gpt_oss_20b ↔ ministral | 0.133 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull30 | 0.128 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull | 0.127 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jp | 0.124 |
| gemma ↔ granite | 0.119 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull | 0.093 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull30 | 0.092 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.084 |
| gemma ↔ rinna_jp_3_6b_jpfull30 | 0.082 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b | 0.080 |
| granite ↔ rinna_jp_3_6b_jpfull30 | 0.070 |
| granite ↔ rinna_jp_3_6b_jpfull | 0.065 |
| ministral ↔ rinna_jp_3_6b_jpfull30 | 0.062 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull | 0.060 |
| ministral ↔ qwen | 0.059 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b | 0.058 |
| gemma ↔ qwen | 0.055 |
| gemma ↔ rinna_bilingual_4b_jpfull | 0.053 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull | 0.052 |
| gemma ↔ rinna_jp_3_6b_jp | 0.051 |
| qwen ↔ rinna_jp_3_6b_jpfull30 | 0.051 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull30 | 0.047 |
| granite ↔ rinna_bilingual_4b_jpfull30 | 0.047 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull30 | 0.046 |
| gemma ↔ rinna_bilingual_4b_jpfull30 | 0.043 |
| granite ↔ qwen | 0.043 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull30 | 0.042 |
| gemma ↔ rinna_bilingual_4b_jp | 0.041 |
| qwen ↔ rinna_bilingual_4b | 0.041 |
| gemma ↔ rinna_jp_3_6b | 0.041 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull30 | 0.039 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jp | 0.039 |
| ministral ↔ rinna_bilingual_4b_jp | 0.037 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull | 0.036 |
| gpt_oss_20b ↔ qwen | 0.034 |
| granite ↔ rinna_bilingual_4b_jpfull | 0.032 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull | 0.030 |
| qwen ↔ rinna_bilingual_4b_jpfull30 | 0.028 |
| granite ↔ rinna_bilingual_4b | 0.026 |
| granite ↔ rinna_jp_3_6b_jp | 0.026 |
| gemma ↔ rinna_jp_3_6b_jpfull | 0.024 |
| gpt_oss_20b ↔ rinna_bilingual_4b | 0.024 |
| ministral ↔ rinna_jp_3_6b_jpfull | 0.023 |
| ministral ↔ rinna_jp_3_6b_jp | 0.021 |
| qwen ↔ rinna_jp_3_6b_jpfull | 0.021 |
| qwen ↔ rinna_bilingual_4b_jpfull | 0.020 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jp | 0.020 |
| granite ↔ rinna_bilingual_4b_jp | 0.019 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jp | 0.019 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jp | 0.016 |
| gemma ↔ rinna_bilingual_4b | 0.016 |
| ministral ↔ rinna_bilingual_4b_jpfull30 | 0.016 |
| ministral ↔ rinna_bilingual_4b_jpfull | 0.011 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull | 0.010 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull30 | 0.010 |
| qwen ↔ rinna_jp_3_6b_jp | -0.003 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull30 | -0.007 |
| ministral ↔ rinna_bilingual_4b | -0.010 |
| gpt_oss_20b ↔ rinna_jp_3_6b | -0.010 |
| ministral ↔ rinna_jp_3_6b | -0.010 |
| qwen ↔ rinna_bilingual_4b_jp | -0.013 |
| granite ↔ rinna_jp_3_6b | -0.014 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b | -0.014 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jp | -0.015 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull | -0.019 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull | -0.029 |
| qwen ↔ rinna_jp_3_6b | -0.042 |

## Top 15 subsets by weighted-vote accuracy

| rank | size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 72.5% (37/51) | 0.663 | 78.6% (11/14) | 72.5% |
| 2 | 9 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 72.5% (37/51) | 0.662 | 75.0% (15/20) | 60.8% |
| 3 | 4 | {gemma,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 72.5% (37/51) | 0.663 | 75.0% (9/12) | 76.5% |
| 4 | 7 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 72.5% (37/51) | 0.662 | 77.3% (17/22) | 56.9% |
| 5 | 9 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 70.6% (36/51) | 0.637 | 80.0% (16/20) | 60.8% |
| 6 | 7 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30} | 70.6% (36/51) | 0.638 | 76.9% (20/26) | 49.0% |
| 7 | 8 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 70.6% (36/51) | 0.636 | 83.3% (10/12) | 76.5% |
| 8 | 7 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 70.6% (36/51) | 0.638 | 77.3% (17/22) | 56.9% |
| 9 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 70.6% (36/51) | 0.639 | 84.6% (11/13) | 74.5% |
| 10 | 9 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 70.6% (36/51) | 0.636 | 80.0% (12/15) | 70.6% |
| 11 | 5 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 70.6% (36/51) | 0.640 | 73.1% (19/26) | 49.0% |
| 12 | 8 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 70.6% (36/51) | 0.637 | 92.9% (13/14) | 72.5% |
| 13 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30} | 70.6% (36/51) | 0.638 | 76.5% (13/17) | 66.7% |
| 14 | 8 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 70.6% (36/51) | 0.637 | 80.0% (8/10) | 80.4% |
| 15 | 5 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_jp_3_6b_jpfull} | 70.6% (36/51) | 0.640 | 75.9% (22/29) | 43.1% |

## Top 15 subsets by weighted-vote Cohen's κ

(Class-imbalanced subsets that ride the empirical majority-class base rate score lower here than under raw accuracy.)

| rank | size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 0.663 | 72.5% (37/51) | 78.6% (11/14) | 72.5% |
| 2 | 4 | {gemma,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 0.663 | 72.5% (37/51) | 75.0% (9/12) | 76.5% |
| 3 | 9 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.662 | 72.5% (37/51) | 75.0% (15/20) | 60.8% |
| 4 | 7 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 0.662 | 72.5% (37/51) | 77.3% (17/22) | 56.9% |
| 5 | 7 | {gemma,gpt_oss_20b,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 0.641 | 70.6% (36/51) | 72.2% (13/18) | 64.7% |
| 6 | 5 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 0.640 | 70.6% (36/51) | 73.1% (19/26) | 49.0% |
| 7 | 5 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_jp_3_6b_jpfull} | 0.640 | 70.6% (36/51) | 75.9% (22/29) | 43.1% |
| 8 | 6 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 0.640 | 70.6% (36/51) | 84.6% (11/13) | 74.5% |
| 9 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.640 | 70.6% (36/51) | 100.0% (13/13) | 74.5% |
| 10 | 7 | {gemma,gpt_oss_20b,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull30} | 0.640 | 70.6% (36/51) | 78.9% (15/19) | 62.7% |
| 11 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.640 | 70.6% (36/51) | 77.3% (17/22) | 56.9% |
| 12 | 6 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b} | 0.639 | 70.6% (36/51) | 87.5% (7/8) | 84.3% |
| 13 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.639 | 70.6% (36/51) | 84.6% (11/13) | 74.5% |
| 14 | 7 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30} | 0.639 | 70.6% (36/51) | 69.2% (18/26) | 49.0% |
| 15 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30} | 0.638 | 70.6% (36/51) | 91.7% (11/12) | 76.5% |

## Top 15 subsets by strict-majority accuracy

(ties broken by larger n_resolved, i.e. more decisive)

| rank | size | encoders | majority(resolved) | weighted | abstain |
|---:|---:|---|---:|---:|---:|
| 1 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 100.0% (14/14) | 68.6% (35/51) | 72.5% |
| 2 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 100.0% (14/14) | 62.7% (32/51) | 72.5% |
| 3 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 100.0% (13/13) | 70.6% (36/51) | 74.5% |
| 4 | 8 | {gemma,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 100.0% (13/13) | 60.8% (31/51) | 74.5% |
| 5 | 8 | {gemma,gpt_oss_20b,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 100.0% (12/12) | 66.7% (34/51) | 76.5% |
| 6 | 6 | {gemma,granite,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 100.0% (12/12) | 64.7% (33/51) | 76.5% |
| 7 | 10 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 100.0% (12/12) | 64.7% (33/51) | 76.5% |
| 8 | 8 | {gemma,gpt_oss_20b,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 100.0% (12/12) | 62.7% (32/51) | 76.5% |
| 9 | 4 | {gemma,granite,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 100.0% (12/12) | 60.8% (31/51) | 76.5% |
| 10 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 100.0% (11/11) | 66.7% (34/51) | 78.4% |
| 11 | 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 100.0% (11/11) | 66.7% (34/51) | 78.4% |
| 12 | 6 | {gemma,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 100.0% (11/11) | 64.7% (33/51) | 78.4% |
| 13 | 6 | {gemma,granite,ministral,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 100.0% (11/11) | 64.7% (33/51) | 78.4% |
| 14 | 10 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 100.0% (11/11) | 64.7% (33/51) | 78.4% |
| 15 | 8 | {gemma,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 100.0% (11/11) | 64.7% (33/51) | 78.4% |

## Best subset per size (by weighted accuracy)

| size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {gemma} | 62.7% (32/51) | 0.548 | 62.7% (32/51) | 0.0% |
| 2 | {gemma,rinna_bilingual_4b_jpfull30} | 66.7% (34/51) | 0.595 | 93.3% (14/15) | 70.6% |
| 3 | {gemma,qwen,rinna_bilingual_4b_jpfull30} | 68.6% (35/51) | 0.618 | 76.0% (19/25) | 51.0% |
| 4 | {gemma,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 72.5% (37/51) | 0.663 | 75.0% (9/12) | 76.5% |
| 5 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 70.6% (36/51) | 0.640 | 73.1% (19/26) | 49.0% |
| 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 72.5% (37/51) | 0.663 | 78.6% (11/14) | 72.5% |
| 7 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 72.5% (37/51) | 0.662 | 77.3% (17/22) | 56.9% |
| 8 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 70.6% (36/51) | 0.636 | 83.3% (10/12) | 76.5% |
| 9 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 72.5% (37/51) | 0.662 | 75.0% (15/20) | 60.8% |
| 10 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull30} | 68.6% (35/51) | 0.613 | 81.8% (9/11) | 78.4% |
| 11 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 68.6% (35/51) | 0.612 | 80.0% (12/15) | 70.6% |
| 12 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 64.7% (33/51) | 0.564 | 83.3% (10/12) | 76.5% |
| 13 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 62.7% (32/51) | 0.541 | 73.3% (11/15) | 70.6% |

## Best subset per size (by κ)

| size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {gemma} | 0.548 | 62.7% (32/51) | 62.7% (32/51) | 0.0% |
| 2 | {gemma,rinna_bilingual_4b_jpfull30} | 0.595 | 66.7% (34/51) | 93.3% (14/15) | 70.6% |
| 3 | {gemma,qwen,rinna_bilingual_4b_jpfull30} | 0.618 | 68.6% (35/51) | 76.0% (19/25) | 51.0% |
| 4 | {gemma,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 0.663 | 72.5% (37/51) | 75.0% (9/12) | 76.5% |
| 5 | {gemma,gpt_oss_20b,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30} | 0.640 | 70.6% (36/51) | 73.1% (19/26) | 49.0% |
| 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 0.663 | 72.5% (37/51) | 78.6% (11/14) | 72.5% |
| 7 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 0.662 | 72.5% (37/51) | 77.3% (17/22) | 56.9% |
| 8 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.640 | 70.6% (36/51) | 100.0% (13/13) | 74.5% |
| 9 | {gemma,gpt_oss_20b,granite,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.662 | 72.5% (37/51) | 75.0% (15/20) | 60.8% |
| 10 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull30} | 0.613 | 68.6% (35/51) | 81.8% (9/11) | 78.4% |
| 11 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.612 | 68.6% (35/51) | 80.0% (12/15) | 70.6% |
| 12 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.564 | 64.7% (33/51) | 83.3% (10/12) | 76.5% |
| 13 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.541 | 62.7% (32/51) | 73.3% (11/15) | 70.6% |

## Encoder inclusion frequency in top-15 weighted-acc

Heuristic: encoders that appear in nearly all top subsets are ensemble-load-bearing; encoders that rarely appear are individually weak AND fail to add complementary signal.

| encoder | top-K acc | top-K κ | solo acc | solo κ |
|---|---:|---:|---:|---:|
| gemma | 15/15 | 15/15 | 62.7% | 0.548 |
| gpt_oss_20b | 14/15 | 14/15 | 47.1% | 0.360 |
| qwen | 13/15 | 15/15 | 31.4% | 0.174 |
| granite | 12/15 | 8/15 | 43.1% | 0.298 |
| rinna_bilingual_4b | 11/15 | 9/15 | 29.4% | 0.115 |
| rinna_bilingual_4b_jpfull30 | 10/15 | 9/15 | 35.3% | 0.212 |
| rinna_jp_3_6b_jp | 10/15 | 7/15 | 29.4% | 0.123 |
| rinna_bilingual_4b_jpfull | 9/15 | 5/15 | 33.3% | 0.217 |
| rinna_jp_3_6b_jpfull | 6/15 | 6/15 | 33.3% | 0.216 |
| ministral | 4/15 | 5/15 | 39.2% | 0.257 |
| rinna_bilingual_4b_jp | 0/15 | 2/15 | 21.6% | 0.057 |
| rinna_jp_3_6b | 0/15 | 2/15 | 23.5% | 0.083 |
| rinna_jp_3_6b_jpfull30 | 0/15 | 1/15 | 33.3% | 0.184 |
