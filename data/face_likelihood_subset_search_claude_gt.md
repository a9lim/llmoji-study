# Face_likelihood — exhaustive subset search

**Encoders:** 15  (gemma, gemma_v7primed, gpt_oss_20b, granite, haiku, ministral, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_bilingual_4b_jpfull, rinna_bilingual_4b_jpfull30, rinna_jp_3_6b, rinna_jp_3_6b_jp, rinna_jp_3_6b_jpfull, rinna_jp_3_6b_jpfull30)
**Faces (overlap):** 573
**GT subset (Claude pilot modal, floor=1):** 51
**Subsets evaluated:** 32767

## Headline

- Best single encoder by accuracy: **haiku** at 58.8% (30/51); Cohen's κ = 0.492
- Best weighted-vote subset by accuracy: **{gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull}** at **68.6%** (35/51) — size 6, +9.8pp over best single; κ = 0.616
- Best weighted-vote subset by κ: **{gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull}** at κ = **0.616** (accuracy 68.6%, size 6)
- Best strict-majority subset: **{haiku}** at 58.8% on 51 resolved (abstains on 0 all-distinct); κ = 0.492

**Reading κ:** Cohen's kappa corrects agreement for chance. 0.0 = no signal beyond random, 0.4–0.6 = moderate, 0.6–0.8 = substantial, >0.8 = near-perfect. Penalizes encoders that always predict the majority class — useful given GLM's 100%-LN bias. Voting models often have lower κ than accuracy because the vote concentrates predictions on common quadrants.

## Per-encoder solo accuracy + Cohen's κ vs empirical

| encoder | accuracy | κ |
|---|---:|---:|
| haiku | 58.8% (30/51) | 0.492 |
| gemma | 56.9% (29/51) | 0.478 |
| gemma_v7primed | 49.0% (25/51) | 0.381 |
| gpt_oss_20b | 47.1% (24/51) | 0.360 |
| granite | 41.2% (21/51) | 0.275 |
| rinna_bilingual_4b_jpfull30 | 35.3% (18/51) | 0.212 |
| rinna_jp_3_6b_jpfull | 33.3% (17/51) | 0.216 |
| rinna_jp_3_6b_jpfull30 | 33.3% (17/51) | 0.184 |
| ministral | 31.4% (16/51) | 0.168 |
| rinna_jp_3_6b_jp | 25.5% (13/51) | 0.068 |
| rinna_bilingual_4b | 23.5% (12/51) | 0.044 |
| rinna_bilingual_4b_jpfull | 23.5% (12/51) | 0.111 |
| qwen | 21.6% (11/51) | 0.051 |
| rinna_bilingual_4b_jp | 21.6% (11/51) | 0.032 |
| rinna_jp_3_6b | 13.7% (7/51) | 0.010 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Useful for ensemble design: encoder pairs with low κ make complementary errors and are more useful to combine than encoder pairs with high κ.

| pair | κ |
|---|---:|
| gemma ↔ gemma_v7primed | 0.757 |
| rinna_bilingual_4b_jpfull ↔ rinna_bilingual_4b_jpfull30 | 0.493 |
| rinna_jp_3_6b_jpfull ↔ rinna_jp_3_6b_jpfull30 | 0.323 |
| gemma_v7primed ↔ haiku | 0.312 |
| gemma ↔ haiku | 0.305 |
| gpt_oss_20b ↔ haiku | 0.263 |
| gemma ↔ gpt_oss_20b | 0.248 |
| gemma_v7primed ↔ gpt_oss_20b | 0.231 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.223 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jp | 0.193 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull | 0.188 |
| granite ↔ ministral | 0.185 |
| granite ↔ haiku | 0.168 |
| gpt_oss_20b ↔ granite | 0.148 |
| gemma ↔ ministral | 0.141 |
| haiku ↔ ministral | 0.126 |
| gpt_oss_20b ↔ ministral | 0.124 |
| gemma ↔ granite | 0.124 |
| gemma_v7primed ↔ ministral | 0.119 |
| gemma_v7primed ↔ granite | 0.116 |
| gpt_oss_20b ↔ qwen | 0.114 |
| gemma_v7primed ↔ rinna_jp_3_6b_jpfull30 | 0.113 |
| qwen ↔ rinna_bilingual_4b_jpfull | 0.103 |
| haiku ↔ qwen | 0.097 |
| gemma_v7primed ↔ qwen | 0.093 |
| haiku ↔ rinna_jp_3_6b_jpfull30 | 0.090 |
| gemma ↔ qwen | 0.089 |
| ministral ↔ qwen | 0.087 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull | 0.077 |
| granite ↔ rinna_jp_3_6b_jpfull | 0.076 |
| gemma ↔ rinna_jp_3_6b_jpfull30 | 0.075 |
| haiku ↔ rinna_bilingual_4b_jpfull30 | 0.075 |
| gemma_v7primed ↔ rinna_bilingual_4b_jpfull30 | 0.074 |
| haiku ↔ rinna_jp_3_6b_jpfull | 0.071 |
| granite ↔ rinna_jp_3_6b_jpfull30 | 0.070 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull | 0.065 |
| gemma_v7primed ↔ rinna_bilingual_4b_jpfull | 0.064 |
| granite ↔ qwen | 0.064 |
| granite ↔ rinna_bilingual_4b_jpfull30 | 0.064 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull30 | 0.062 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jp | 0.062 |
| qwen ↔ rinna_bilingual_4b_jpfull30 | 0.060 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jp | 0.059 |
| gemma ↔ rinna_bilingual_4b_jpfull30 | 0.055 |
| haiku ↔ rinna_bilingual_4b_jpfull | 0.054 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull30 | 0.051 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.050 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull | 0.050 |
| gemma_v7primed ↔ rinna_bilingual_4b | 0.048 |
| haiku ↔ rinna_bilingual_4b | 0.048 |
| ministral ↔ rinna_jp_3_6b_jpfull30 | 0.047 |
| gemma_v7primed ↔ rinna_jp_3_6b_jp | 0.047 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull30 | 0.046 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jp | 0.046 |
| gemma_v7primed ↔ rinna_jp_3_6b_jpfull | 0.046 |
| haiku ↔ rinna_jp_3_6b_jp | 0.045 |
| gemma ↔ rinna_jp_3_6b_jp | 0.043 |
| gemma ↔ rinna_bilingual_4b_jpfull | 0.043 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull30 | 0.042 |
| gemma ↔ rinna_bilingual_4b | 0.041 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull | 0.036 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull | 0.034 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull30 | 0.034 |
| haiku ↔ rinna_bilingual_4b_jp | 0.033 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jp | 0.033 |
| qwen ↔ rinna_jp_3_6b_jpfull30 | 0.031 |
| granite ↔ rinna_bilingual_4b | 0.031 |
| granite ↔ rinna_jp_3_6b_jp | 0.028 |
| gemma ↔ rinna_jp_3_6b_jpfull | 0.028 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull30 | 0.027 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull | 0.027 |
| gpt_oss_20b ↔ rinna_bilingual_4b | 0.026 |
| qwen ↔ rinna_jp_3_6b_jp | 0.025 |
| qwen ↔ rinna_jp_3_6b_jpfull | 0.024 |
| gemma_v7primed ↔ rinna_bilingual_4b_jp | 0.023 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b | 0.023 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull | 0.021 |
| gemma ↔ rinna_bilingual_4b_jp | 0.019 |
| qwen ↔ rinna_bilingual_4b_jp | 0.019 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jp | 0.018 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull | 0.017 |
| ministral ↔ rinna_bilingual_4b_jpfull | 0.017 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b | 0.014 |
| ministral ↔ rinna_bilingual_4b_jp | 0.013 |
| gemma_v7primed ↔ rinna_jp_3_6b | 0.012 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull | 0.010 |
| granite ↔ rinna_bilingual_4b_jp | 0.009 |
| gemma ↔ rinna_jp_3_6b | 0.009 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b | 0.008 |
| ministral ↔ rinna_jp_3_6b_jp | 0.008 |
| granite ↔ rinna_bilingual_4b_jpfull | 0.007 |
| ministral ↔ rinna_jp_3_6b_jpfull | 0.005 |
| ministral ↔ rinna_bilingual_4b_jpfull30 | 0.005 |
| granite ↔ rinna_jp_3_6b | 0.003 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b | 0.003 |
| ministral ↔ rinna_bilingual_4b | 0.003 |
| haiku ↔ rinna_jp_3_6b | -0.000 |
| ministral ↔ rinna_jp_3_6b | -0.002 |
| qwen ↔ rinna_jp_3_6b | -0.003 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull30 | -0.007 |
| gpt_oss_20b ↔ rinna_jp_3_6b | -0.009 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jp | -0.010 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull30 | -0.013 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jp | -0.013 |
| qwen ↔ rinna_bilingual_4b | -0.034 |

## Top 8 subsets by weighted-vote accuracy

| rank | size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 68.6% (35/51) | 0.616 | 70.6% (12/17) | 66.7% |
| 2 | 5 | {gemma,gpt_oss_20b,granite,ministral,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.592 | 66.7% (18/27) | 47.1% |
| 3 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.591 | 70.8% (17/24) | 52.9% |
| 4 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.590 | 66.7% (14/21) | 58.8% |
| 5 | 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jp,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.593 | 62.5% (10/16) | 68.6% |
| 6 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.593 | 64.0% (16/25) | 51.0% |
| 7 | 9 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 64.7% (33/51) | 0.569 | 77.8% (14/18) | 64.7% |
| 8 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 64.7% (33/51) | 0.570 | 66.7% (16/24) | 52.9% |

## Top 8 subsets by weighted-vote Cohen's κ

(Class-imbalanced subsets that ride the empirical majority-class base rate score lower here than under raw accuracy.)

| rank | size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 0.616 | 68.6% (35/51) | 70.6% (12/17) | 66.7% |
| 2 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_jp_3_6b_jpfull} | 0.593 | 66.7% (34/51) | 64.0% (16/25) | 51.0% |
| 3 | 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jp,rinna_jp_3_6b_jpfull} | 0.593 | 66.7% (34/51) | 62.5% (10/16) | 68.6% |
| 4 | 5 | {gemma,gpt_oss_20b,granite,ministral,rinna_jp_3_6b_jpfull} | 0.592 | 66.7% (34/51) | 66.7% (18/27) | 47.1% |
| 5 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 0.591 | 66.7% (34/51) | 70.8% (17/24) | 52.9% |
| 6 | 6 | {gemma,gpt_oss_20b,granite,qwen,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.590 | 66.7% (34/51) | 66.7% (14/21) | 58.8% |
| 7 | 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 0.570 | 64.7% (33/51) | 68.8% (11/16) | 68.6% |
| 8 | 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 0.570 | 64.7% (33/51) | 66.7% (16/24) | 52.9% |

## Top 8 subsets by strict-majority accuracy

(ties broken by larger n_resolved, i.e. more decisive)

| rank | size | encoders | majority(resolved) | weighted | abstain |
|---:|---:|---|---:|---:|---:|
| 1 | 8 | {gpt_oss_20b,granite,haiku,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 100.0% (10/10) | 47.1% (24/51) | 80.4% |
| 2 | 8 | {gpt_oss_20b,granite,haiku,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 100.0% (9/9) | 47.1% (24/51) | 82.4% |
| 3 | 8 | {gemma,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 100.0% (8/8) | 56.9% (29/51) | 84.3% |
| 4 | 8 | {gemma_v7primed,granite,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 100.0% (8/8) | 52.9% (27/51) | 84.3% |
| 5 | 6 | {gemma,granite,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull30} | 100.0% (8/8) | 52.9% (27/51) | 84.3% |
| 6 | 6 | {granite,haiku,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull30} | 100.0% (8/8) | 51.0% (26/51) | 84.3% |
| 7 | 6 | {gemma_v7primed,granite,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jpfull30} | 100.0% (8/8) | 49.0% (25/51) | 84.3% |
| 8 | 8 | {gpt_oss_20b,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 100.0% (8/8) | 47.1% (24/51) | 84.3% |

## Best subset per size (by weighted accuracy)

| size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {haiku} | 58.8% (30/51) | 0.492 | 58.8% (30/51) | 0.0% |
| 2 | {haiku,ministral} | 58.8% (30/51) | 0.492 | 73.3% (11/15) | 70.6% |
| 3 | {haiku,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 62.7% (32/51) | 0.541 | 47.6% (10/21) | 58.8% |
| 4 | {gpt_oss_20b,haiku,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull30} | 62.7% (32/51) | 0.541 | 76.5% (13/17) | 66.7% |
| 5 | {gemma,gpt_oss_20b,granite,ministral,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.592 | 66.7% (18/27) | 47.1% |
| 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 68.6% (35/51) | 0.616 | 70.6% (12/17) | 66.7% |
| 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 66.7% (34/51) | 0.591 | 70.8% (17/24) | 52.9% |
| 8 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull30} | 64.7% (33/51) | 0.566 | 81.0% (17/21) | 58.8% |
| 9 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 64.7% (33/51) | 0.569 | 77.8% (14/18) | 64.7% |
| 10 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 64.7% (33/51) | 0.566 | 81.2% (13/16) | 68.6% |
| 11 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 64.7% (33/51) | 0.566 | 65.2% (15/23) | 54.9% |
| 12 | {gemma,gemma_v7primed,granite,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 62.7% (32/51) | 0.543 | 80.0% (12/15) | 70.6% |
| 13 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 60.8% (31/51) | 0.520 | 81.2% (13/16) | 68.6% |
| 14 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 60.8% (31/51) | 0.519 | 64.7% (11/17) | 66.7% |
| 15 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 56.9% (29/51) | 0.472 | 60.0% (12/20) | 60.8% |

## Best subset per size (by κ)

| size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {haiku} | 0.492 | 58.8% (30/51) | 58.8% (30/51) | 0.0% |
| 2 | {gemma,granite} | 0.497 | 58.8% (30/51) | 63.2% (12/19) | 62.7% |
| 3 | {haiku,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 0.541 | 62.7% (32/51) | 47.6% (10/21) | 58.8% |
| 4 | {gemma,gpt_oss_20b,granite,rinna_bilingual_4b_jpfull30} | 0.545 | 62.7% (32/51) | 70.8% (17/24) | 52.9% |
| 5 | {gemma,gpt_oss_20b,granite,ministral,rinna_jp_3_6b_jpfull} | 0.592 | 66.7% (34/51) | 66.7% (18/27) | 47.1% |
| 6 | {gemma,gpt_oss_20b,granite,ministral,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 0.616 | 68.6% (35/51) | 70.6% (12/17) | 66.7% |
| 7 | {gemma,gpt_oss_20b,granite,ministral,qwen,rinna_bilingual_4b_jp,rinna_jp_3_6b_jpfull} | 0.593 | 66.7% (34/51) | 64.0% (16/25) | 51.0% |
| 8 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.569 | 64.7% (33/51) | 86.7% (13/15) | 70.6% |
| 9 | {gemma,gpt_oss_20b,haiku,ministral,rinna_bilingual_4b,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.569 | 64.7% (33/51) | 77.8% (14/18) | 64.7% |
| 10 | {gemma_v7primed,gpt_oss_20b,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 0.568 | 64.7% (33/51) | 53.3% (8/15) | 70.6% |
| 11 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.566 | 64.7% (33/51) | 65.2% (15/23) | 54.9% |
| 12 | {gemma,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.544 | 62.7% (32/51) | 69.2% (9/13) | 74.5% |
| 13 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.521 | 60.8% (31/51) | 72.7% (16/22) | 56.9% |
| 14 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.521 | 60.8% (31/51) | 70.6% (12/17) | 66.7% |
| 15 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.472 | 56.9% (29/51) | 60.0% (12/20) | 60.8% |

## Encoder inclusion frequency in top-8 weighted-acc

Heuristic: encoders that appear in nearly all top subsets are ensemble-load-bearing; encoders that rarely appear are individually weak AND fail to add complementary signal.

| encoder | top-K acc | top-K κ | solo acc | solo κ |
|---|---:|---:|---:|---:|
| gemma | 8/8 | 8/8 | 56.9% | 0.478 |
| gpt_oss_20b | 8/8 | 8/8 | 47.1% | 0.360 |
| rinna_jp_3_6b_jpfull | 8/8 | 8/8 | 33.3% | 0.216 |
| granite | 7/8 | 8/8 | 41.2% | 0.275 |
| ministral | 7/8 | 7/8 | 31.4% | 0.168 |
| qwen | 4/8 | 4/8 | 21.6% | 0.051 |
| rinna_bilingual_4b_jpfull30 | 3/8 | 2/8 | 35.3% | 0.212 |
| rinna_bilingual_4b_jp | 2/8 | 2/8 | 21.6% | 0.032 |
| rinna_bilingual_4b_jpfull | 1/8 | 2/8 | 23.5% | 0.111 |
| rinna_jp_3_6b_jp | 2/8 | 1/8 | 25.5% | 0.068 |
| haiku | 1/8 | 0/8 | 58.8% | 0.492 |
| rinna_bilingual_4b | 1/8 | 0/8 | 23.5% | 0.044 |
| rinna_jp_3_6b | 1/8 | 0/8 | 13.7% | 0.010 |
| gemma_v7primed | 0/8 | 0/8 | 49.0% | 0.381 |
| rinna_jp_3_6b_jpfull30 | 0/8 | 0/8 | 33.3% | 0.184 |
