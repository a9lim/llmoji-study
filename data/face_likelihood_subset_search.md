# Face_likelihood — exhaustive subset search

**Encoders:** 5  (gemma, gpt_oss_20b, granite, ministral, qwen)
**Faces (overlap):** 573
**GT subset (≥3 emits):** 166
**Subsets evaluated:** 31

## Headline

- Best single encoder by accuracy: **gemma** at 51.8% (86/166); Cohen's κ = 0.418
- Best weighted-vote subset by accuracy: **{gemma,gpt_oss_20b,granite,qwen}** at **54.8%** (91/166) — size 4, +3.0pp over best single; κ = 0.451
- Best weighted-vote subset by κ: **{gemma,gpt_oss_20b,granite,qwen}** at κ = **0.451** (accuracy 54.8%, size 4)
- Best strict-majority subset: **{gemma}** at 51.8% on 166 resolved (abstains on 0 all-distinct); κ = 0.418

**Reading κ:** Cohen's kappa corrects agreement for chance. 0.0 = no signal beyond random, 0.4–0.6 = moderate, 0.6–0.8 = substantial, >0.8 = near-perfect. Penalizes encoders that always predict the majority class — useful given GLM's 100%-LN bias. Voting models often have lower κ than accuracy because the vote concentrates predictions on common quadrants.

## Per-encoder solo accuracy + Cohen's κ vs empirical

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 51.8% (86/166) | 0.418 |
| gpt_oss_20b | 44.6% (74/166) | 0.334 |
| granite | 34.9% (58/166) | 0.205 |
| ministral | 31.9% (52/166) | 0.178 |
| qwen | 30.1% (50/166) | 0.167 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Useful for ensemble design: encoder pairs with low κ make complementary errors and are more useful to combine than encoder pairs with high κ.

| pair | κ |
|---|---:|
| gemma ↔ gpt_oss_20b | 0.248 |
| granite ↔ ministral | 0.185 |
| gpt_oss_20b ↔ granite | 0.148 |
| gemma ↔ ministral | 0.141 |
| gpt_oss_20b ↔ ministral | 0.124 |
| gemma ↔ granite | 0.124 |
| gpt_oss_20b ↔ qwen | 0.114 |
| gemma ↔ qwen | 0.089 |
| ministral ↔ qwen | 0.087 |
| granite ↔ qwen | 0.064 |

## Top 25 subsets by weighted-vote accuracy

| rank | size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 4 | {gemma,gpt_oss_20b,granite,qwen} | 54.8% (91/166) | 0.451 | 66.7% (38/57) | 65.7% |
| 2 | 3 | {gemma,ministral,qwen} | 54.2% (90/166) | 0.446 | 52.5% (53/101) | 39.2% |
| 3 | 2 | {gemma,ministral} | 52.4% (87/166) | 0.425 | 57.7% (30/52) | 68.7% |
| 4 | 1 | {gemma} | 51.8% (86/166) | 0.418 | 51.8% (86/166) | 0.0% |
| 5 | 4 | {gemma,gpt_oss_20b,ministral,qwen} | 51.8% (86/166) | 0.418 | 62.5% (35/56) | 66.3% |
| 6 | 3 | {gemma,gpt_oss_20b,qwen} | 51.8% (86/166) | 0.418 | 57.0% (69/121) | 27.1% |
| 7 | 3 | {gemma,gpt_oss_20b,granite} | 51.8% (86/166) | 0.415 | 61.3% (73/119) | 28.3% |
| 8 | 3 | {gemma,gpt_oss_20b,ministral} | 51.2% (85/166) | 0.411 | 61.7% (74/120) | 27.7% |
| 9 | 2 | {gemma,qwen} | 51.2% (85/166) | 0.411 | 62.2% (28/45) | 72.9% |
| 10 | 2 | {gemma,gpt_oss_20b} | 51.2% (85/166) | 0.411 | 66.7% (52/78) | 53.0% |
| 11 | 2 | {gemma,granite} | 50.6% (84/166) | 0.400 | 66.0% (33/50) | 69.9% |
| 12 | 5 | {gemma,gpt_oss_20b,granite,ministral,qwen} | 50.6% (84/166) | 0.400 | 62.7% (52/83) | 50.0% |
| 13 | 4 | {gemma,gpt_oss_20b,granite,ministral} | 50.0% (83/166) | 0.392 | 66.7% (38/57) | 65.7% |
| 14 | 3 | {gemma,granite,qwen} | 48.8% (81/166) | 0.378 | 57.0% (53/93) | 44.0% |
| 15 | 2 | {gpt_oss_20b,qwen} | 47.0% (78/166) | 0.363 | 52.1% (25/48) | 71.1% |
| 16 | 3 | {gemma,granite,ministral} | 46.4% (77/166) | 0.349 | 48.7% (55/113) | 31.9% |
| 17 | 1 | {gpt_oss_20b} | 44.6% (74/166) | 0.334 | 44.6% (74/166) | 0.0% |
| 18 | 4 | {gemma,granite,ministral,qwen} | 44.6% (74/166) | 0.326 | 71.7% (33/46) | 72.3% |
| 19 | 3 | {gpt_oss_20b,ministral,qwen} | 41.6% (69/166) | 0.296 | 51.1% (48/94) | 43.4% |
| 20 | 4 | {gpt_oss_20b,granite,ministral,qwen} | 38.6% (64/166) | 0.251 | 61.5% (32/52) | 68.7% |
| 21 | 2 | {gpt_oss_20b,ministral} | 37.3% (62/166) | 0.245 | 57.1% (24/42) | 74.7% |
| 22 | 3 | {granite,ministral,qwen} | 36.7% (61/166) | 0.229 | 43.7% (45/103) | 38.0% |
| 23 | 3 | {gpt_oss_20b,granite,ministral} | 36.7% (61/166) | 0.229 | 48.1% (52/108) | 34.9% |
| 24 | 2 | {granite,qwen} | 35.5% (59/166) | 0.213 | 58.8% (20/34) | 79.5% |
| 25 | 1 | {granite} | 34.9% (58/166) | 0.205 | 34.9% (58/166) | 0.0% |

## Top 25 subsets by weighted-vote Cohen's κ

(Class-imbalanced subsets that ride the empirical majority-class base rate score lower here than under raw accuracy.)

| rank | size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 4 | {gemma,gpt_oss_20b,granite,qwen} | 0.451 | 54.8% (91/166) | 66.7% (38/57) | 65.7% |
| 2 | 3 | {gemma,ministral,qwen} | 0.446 | 54.2% (90/166) | 52.5% (53/101) | 39.2% |
| 3 | 2 | {gemma,ministral} | 0.425 | 52.4% (87/166) | 57.7% (30/52) | 68.7% |
| 4 | 3 | {gemma,gpt_oss_20b,qwen} | 0.418 | 51.8% (86/166) | 57.0% (69/121) | 27.1% |
| 5 | 4 | {gemma,gpt_oss_20b,ministral,qwen} | 0.418 | 51.8% (86/166) | 62.5% (35/56) | 66.3% |
| 6 | 1 | {gemma} | 0.418 | 51.8% (86/166) | 51.8% (86/166) | 0.0% |
| 7 | 3 | {gemma,gpt_oss_20b,granite} | 0.415 | 51.8% (86/166) | 61.3% (73/119) | 28.3% |
| 8 | 2 | {gemma,qwen} | 0.411 | 51.2% (85/166) | 62.2% (28/45) | 72.9% |
| 9 | 2 | {gemma,gpt_oss_20b} | 0.411 | 51.2% (85/166) | 66.7% (52/78) | 53.0% |
| 10 | 3 | {gemma,gpt_oss_20b,ministral} | 0.411 | 51.2% (85/166) | 61.7% (74/120) | 27.7% |
| 11 | 5 | {gemma,gpt_oss_20b,granite,ministral,qwen} | 0.400 | 50.6% (84/166) | 62.7% (52/83) | 50.0% |
| 12 | 2 | {gemma,granite} | 0.400 | 50.6% (84/166) | 66.0% (33/50) | 69.9% |
| 13 | 4 | {gemma,gpt_oss_20b,granite,ministral} | 0.392 | 50.0% (83/166) | 66.7% (38/57) | 65.7% |
| 14 | 3 | {gemma,granite,qwen} | 0.378 | 48.8% (81/166) | 57.0% (53/93) | 44.0% |
| 15 | 2 | {gpt_oss_20b,qwen} | 0.363 | 47.0% (78/166) | 52.1% (25/48) | 71.1% |
| 16 | 3 | {gemma,granite,ministral} | 0.349 | 46.4% (77/166) | 48.7% (55/113) | 31.9% |
| 17 | 1 | {gpt_oss_20b} | 0.334 | 44.6% (74/166) | 44.6% (74/166) | 0.0% |
| 18 | 4 | {gemma,granite,ministral,qwen} | 0.326 | 44.6% (74/166) | 71.7% (33/46) | 72.3% |
| 19 | 3 | {gpt_oss_20b,ministral,qwen} | 0.296 | 41.6% (69/166) | 51.1% (48/94) | 43.4% |
| 20 | 4 | {gpt_oss_20b,granite,ministral,qwen} | 0.251 | 38.6% (64/166) | 61.5% (32/52) | 68.7% |
| 21 | 2 | {gpt_oss_20b,ministral} | 0.245 | 37.3% (62/166) | 57.1% (24/42) | 74.7% |
| 22 | 3 | {gpt_oss_20b,granite,ministral} | 0.229 | 36.7% (61/166) | 48.1% (52/108) | 34.9% |
| 23 | 3 | {granite,ministral,qwen} | 0.229 | 36.7% (61/166) | 43.7% (45/103) | 38.0% |
| 24 | 2 | {granite,qwen} | 0.213 | 35.5% (59/166) | 58.8% (20/34) | 79.5% |
| 25 | 2 | {granite,ministral} | 0.206 | 34.9% (58/166) | 44.8% (30/67) | 59.6% |

## Top 25 subsets by strict-majority accuracy

(ties broken by larger n_resolved, i.e. more decisive)

| rank | size | encoders | majority(resolved) | weighted | abstain |
|---:|---:|---|---:|---:|---:|
| 1 | 4 | {gemma,granite,ministral,qwen} | 71.7% (33/46) | 44.6% (74/166) | 72.3% |
| 2 | 2 | {gemma,gpt_oss_20b} | 66.7% (52/78) | 51.2% (85/166) | 53.0% |
| 3 | 4 | {gemma,gpt_oss_20b,granite,qwen} | 66.7% (38/57) | 54.8% (91/166) | 65.7% |
| 4 | 4 | {gemma,gpt_oss_20b,granite,ministral} | 66.7% (38/57) | 50.0% (83/166) | 65.7% |
| 5 | 2 | {gemma,granite} | 66.0% (33/50) | 50.6% (84/166) | 69.9% |
| 6 | 5 | {gemma,gpt_oss_20b,granite,ministral,qwen} | 62.7% (52/83) | 50.6% (84/166) | 50.0% |
| 7 | 4 | {gemma,gpt_oss_20b,ministral,qwen} | 62.5% (35/56) | 51.8% (86/166) | 66.3% |
| 8 | 2 | {gemma,qwen} | 62.2% (28/45) | 51.2% (85/166) | 72.9% |
| 9 | 3 | {gemma,gpt_oss_20b,ministral} | 61.7% (74/120) | 51.2% (85/166) | 27.7% |
| 10 | 4 | {gpt_oss_20b,granite,ministral,qwen} | 61.5% (32/52) | 38.6% (64/166) | 68.7% |
| 11 | 3 | {gemma,gpt_oss_20b,granite} | 61.3% (73/119) | 51.8% (86/166) | 28.3% |
| 12 | 2 | {granite,qwen} | 58.8% (20/34) | 35.5% (59/166) | 79.5% |
| 13 | 2 | {gpt_oss_20b,granite} | 58.2% (32/55) | 34.9% (58/166) | 66.9% |
| 14 | 2 | {gemma,ministral} | 57.7% (30/52) | 52.4% (87/166) | 68.7% |
| 15 | 2 | {gpt_oss_20b,ministral} | 57.1% (24/42) | 37.3% (62/166) | 74.7% |
| 16 | 3 | {gemma,gpt_oss_20b,qwen} | 57.0% (69/121) | 51.8% (86/166) | 27.1% |
| 17 | 3 | {gemma,granite,qwen} | 57.0% (53/93) | 48.8% (81/166) | 44.0% |
| 18 | 2 | {ministral,qwen} | 52.8% (19/36) | 31.9% (53/166) | 78.3% |
| 19 | 3 | {gemma,ministral,qwen} | 52.5% (53/101) | 54.2% (90/166) | 39.2% |
| 20 | 2 | {gpt_oss_20b,qwen} | 52.1% (25/48) | 47.0% (78/166) | 71.1% |
| 21 | 1 | {gemma} | 51.8% (86/166) | 51.8% (86/166) | 0.0% |
| 22 | 3 | {gpt_oss_20b,granite,qwen} | 51.6% (49/95) | 34.3% (57/166) | 42.8% |
| 23 | 3 | {gpt_oss_20b,ministral,qwen} | 51.1% (48/94) | 41.6% (69/166) | 43.4% |
| 24 | 3 | {gemma,granite,ministral} | 48.7% (55/113) | 46.4% (77/166) | 31.9% |
| 25 | 3 | {gpt_oss_20b,granite,ministral} | 48.1% (52/108) | 36.7% (61/166) | 34.9% |

## Best subset per size (by weighted accuracy)

| size | encoders | acc | κ | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {gemma} | 51.8% (86/166) | 0.418 | 51.8% (86/166) | 0.0% |
| 2 | {gemma,ministral} | 52.4% (87/166) | 0.425 | 57.7% (30/52) | 68.7% |
| 3 | {gemma,ministral,qwen} | 54.2% (90/166) | 0.446 | 52.5% (53/101) | 39.2% |
| 4 | {gemma,gpt_oss_20b,granite,qwen} | 54.8% (91/166) | 0.451 | 66.7% (38/57) | 65.7% |
| 5 | {gemma,gpt_oss_20b,granite,ministral,qwen} | 50.6% (84/166) | 0.400 | 62.7% (52/83) | 50.0% |

## Best subset per size (by κ)

| size | encoders | κ | acc | majority(resolved) | abstain |
|---:|---|---:|---:|---:|---:|
| 1 | {gemma} | 0.418 | 51.8% (86/166) | 51.8% (86/166) | 0.0% |
| 2 | {gemma,ministral} | 0.425 | 52.4% (87/166) | 57.7% (30/52) | 68.7% |
| 3 | {gemma,ministral,qwen} | 0.446 | 54.2% (90/166) | 52.5% (53/101) | 39.2% |
| 4 | {gemma,gpt_oss_20b,granite,qwen} | 0.451 | 54.8% (91/166) | 66.7% (38/57) | 65.7% |
| 5 | {gemma,gpt_oss_20b,granite,ministral,qwen} | 0.400 | 50.6% (84/166) | 62.7% (52/83) | 50.0% |

## Encoder inclusion frequency in top-25 weighted-acc

Heuristic: encoders that appear in nearly all top subsets are ensemble-load-bearing; encoders that rarely appear are individually weak AND fail to add complementary signal.

| encoder | top-K acc | top-K κ | solo acc | solo κ |
|---|---:|---:|---:|---:|
| gemma | 16/25 | 16/25 | 51.8% | 0.418 |
| gpt_oss_20b | 14/25 | 14/25 | 44.6% | 0.334 |
| ministral | 13/25 | 14/25 | 31.9% | 0.178 |
| granite | 13/25 | 13/25 | 34.9% | 0.205 |
| qwen | 13/25 | 13/25 | 30.1% | 0.167 |
