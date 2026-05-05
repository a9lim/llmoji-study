# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 9  (gemma, gemma_v7primed, gpt_oss_20b, granite, haiku, ministral, qwen, rinna_bilingual_4b_jpfull, rinna_jp_3_6b_jpfull)
**Faces (overlap):** 164
**GT subset (Claude empirical, total ≥ 1):** 49
**Subsets evaluated:** 511

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **gemma** at **face-uniform similarity = 0.658** (emit-weighted 0.706)
- Best ensemble subset: **{gemma,haiku}** at **face-uniform similarity = 0.702** (emit-weighted 0.770); size 2; Δ vs best solo (face-uniform) = +0.044

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| gemma | 0.658 | 0.706 | 0.2369 |
| haiku | 0.655 | 0.734 | 0.2394 |
| gemma_v7primed | 0.640 | 0.754 | 0.2494 |
| gpt_oss_20b | 0.532 | 0.661 | 0.3243 |
| ministral | 0.490 | 0.674 | 0.3532 |
| qwen | 0.445 | 0.567 | 0.3844 |
| granite | 0.434 | 0.565 | 0.3922 |
| rinna_bilingual_4b_jpfull | 0.427 | 0.508 | 0.3973 |
| rinna_jp_3_6b_jpfull | 0.416 | 0.550 | 0.4050 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ gemma_v7primed | 0.770 |
| gemma ↔ haiku | 0.385 |
| gemma_v7primed ↔ haiku | 0.362 |
| gpt_oss_20b ↔ haiku | 0.361 |
| gemma ↔ gpt_oss_20b | 0.341 |
| gemma_v7primed ↔ gpt_oss_20b | 0.317 |
| granite ↔ ministral | 0.258 |
| gpt_oss_20b ↔ granite | 0.198 |
| gemma ↔ ministral | 0.197 |
| gemma ↔ granite | 0.178 |
| granite ↔ haiku | 0.169 |
| gemma_v7primed ↔ granite | 0.168 |
| gemma_v7primed ↔ ministral | 0.162 |
| haiku ↔ qwen | 0.154 |
| gpt_oss_20b ↔ qwen | 0.152 |
| gemma_v7primed ↔ qwen | 0.144 |
| gpt_oss_20b ↔ ministral | 0.142 |
| haiku ↔ ministral | 0.142 |
| gemma ↔ qwen | 0.131 |
| ministral ↔ qwen | 0.112 |
| qwen ↔ rinna_bilingual_4b_jpfull | 0.111 |
| granite ↔ rinna_jp_3_6b_jpfull | 0.089 |
| haiku ↔ rinna_jp_3_6b_jpfull | 0.085 |
| granite ↔ qwen | 0.080 |
| gemma_v7primed ↔ rinna_bilingual_4b_jpfull | 0.059 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull | 0.059 |
| haiku ↔ rinna_bilingual_4b_jpfull | 0.041 |
| gemma ↔ rinna_bilingual_4b_jpfull | 0.036 |
| gemma_v7primed ↔ rinna_jp_3_6b_jpfull | 0.034 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull | 0.034 |
| gemma ↔ rinna_jp_3_6b_jpfull | 0.031 |
| ministral ↔ rinna_jp_3_6b_jpfull | 0.014 |
| qwen ↔ rinna_jp_3_6b_jpfull | 0.005 |
| ministral ↔ rinna_bilingual_4b_jpfull | 0.003 |
| granite ↔ rinna_bilingual_4b_jpfull | -0.012 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull | -0.018 |

## Top 10 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 2 | {gemma,haiku} | 0.702 | 0.770 |
| 2 | 3 | {gemma,gemma_v7primed,haiku} | 0.695 | 0.777 |
| 3 | 2 | {gemma_v7primed,haiku} | 0.688 | 0.782 |
| 4 | 4 | {gemma,gemma_v7primed,gpt_oss_20b,haiku} | 0.675 | 0.782 |
| 5 | 4 | {gemma,gemma_v7primed,haiku,ministral} | 0.672 | 0.782 |
| 6 | 4 | {gemma,gemma_v7primed,granite,haiku} | 0.670 | 0.762 |
| 7 | 3 | {gemma,gpt_oss_20b,haiku} | 0.669 | 0.770 |
| 8 | 3 | {gemma,haiku,ministral} | 0.665 | 0.773 |
| 9 | 4 | {gemma,gemma_v7primed,haiku,qwen} | 0.662 | 0.763 |
| 10 | 3 | {gemma,granite,haiku} | 0.662 | 0.746 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {gemma} | 0.658 | 0.706 |
| 2 | {gemma,haiku} | 0.702 | 0.770 |
| 3 | {gemma,gemma_v7primed,haiku} | 0.695 | 0.777 |
| 4 | {gemma,gemma_v7primed,gpt_oss_20b,haiku} | 0.675 | 0.782 |
| 5 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku} | 0.657 | 0.767 |
| 6 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral} | 0.641 | 0.763 |
| 7 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,rinna_bilingual_4b_jpfull} | 0.623 | 0.742 |
| 8 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jpfull} | 0.608 | 0.729 |
| 9 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 0.594 | 0.716 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| haiku | 65.3% (31/49) | 0.559 |
| gemma | 61.2% (30/49) | 0.523 |
| gemma_v7primed | 53.1% (26/49) | 0.422 |
| gpt_oss_20b | 44.9% (22/49) | 0.329 |
| granite | 36.7% (18/49) | 0.251 |
| rinna_jp_3_6b_jpfull | 26.5% (13/49) | 0.170 |
| ministral | 24.5% (12/49) | 0.100 |
| qwen | 22.4% (11/49) | 0.049 |
| rinna_bilingual_4b_jpfull | 16.3% (7/49) | 0.058 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 2 | {gpt_oss_20b,haiku} | 65.3% (32/49) | 0.559 | 0.626 |
| 4 | {gemma,gpt_oss_20b,haiku,rinna_jp_3_6b_jpfull} | 65.3% (32/49) | 0.566 | 0.630 |
| 3 | {haiku,ministral,rinna_bilingual_4b_jpfull} | 65.3% (32/49) | 0.563 | 0.581 |
| 4 | {gemma,gpt_oss_20b,haiku,ministral} | 65.3% (32/49) | 0.566 | 0.642 |
| 6 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,qwen} | 65.3% (32/49) | 0.567 | 0.635 |
| 5 | {gemma,gpt_oss_20b,haiku,ministral,qwen} | 65.3% (32/49) | 0.566 | 0.614 |
| 3 | {gpt_oss_20b,haiku,qwen} | 65.3% (32/49) | 0.559 | 0.581 |
| 5 | {gemma,gpt_oss_20b,granite,haiku,qwen} | 65.3% (32/49) | 0.567 | 0.619 |
| 4 | {gemma,gpt_oss_20b,granite,haiku} | 65.3% (32/49) | 0.567 | 0.645 |
| 3 | {haiku,ministral,qwen} | 65.3% (32/49) | 0.564 | 0.576 |

