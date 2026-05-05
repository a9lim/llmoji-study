# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 12  (gemma, gemma_intro_v7_primed, gpt_oss_20b, granite, haiku, ministral, opus, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_jp_3_6b, rinna_jp_3_6b_jp)
**Faces (overlap):** 85
**GT subset (≥3 pooled emits):** 71
**Subsets evaluated:** 4095

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **opus** at **face-uniform similarity = 0.783** (emit-weighted 0.855)
- Best ensemble subset: **{gemma,gemma_intro_v7_primed,opus}** at **face-uniform similarity = 0.829** (emit-weighted 0.869); size 3; Δ vs best solo (face-uniform) = +0.047

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| opus | 0.783 | 0.855 | 0.1507 |
| gemma_intro_v7_primed | 0.774 | 0.794 | 0.1569 |
| gemma | 0.771 | 0.788 | 0.1588 |
| haiku | 0.723 | 0.814 | 0.1917 |
| gpt_oss_20b | 0.697 | 0.797 | 0.2097 |
| ministral | 0.636 | 0.758 | 0.2524 |
| qwen | 0.596 | 0.706 | 0.2800 |
| rinna_jp_3_6b_jp | 0.556 | 0.625 | 0.3080 |
| rinna_bilingual_4b | 0.554 | 0.642 | 0.3092 |
| granite | 0.553 | 0.678 | 0.3096 |
| rinna_jp_3_6b | 0.542 | 0.656 | 0.3175 |
| rinna_bilingual_4b_jp | 0.479 | 0.536 | 0.3611 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ gemma_intro_v7_primed | 0.788 |
| haiku ↔ opus | 0.575 |
| gemma_intro_v7_primed ↔ opus | 0.547 |
| gemma ↔ opus | 0.535 |
| gemma ↔ gpt_oss_20b | 0.506 |
| gemma_intro_v7_primed ↔ gpt_oss_20b | 0.460 |
| gemma_intro_v7_primed ↔ haiku | 0.456 |
| gpt_oss_20b ↔ opus | 0.431 |
| gemma ↔ haiku | 0.421 |
| gpt_oss_20b ↔ haiku | 0.368 |
| granite ↔ opus | 0.317 |
| gpt_oss_20b ↔ granite | 0.305 |
| granite ↔ ministral | 0.276 |
| gemma_intro_v7_primed ↔ granite | 0.250 |
| gemma ↔ granite | 0.235 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jp | 0.195 |
| gemma_intro_v7_primed ↔ qwen | 0.178 |
| haiku ↔ qwen | 0.172 |
| gemma ↔ ministral | 0.171 |
| granite ↔ haiku | 0.167 |
| gemma_intro_v7_primed ↔ rinna_jp_3_6b_jp | 0.153 |
| gemma_intro_v7_primed ↔ ministral | 0.145 |
| ministral ↔ opus | 0.145 |
| gpt_oss_20b ↔ ministral | 0.141 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jp | 0.134 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jp | 0.131 |
| gemma ↔ qwen | 0.126 |
| opus ↔ qwen | 0.126 |
| opus ↔ rinna_jp_3_6b_jp | 0.116 |
| opus ↔ rinna_bilingual_4b | 0.110 |
| haiku ↔ rinna_jp_3_6b_jp | 0.106 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jp | 0.101 |
| ministral ↔ qwen | 0.100 |
| gpt_oss_20b ↔ qwen | 0.100 |
| gemma ↔ rinna_jp_3_6b_jp | 0.091 |
| granite ↔ qwen | 0.082 |
| granite ↔ rinna_bilingual_4b | 0.076 |
| qwen ↔ rinna_bilingual_4b_jp | 0.066 |
| gemma_intro_v7_primed ↔ rinna_bilingual_4b_jp | 0.059 |
| granite ↔ rinna_jp_3_6b_jp | 0.059 |
| gemma_intro_v7_primed ↔ rinna_bilingual_4b | 0.058 |
| ministral ↔ rinna_bilingual_4b_jp | 0.058 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jp | 0.053 |
| qwen ↔ rinna_jp_3_6b_jp | 0.052 |
| gemma ↔ rinna_bilingual_4b | 0.049 |
| gemma ↔ rinna_bilingual_4b_jp | 0.044 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b | 0.043 |
| haiku ↔ rinna_bilingual_4b_jp | 0.043 |
| haiku ↔ ministral | 0.042 |
| gpt_oss_20b ↔ rinna_bilingual_4b | 0.037 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jp | 0.033 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b | 0.026 |
| opus ↔ rinna_bilingual_4b_jp | 0.020 |
| ministral ↔ rinna_jp_3_6b_jp | 0.017 |
| qwen ↔ rinna_jp_3_6b | 0.012 |
| haiku ↔ rinna_bilingual_4b | 0.007 |
| qwen ↔ rinna_bilingual_4b | 0.007 |
| granite ↔ rinna_bilingual_4b_jp | 0.005 |
| ministral ↔ rinna_jp_3_6b | 0.002 |
| granite ↔ rinna_jp_3_6b | 0.000 |
| gemma_intro_v7_primed ↔ rinna_jp_3_6b | -0.002 |
| gemma ↔ rinna_jp_3_6b | -0.007 |
| ministral ↔ rinna_bilingual_4b | -0.007 |
| opus ↔ rinna_jp_3_6b | -0.010 |
| haiku ↔ rinna_jp_3_6b | -0.014 |
| gpt_oss_20b ↔ rinna_jp_3_6b | -0.058 |

## Top 5 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 3 | {gemma,gemma_intro_v7_primed,opus} | 0.829 | 0.869 |
| 2 | 2 | {gemma,opus} | 0.828 | 0.873 |
| 3 | 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.827 | 0.875 |
| 4 | 2 | {gemma_intro_v7_primed,opus} | 0.826 | 0.873 |
| 5 | 4 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,opus} | 0.826 | 0.889 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {opus} | 0.783 | 0.855 |
| 2 | {gemma,opus} | 0.828 | 0.873 |
| 3 | {gemma,gemma_intro_v7_primed,opus} | 0.829 | 0.869 |
| 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.827 | 0.875 |
| 5 | {gemma,gemma_intro_v7_primed,haiku,ministral,opus} | 0.823 | 0.895 |
| 6 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,haiku,ministral,opus} | 0.812 | 0.891 |
| 7 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus} | 0.802 | 0.884 |
| 8 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_jp_3_6b_jp} | 0.787 | 0.872 |
| 9 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_jp_3_6b_jp} | 0.774 | 0.865 |
| 10 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b_jp} | 0.760 | 0.854 |
| 11 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.748 | 0.847 |
| 12 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.736 | 0.835 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 73.2% (52/71) | 0.676 |
| gemma_intro_v7_primed | 67.6% (48/71) | 0.606 |
| opus | 63.4% (45/71) | 0.547 |
| haiku | 52.1% (37/71) | 0.406 |
| gpt_oss_20b | 49.3% (35/71) | 0.390 |
| granite | 38.0% (27/71) | 0.244 |
| qwen | 31.0% (22/71) | 0.167 |
| ministral | 29.6% (21/71) | 0.156 |
| rinna_jp_3_6b_jp | 29.6% (21/71) | 0.120 |
| rinna_bilingual_4b | 26.8% (19/71) | 0.083 |
| rinna_bilingual_4b_jp | 21.1% (15/71) | 0.032 |
| rinna_jp_3_6b | 12.7% (9/71) | -0.043 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 7 | {gemma,gemma_intro_v7_primed,granite,ministral,opus,rinna_bilingual_4b,rinna_jp_3_6b} | 77.5% (55/71) | 0.724 | 0.766 |
| 6 | {gemma,gemma_intro_v7_primed,granite,opus,rinna_bilingual_4b,rinna_jp_3_6b} | 77.5% (55/71) | 0.724 | 0.772 |
| 5 | {gemma,gemma_intro_v7_primed,granite,opus,rinna_bilingual_4b} | 77.5% (55/71) | 0.724 | 0.792 |
| 5 | {gemma,gemma_intro_v7_primed,granite,opus,rinna_jp_3_6b} | 77.5% (55/71) | 0.724 | 0.794 |
| 7 | {gemma,gemma_intro_v7_primed,granite,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b} | 77.5% (55/71) | 0.724 | 0.758 |
| 7 | {gemma,gemma_intro_v7_primed,granite,haiku,ministral,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 77.5% (55/71) | 0.724 | 0.767 |
| 8 | {gemma,gemma_intro_v7_primed,granite,haiku,ministral,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 77.5% (55/71) | 0.724 | 0.753 |
| 6 | {gemma,gemma_intro_v7_primed,granite,opus,qwen,rinna_bilingual_4b} | 77.5% (55/71) | 0.724 | 0.777 |
| 6 | {gemma,gemma_intro_v7_primed,granite,opus,qwen,rinna_jp_3_6b} | 77.5% (55/71) | 0.724 | 0.777 |
| 6 | {gemma,gpt_oss_20b,granite,opus,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 76.1% (54/71) | 0.707 | 0.744 |

