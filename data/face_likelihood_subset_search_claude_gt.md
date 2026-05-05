# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 16  (gemma, gemma_v7primed, gpt_oss_20b, granite, haiku, ministral, opus, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_bilingual_4b_jpfull, rinna_bilingual_4b_jpfull30, rinna_jp_3_6b, rinna_jp_3_6b_jp, rinna_jp_3_6b_jpfull, rinna_jp_3_6b_jpfull30)
**Faces (overlap):** 85
**GT subset (Claude empirical, total ≥ 3):** 49
**Subsets evaluated:** 65535

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **gemma_v7primed** at **face-uniform similarity = 0.777** (emit-weighted 0.801)
- Best ensemble subset: **{gemma_v7primed,opus}** at **face-uniform similarity = 0.788** (emit-weighted 0.829); size 2; Δ vs best solo (face-uniform) = +0.011

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| gemma_v7primed | 0.777 | 0.801 | 0.1548 |
| gemma | 0.746 | 0.755 | 0.1760 |
| opus | 0.739 | 0.797 | 0.1810 |
| haiku | 0.681 | 0.723 | 0.2214 |
| gpt_oss_20b | 0.597 | 0.667 | 0.2793 |
| rinna_bilingual_4b_jpfull30 | 0.557 | 0.560 | 0.3071 |
| ministral | 0.532 | 0.579 | 0.3247 |
| rinna_bilingual_4b_jpfull | 0.517 | 0.543 | 0.3345 |
| rinna_jp_3_6b_jp | 0.512 | 0.504 | 0.3380 |
| rinna_jp_3_6b_jpfull30 | 0.512 | 0.519 | 0.3384 |
| granite | 0.506 | 0.586 | 0.3421 |
| rinna_jp_3_6b_jpfull | 0.505 | 0.544 | 0.3430 |
| rinna_bilingual_4b | 0.492 | 0.505 | 0.3519 |
| qwen | 0.484 | 0.536 | 0.3575 |
| rinna_jp_3_6b | 0.465 | 0.523 | 0.3707 |
| rinna_bilingual_4b_jp | 0.404 | 0.397 | 0.4133 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ gemma_v7primed | 0.788 |
| haiku ↔ opus | 0.575 |
| gemma_v7primed ↔ opus | 0.547 |
| gemma ↔ opus | 0.535 |
| gemma ↔ gpt_oss_20b | 0.506 |
| gemma_v7primed ↔ gpt_oss_20b | 0.460 |
| gemma_v7primed ↔ haiku | 0.456 |
| gpt_oss_20b ↔ opus | 0.431 |
| rinna_bilingual_4b_jpfull ↔ rinna_bilingual_4b_jpfull30 | 0.424 |
| gemma ↔ haiku | 0.421 |
| rinna_jp_3_6b_jpfull ↔ rinna_jp_3_6b_jpfull30 | 0.370 |
| gpt_oss_20b ↔ haiku | 0.368 |
| granite ↔ opus | 0.317 |
| gpt_oss_20b ↔ granite | 0.305 |
| granite ↔ ministral | 0.276 |
| gemma_v7primed ↔ granite | 0.250 |
| gemma ↔ granite | 0.235 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull30 | 0.219 |
| gemma_v7primed ↔ rinna_bilingual_4b_jpfull30 | 0.219 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull | 0.216 |
| gemma_v7primed ↔ rinna_jp_3_6b_jpfull30 | 0.213 |
| opus ↔ rinna_jp_3_6b_jpfull | 0.209 |
| gemma ↔ rinna_jp_3_6b_jpfull30 | 0.208 |
| haiku ↔ rinna_bilingual_4b_jpfull30 | 0.208 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.204 |
| opus ↔ rinna_jp_3_6b_jpfull30 | 0.202 |
| gemma_v7primed ↔ rinna_jp_3_6b_jpfull | 0.197 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jp | 0.195 |
| qwen ↔ rinna_bilingual_4b_jpfull | 0.191 |
| opus ↔ rinna_bilingual_4b_jpfull30 | 0.184 |
| gemma_v7primed ↔ qwen | 0.178 |
| haiku ↔ qwen | 0.172 |
| gemma ↔ ministral | 0.171 |
| granite ↔ haiku | 0.167 |
| haiku ↔ rinna_jp_3_6b_jpfull | 0.165 |
| gemma ↔ rinna_jp_3_6b_jpfull | 0.163 |
| gemma_v7primed ↔ rinna_jp_3_6b_jp | 0.153 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull30 | 0.151 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull | 0.147 |
| gemma ↔ rinna_bilingual_4b_jpfull30 | 0.146 |
| gemma_v7primed ↔ ministral | 0.145 |
| ministral ↔ opus | 0.145 |
| gpt_oss_20b ↔ ministral | 0.141 |
| granite ↔ rinna_jp_3_6b_jpfull30 | 0.140 |
| haiku ↔ rinna_bilingual_4b_jpfull | 0.136 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jp | 0.134 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jp | 0.131 |
| granite ↔ rinna_jp_3_6b_jpfull | 0.130 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull | 0.126 |
| gemma ↔ qwen | 0.126 |
| opus ↔ qwen | 0.126 |
| opus ↔ rinna_jp_3_6b_jp | 0.116 |
| rinna_jp_3_6b_jp ↔ rinna_jp_3_6b_jpfull | 0.113 |
| haiku ↔ rinna_jp_3_6b_jpfull30 | 0.110 |
| ministral ↔ rinna_jp_3_6b_jpfull30 | 0.110 |
| opus ↔ rinna_bilingual_4b | 0.110 |
| gemma_v7primed ↔ rinna_bilingual_4b_jpfull | 0.109 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull30 | 0.107 |
| opus ↔ rinna_bilingual_4b_jpfull | 0.106 |
| haiku ↔ rinna_jp_3_6b_jp | 0.106 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jp | 0.101 |
| ministral ↔ qwen | 0.100 |
| gpt_oss_20b ↔ qwen | 0.100 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull | 0.099 |
| gemma ↔ rinna_jp_3_6b_jp | 0.091 |
| ministral ↔ rinna_jp_3_6b_jpfull | 0.089 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jpfull30 | 0.083 |
| granite ↔ qwen | 0.082 |
| qwen ↔ rinna_jp_3_6b_jpfull30 | 0.082 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b_jp | 0.077 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull | 0.077 |
| granite ↔ rinna_bilingual_4b | 0.076 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull30 | 0.073 |
| gemma ↔ rinna_bilingual_4b_jpfull | 0.070 |
| qwen ↔ rinna_bilingual_4b_jp | 0.066 |
| rinna_bilingual_4b_jp ↔ rinna_bilingual_4b_jpfull | 0.062 |
| gemma_v7primed ↔ rinna_bilingual_4b_jp | 0.059 |
| granite ↔ rinna_jp_3_6b_jp | 0.059 |
| gemma_v7primed ↔ rinna_bilingual_4b | 0.058 |
| ministral ↔ rinna_bilingual_4b_jp | 0.058 |
| qwen ↔ rinna_bilingual_4b_jpfull30 | 0.055 |
| ministral ↔ rinna_bilingual_4b_jpfull30 | 0.053 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jp | 0.053 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull30 | 0.052 |
| qwen ↔ rinna_jp_3_6b_jp | 0.052 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jpfull30 | 0.049 |
| gemma ↔ rinna_bilingual_4b | 0.049 |
| rinna_bilingual_4b_jpfull30 ↔ rinna_jp_3_6b | 0.048 |
| gemma ↔ rinna_bilingual_4b_jp | 0.044 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b | 0.043 |
| haiku ↔ rinna_bilingual_4b_jp | 0.043 |
| qwen ↔ rinna_jp_3_6b_jpfull | 0.043 |
| haiku ↔ ministral | 0.042 |
| gpt_oss_20b ↔ rinna_bilingual_4b | 0.037 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull30 | 0.034 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jp | 0.033 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b | 0.033 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b | 0.026 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jpfull | 0.023 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jp | 0.023 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jpfull | 0.022 |
| opus ↔ rinna_bilingual_4b_jp | 0.020 |
| granite ↔ rinna_bilingual_4b_jpfull30 | 0.020 |
| ministral ↔ rinna_jp_3_6b_jp | 0.017 |
| qwen ↔ rinna_jp_3_6b | 0.012 |
| ministral ↔ rinna_bilingual_4b_jpfull | 0.009 |
| haiku ↔ rinna_bilingual_4b | 0.007 |
| qwen ↔ rinna_bilingual_4b | 0.007 |
| granite ↔ rinna_bilingual_4b_jp | 0.005 |
| ministral ↔ rinna_jp_3_6b | 0.002 |
| granite ↔ rinna_jp_3_6b | 0.000 |
| gemma_v7primed ↔ rinna_jp_3_6b | -0.002 |
| gemma ↔ rinna_jp_3_6b | -0.007 |
| ministral ↔ rinna_bilingual_4b | -0.007 |
| opus ↔ rinna_jp_3_6b | -0.010 |
| haiku ↔ rinna_jp_3_6b | -0.014 |
| granite ↔ rinna_bilingual_4b_jpfull | -0.016 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull30 | -0.017 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jpfull | -0.043 |
| gpt_oss_20b ↔ rinna_jp_3_6b | -0.058 |

## Top 25 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 2 | {gemma_v7primed,opus} | 0.788 | 0.829 |
| 2 | 3 | {gemma,gemma_v7primed,opus} | 0.787 | 0.821 |
| 3 | 4 | {gemma,gemma_v7primed,haiku,opus} | 0.780 | 0.821 |
| 4 | 3 | {gemma,gemma_v7primed,haiku} | 0.780 | 0.816 |
| 5 | 1 | {gemma_v7primed} | 0.777 | 0.801 |
| 6 | 2 | {gemma_v7primed,haiku} | 0.774 | 0.814 |
| 7 | 2 | {gemma,opus} | 0.774 | 0.815 |
| 8 | 3 | {gemma_v7primed,haiku,opus} | 0.772 | 0.817 |
| 9 | 2 | {gemma,gemma_v7primed} | 0.770 | 0.788 |
| 10 | 3 | {gemma,haiku,opus} | 0.764 | 0.809 |
| 11 | 4 | {gemma,gemma_v7primed,gpt_oss_20b,opus} | 0.763 | 0.810 |
| 12 | 2 | {gemma,haiku} | 0.761 | 0.800 |
| 13 | 5 | {gemma,gemma_v7primed,granite,haiku,opus} | 0.761 | 0.807 |
| 14 | 5 | {gemma,gemma_v7primed,gpt_oss_20b,haiku,opus} | 0.760 | 0.808 |
| 15 | 4 | {gemma,gemma_v7primed,granite,opus} | 0.758 | 0.800 |
| 16 | 4 | {gemma,gemma_v7primed,granite,haiku} | 0.758 | 0.799 |
| 17 | 4 | {gemma,gemma_v7primed,gpt_oss_20b,haiku} | 0.757 | 0.804 |
| 18 | 3 | {gemma,gemma_v7primed,gpt_oss_20b} | 0.757 | 0.800 |
| 19 | 4 | {gemma,gemma_v7primed,opus,rinna_bilingual_4b_jpfull30} | 0.755 | 0.789 |
| 20 | 5 | {gemma,gemma_v7primed,haiku,opus,rinna_bilingual_4b_jpfull30} | 0.755 | 0.793 |
| 21 | 5 | {gemma,gemma_v7primed,haiku,ministral,opus} | 0.754 | 0.797 |
| 22 | 4 | {gemma,gemma_v7primed,ministral,opus} | 0.754 | 0.793 |
| 23 | 5 | {gemma,gemma_v7primed,haiku,opus,rinna_jp_3_6b_jp} | 0.751 | 0.790 |
| 24 | 4 | {gemma,gemma_v7primed,opus,rinna_jp_3_6b_jp} | 0.750 | 0.785 |
| 25 | 5 | {gemma,gemma_v7primed,haiku,opus,rinna_jp_3_6b_jpfull30} | 0.750 | 0.790 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {gemma_v7primed} | 0.777 | 0.801 |
| 2 | {gemma_v7primed,opus} | 0.788 | 0.829 |
| 3 | {gemma,gemma_v7primed,opus} | 0.787 | 0.821 |
| 4 | {gemma,gemma_v7primed,haiku,opus} | 0.780 | 0.821 |
| 5 | {gemma,gemma_v7primed,granite,haiku,opus} | 0.761 | 0.807 |
| 6 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,opus} | 0.745 | 0.797 |
| 7 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,opus,rinna_bilingual_4b_jpfull30} | 0.730 | 0.778 |
| 8 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b_jpfull30} | 0.715 | 0.763 |
| 9 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp} | 0.703 | 0.748 |
| 10 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 0.691 | 0.734 |
| 11 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 0.680 | 0.722 |
| 12 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull30} | 0.670 | 0.712 |
| 13 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.661 | 0.702 |
| 14 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.652 | 0.694 |
| 15 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.643 | 0.686 |
| 16 | {gemma,gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull,rinna_jp_3_6b_jpfull30} | 0.635 | 0.677 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 69.4% (34/49) | 0.630 |
| gemma_v7primed | 69.4% (34/49) | 0.627 |
| opus | 67.3% (33/49) | 0.593 |
| haiku | 55.1% (26/49) | 0.435 |
| gpt_oss_20b | 51.0% (25/49) | 0.411 |
| rinna_jp_3_6b_jpfull | 40.8% (20/49) | 0.311 |
| granite | 38.8% (19/49) | 0.264 |
| rinna_bilingual_4b_jpfull30 | 38.8% (19/49) | 0.249 |
| rinna_jp_3_6b_jp | 36.7% (18/49) | 0.189 |
| rinna_jp_3_6b_jpfull30 | 34.7% (17/49) | 0.217 |
| qwen | 30.6% (15/49) | 0.156 |
| rinna_bilingual_4b | 28.6% (14/49) | 0.076 |
| ministral | 26.5% (13/49) | 0.137 |
| rinna_bilingual_4b_jpfull | 24.5% (12/49) | 0.125 |
| rinna_bilingual_4b_jp | 22.4% (11/49) | 0.025 |
| rinna_jp_3_6b | 16.3% (7/49) | -0.014 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 7 | {gemma_v7primed,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b_jpfull} | 77.6% (38/49) | 0.724 | 0.628 |
| 6 | {gemma_v7primed,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp} | 77.6% (38/49) | 0.724 | 0.645 |
| 6 | {gemma_v7primed,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b} | 77.6% (38/49) | 0.724 | 0.634 |
| 6 | {gemma_v7primed,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 77.6% (38/49) | 0.724 | 0.633 |
| 9 | {gemma_v7primed,gpt_oss_20b,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jpfull} | 77.6% (38/49) | 0.724 | 0.611 |
| 9 | {gemma_v7primed,gpt_oss_20b,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b,rinna_jp_3_6b_jpfull30} | 77.6% (38/49) | 0.724 | 0.614 |
| 6 | {gemma_v7primed,opus,rinna_bilingual_4b,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jp,rinna_jp_3_6b_jpfull} | 77.6% (38/49) | 0.724 | 0.650 |
| 6 | {gemma_v7primed,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull30} | 77.6% (38/49) | 0.724 | 0.643 |
| 7 | {gemma_v7primed,gpt_oss_20b,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 77.6% (38/49) | 0.724 | 0.638 |
| 10 | {gemma_v7primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b_jpfull,rinna_bilingual_4b_jpfull30,rinna_jp_3_6b} | 77.6% (38/49) | 0.723 | 0.652 |

