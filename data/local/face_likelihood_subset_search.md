# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 13  (bol, gemma, gemma_intro_v7_primed, gpt_oss_20b, granite, haiku, ministral, opus, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_jp_3_6b, rinna_jp_3_6b_jp)
**Faces (overlap):** 62
**GT subset (≥3 pooled emits):** 54
**Subsets evaluated:** 8191

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **opus** at **face-uniform similarity = 0.784** (emit-weighted 0.859)
- Best ensemble subset: **{gemma,gemma_intro_v7_primed,ministral,opus}** at **face-uniform similarity = 0.832** (emit-weighted 0.904); size 4; Δ vs best solo (face-uniform) = +0.048

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| opus | 0.784 | 0.859 | 0.1494 |
| gemma_intro_v7_primed | 0.769 | 0.792 | 0.1600 |
| gemma | 0.763 | 0.787 | 0.1640 |
| haiku | 0.715 | 0.815 | 0.1979 |
| gpt_oss_20b | 0.700 | 0.800 | 0.2077 |
| ministral | 0.669 | 0.780 | 0.2294 |
| qwen | 0.620 | 0.718 | 0.2634 |
| granite | 0.602 | 0.689 | 0.2757 |
| rinna_jp_3_6b_jp | 0.583 | 0.647 | 0.2892 |
| rinna_bilingual_4b | 0.567 | 0.654 | 0.2998 |
| rinna_jp_3_6b | 0.561 | 0.666 | 0.3046 |
| bol | 0.553 | 0.510 | 0.3100 |
| rinna_bilingual_4b_jp | 0.477 | 0.547 | 0.3626 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ gemma_intro_v7_primed | 0.726 |
| haiku ↔ opus | 0.593 |
| gemma_intro_v7_primed ↔ opus | 0.522 |
| gemma ↔ opus | 0.508 |
| gemma ↔ gpt_oss_20b | 0.474 |
| gemma_intro_v7_primed ↔ haiku | 0.461 |
| gemma ↔ haiku | 0.433 |
| gemma_intro_v7_primed ↔ gpt_oss_20b | 0.392 |
| gpt_oss_20b ↔ opus | 0.388 |
| gpt_oss_20b ↔ haiku | 0.365 |
| granite ↔ opus | 0.296 |
| granite ↔ ministral | 0.280 |
| gpt_oss_20b ↔ granite | 0.273 |
| gemma_intro_v7_primed ↔ granite | 0.236 |
| bol ↔ opus | 0.219 |
| bol ↔ gpt_oss_20b | 0.217 |
| bol ↔ gemma | 0.214 |
| haiku ↔ qwen | 0.213 |
| gemma ↔ granite | 0.197 |
| bol ↔ gemma_intro_v7_primed | 0.196 |
| bol ↔ haiku | 0.195 |
| gemma_intro_v7_primed ↔ qwen | 0.186 |
| gemma_intro_v7_primed ↔ rinna_jp_3_6b_jp | 0.182 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b_jp | 0.169 |
| gemma ↔ ministral | 0.159 |
| rinna_bilingual_4b ↔ rinna_bilingual_4b_jp | 0.152 |
| ministral ↔ opus | 0.143 |
| opus ↔ rinna_jp_3_6b_jp | 0.142 |
| gemma_intro_v7_primed ↔ ministral | 0.135 |
| gpt_oss_20b ↔ ministral | 0.132 |
| granite ↔ haiku | 0.128 |
| opus ↔ rinna_bilingual_4b | 0.125 |
| haiku ↔ rinna_jp_3_6b_jp | 0.117 |
| bol ↔ granite | 0.117 |
| rinna_jp_3_6b ↔ rinna_jp_3_6b_jp | 0.114 |
| bol ↔ rinna_jp_3_6b_jp | 0.114 |
| gemma ↔ rinna_jp_3_6b_jp | 0.101 |
| gemma ↔ qwen | 0.098 |
| qwen ↔ rinna_bilingual_4b_jp | 0.091 |
| qwen ↔ rinna_jp_3_6b_jp | 0.090 |
| ministral ↔ qwen | 0.087 |
| granite ↔ rinna_bilingual_4b | 0.081 |
| gemma_intro_v7_primed ↔ rinna_bilingual_4b | 0.081 |
| ministral ↔ rinna_bilingual_4b_jp | 0.077 |
| gpt_oss_20b ↔ qwen | 0.075 |
| opus ↔ qwen | 0.074 |
| bol ↔ ministral | 0.072 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b_jp | 0.069 |
| gemma_intro_v7_primed ↔ rinna_bilingual_4b_jp | 0.059 |
| bol ↔ rinna_jp_3_6b | 0.057 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jp | 0.054 |
| granite ↔ rinna_jp_3_6b_jp | 0.053 |
| gemma ↔ rinna_bilingual_4b | 0.051 |
| gpt_oss_20b ↔ rinna_bilingual_4b | 0.045 |
| gemma ↔ rinna_bilingual_4b_jp | 0.043 |
| rinna_bilingual_4b ↔ rinna_jp_3_6b | 0.040 |
| granite ↔ qwen | 0.036 |
| haiku ↔ rinna_jp_3_6b | 0.036 |
| bol ↔ rinna_bilingual_4b_jp | 0.034 |
| haiku ↔ rinna_bilingual_4b_jp | 0.034 |
| qwen ↔ rinna_jp_3_6b | 0.033 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jp | 0.031 |
| gemma_intro_v7_primed ↔ rinna_jp_3_6b | 0.030 |
| opus ↔ rinna_jp_3_6b | 0.029 |
| rinna_bilingual_4b_jp ↔ rinna_jp_3_6b | 0.029 |
| gemma ↔ rinna_jp_3_6b | 0.027 |
| haiku ↔ ministral | 0.022 |
| qwen ↔ rinna_bilingual_4b | 0.021 |
| opus ↔ rinna_bilingual_4b_jp | 0.018 |
| granite ↔ rinna_bilingual_4b_jp | 0.016 |
| haiku ↔ rinna_bilingual_4b | 0.012 |
| ministral ↔ rinna_jp_3_6b_jp | 0.007 |
| ministral ↔ rinna_jp_3_6b | -0.001 |
| ministral ↔ rinna_bilingual_4b | -0.006 |
| granite ↔ rinna_jp_3_6b | -0.009 |
| bol ↔ rinna_bilingual_4b | -0.011 |
| gpt_oss_20b ↔ rinna_jp_3_6b | -0.026 |
| bol ↔ qwen | -0.038 |

## Top 25 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 4 | {gemma,gemma_intro_v7_primed,ministral,opus} | 0.832 | 0.904 |
| 2 | 3 | {gemma,gemma_intro_v7_primed,opus} | 0.830 | 0.870 |
| 3 | 2 | {gemma,opus} | 0.830 | 0.876 |
| 4 | 4 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,opus} | 0.828 | 0.891 |
| 5 | 2 | {gemma_intro_v7_primed,opus} | 0.828 | 0.875 |
| 6 | 5 | {gemma,gemma_intro_v7_primed,haiku,ministral,opus} | 0.827 | 0.901 |
| 7 | 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.825 | 0.877 |
| 8 | 4 | {gemma,gemma_intro_v7_primed,haiku,ministral} | 0.823 | 0.898 |
| 9 | 5 | {gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.823 | 0.874 |
| 10 | 4 | {gemma,gemma_intro_v7_primed,granite,opus} | 0.823 | 0.865 |
| 11 | 5 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,haiku,opus} | 0.821 | 0.888 |
| 12 | 5 | {gemma,gemma_intro_v7_primed,granite,ministral,opus} | 0.821 | 0.890 |
| 13 | 4 | {gemma,gemma_intro_v7_primed,opus,qwen} | 0.821 | 0.895 |
| 14 | 5 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,opus} | 0.821 | 0.882 |
| 15 | 6 | {gemma,gemma_intro_v7_primed,granite,haiku,ministral,opus} | 0.820 | 0.891 |
| 16 | 5 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,ministral,opus} | 0.820 | 0.899 |
| 17 | 3 | {gemma,ministral,opus} | 0.820 | 0.902 |
| 18 | 3 | {gemma_intro_v7_primed,ministral,opus} | 0.820 | 0.900 |
| 19 | 3 | {gemma,gemma_intro_v7_primed,ministral} | 0.820 | 0.894 |
| 20 | 5 | {bol,gemma,gemma_intro_v7_primed,ministral,opus} | 0.818 | 0.878 |
| 21 | 6 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,opus} | 0.818 | 0.883 |
| 22 | 3 | {gemma,gemma_intro_v7_primed,gpt_oss_20b} | 0.818 | 0.883 |
| 23 | 4 | {gemma,gemma_intro_v7_primed,granite,haiku} | 0.818 | 0.862 |
| 24 | 3 | {gemma,gpt_oss_20b,opus} | 0.817 | 0.889 |
| 25 | 5 | {gemma,gemma_intro_v7_primed,granite,haiku,ministral} | 0.817 | 0.887 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {opus} | 0.784 | 0.859 |
| 2 | {gemma,opus} | 0.830 | 0.876 |
| 3 | {gemma,gemma_intro_v7_primed,opus} | 0.830 | 0.870 |
| 4 | {gemma,gemma_intro_v7_primed,ministral,opus} | 0.832 | 0.904 |
| 5 | {gemma,gemma_intro_v7_primed,haiku,ministral,opus} | 0.827 | 0.901 |
| 6 | {gemma,gemma_intro_v7_primed,granite,haiku,ministral,opus} | 0.820 | 0.891 |
| 7 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus} | 0.813 | 0.890 |
| 8 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus} | 0.806 | 0.875 |
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen} | 0.796 | 0.873 |
| 10 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_jp_3_6b_jp} | 0.785 | 0.863 |
| 11 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.774 | 0.859 |
| 12 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.763 | 0.850 |
| 13 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.752 | 0.840 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 74.1% (40/54) | 0.679 |
| gemma_intro_v7_primed | 68.5% (37/54) | 0.608 |
| opus | 64.8% (35/54) | 0.550 |
| haiku | 50.0% (27/54) | 0.366 |
| gpt_oss_20b | 46.3% (25/54) | 0.345 |
| granite | 42.6% (23/54) | 0.271 |
| bol | 40.7% (22/54) | 0.227 |
| rinna_jp_3_6b_jp | 31.5% (17/54) | 0.142 |
| ministral | 29.6% (16/54) | 0.133 |
| qwen | 27.8% (15/54) | 0.121 |
| rinna_bilingual_4b | 27.8% (15/54) | 0.100 |
| rinna_bilingual_4b_jp | 20.4% (11/54) | 0.041 |
| rinna_jp_3_6b | 13.0% (7/54) | -0.008 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,ministral,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp} | 81.5% (44/54) | 0.766 | 0.776 |
| 12 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 81.5% (44/54) | 0.766 | 0.763 |
| 8 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,rinna_jp_3_6b_jp} | 81.5% (44/54) | 0.766 | 0.790 |
| 8 | {bol,gemma,gemma_intro_v7_primed,granite,haiku,ministral,opus,rinna_bilingual_4b} | 81.5% (44/54) | 0.766 | 0.797 |
| 8 | {bol,gemma,gemma_intro_v7_primed,granite,haiku,ministral,opus,rinna_bilingual_4b_jp} | 81.5% (44/54) | 0.766 | 0.797 |
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b_jp} | 81.5% (44/54) | 0.766 | 0.773 |
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,ministral,opus,rinna_bilingual_4b,rinna_jp_3_6b} | 81.5% (44/54) | 0.766 | 0.777 |
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,ministral,opus,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 81.5% (44/54) | 0.766 | 0.777 |
| 10 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 81.5% (44/54) | 0.766 | 0.760 |
| 10 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,ministral,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp} | 81.5% (44/54) | 0.766 | 0.767 |

