# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 13  (bol, gemma, gemma_intro_v7_primed, gpt_oss_20b, granite, haiku, ministral, opus, qwen, rinna_bilingual_4b, rinna_bilingual_4b_jp, rinna_jp_3_6b, rinna_jp_3_6b_jp)
**Faces (overlap):** 62
**GT subset (Claude empirical, total ≥ 3):** 38
**Subsets evaluated:** 8191

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **gemma_intro_v7_primed** at **face-uniform similarity = 0.797** (emit-weighted 0.799)
- Best ensemble subset: **{gemma_intro_v7_primed}** at **face-uniform similarity = 0.797** (emit-weighted 0.799); size 1; Δ vs best solo (face-uniform) = +0.000

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| gemma_intro_v7_primed | 0.797 | 0.799 | 0.1408 |
| gemma | 0.754 | 0.741 | 0.1706 |
| opus | 0.732 | 0.780 | 0.1861 |
| haiku | 0.670 | 0.701 | 0.2284 |
| gpt_oss_20b | 0.585 | 0.643 | 0.2879 |
| bol | 0.556 | 0.456 | 0.3076 |
| rinna_jp_3_6b_jp | 0.547 | 0.559 | 0.3143 |
| ministral | 0.538 | 0.624 | 0.3205 |
| granite | 0.528 | 0.577 | 0.3273 |
| rinna_bilingual_4b | 0.507 | 0.532 | 0.3419 |
| qwen | 0.489 | 0.545 | 0.3542 |
| rinna_jp_3_6b | 0.468 | 0.520 | 0.3689 |
| rinna_bilingual_4b_jp | 0.411 | 0.424 | 0.4082 |

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

## Top 20 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 1 | {gemma_intro_v7_primed} | 0.797 | 0.799 |
| 2 | 3 | {gemma,gemma_intro_v7_primed,opus} | 0.795 | 0.812 |
| 3 | 2 | {gemma_intro_v7_primed,opus} | 0.793 | 0.820 |
| 4 | 2 | {gemma,gemma_intro_v7_primed} | 0.786 | 0.782 |
| 5 | 3 | {gemma,gemma_intro_v7_primed,haiku} | 0.784 | 0.800 |
| 6 | 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.782 | 0.807 |
| 7 | 2 | {gemma,opus} | 0.776 | 0.802 |
| 8 | 2 | {gemma_intro_v7_primed,haiku} | 0.776 | 0.798 |
| 9 | 3 | {gemma_intro_v7_primed,haiku,opus} | 0.771 | 0.802 |
| 10 | 4 | {gemma,gemma_intro_v7_primed,granite,opus} | 0.770 | 0.794 |
| 11 | 4 | {bol,gemma,gemma_intro_v7_primed,opus} | 0.770 | 0.774 |
| 12 | 4 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,opus} | 0.767 | 0.799 |
| 13 | 5 | {gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.766 | 0.796 |
| 14 | 4 | {gemma,gemma_intro_v7_primed,granite,haiku} | 0.766 | 0.788 |
| 15 | 3 | {gemma,gemma_intro_v7_primed,gpt_oss_20b} | 0.765 | 0.790 |
| 16 | 5 | {bol,gemma,gemma_intro_v7_primed,haiku,opus} | 0.764 | 0.777 |
| 17 | 4 | {gemma,gemma_intro_v7_primed,opus,rinna_jp_3_6b_jp} | 0.763 | 0.789 |
| 18 | 3 | {gemma,gemma_intro_v7_primed,granite} | 0.763 | 0.770 |
| 19 | 4 | {bol,gemma,gemma_intro_v7_primed,haiku} | 0.762 | 0.765 |
| 20 | 3 | {bol,gemma,gemma_intro_v7_primed} | 0.761 | 0.748 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {gemma_intro_v7_primed} | 0.797 | 0.799 |
| 2 | {gemma_intro_v7_primed,opus} | 0.793 | 0.820 |
| 3 | {gemma,gemma_intro_v7_primed,opus} | 0.795 | 0.812 |
| 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.782 | 0.807 |
| 5 | {gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.766 | 0.796 |
| 6 | {bol,gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.754 | 0.771 |
| 7 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,opus} | 0.741 | 0.766 |
| 8 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,opus,rinna_jp_3_6b_jp} | 0.729 | 0.756 |
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_jp_3_6b_jp} | 0.717 | 0.748 |
| 10 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b,rinna_jp_3_6b_jp} | 0.704 | 0.737 |
| 11 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b_jp} | 0.690 | 0.724 |
| 12 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.678 | 0.714 |
| 13 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 0.667 | 0.704 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 71.1% (27/38) | 0.640 |
| gemma_intro_v7_primed | 71.1% (27/38) | 0.636 |
| opus | 68.4% (26/38) | 0.594 |
| haiku | 55.3% (21/38) | 0.425 |
| gpt_oss_20b | 47.4% (18/38) | 0.356 |
| bol | 44.7% (17/38) | 0.267 |
| rinna_jp_3_6b_jp | 42.1% (16/38) | 0.241 |
| granite | 39.5% (15/38) | 0.252 |
| rinna_bilingual_4b | 31.6% (12/38) | 0.101 |
| qwen | 26.3% (10/38) | 0.092 |
| ministral | 23.7% (9/38) | 0.083 |
| rinna_bilingual_4b_jp | 23.7% (9/38) | 0.042 |
| rinna_jp_3_6b | 15.8% (6/38) | -0.001 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 6 | {gemma_intro_v7_primed,granite,haiku,qwen,rinna_bilingual_4b,rinna_jp_3_6b} | 81.6% (31/38) | 0.765 | 0.657 |
| 9 | {bol,gemma_intro_v7_primed,granite,haiku,ministral,opus,rinna_bilingual_4b_jp,rinna_jp_3_6b,rinna_jp_3_6b_jp} | 81.6% (31/38) | 0.764 | 0.674 |
| 8 | {bol,gemma_intro_v7_primed,granite,haiku,ministral,opus,qwen,rinna_jp_3_6b_jp} | 81.6% (31/38) | 0.764 | 0.689 |
| 8 | {gemma_intro_v7_primed,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 81.6% (31/38) | 0.765 | 0.661 |
| 7 | {gemma_intro_v7_primed,granite,haiku,opus,qwen,rinna_bilingual_4b,rinna_jp_3_6b_jp} | 81.6% (31/38) | 0.765 | 0.683 |
| 8 | {bol,gemma_intro_v7_primed,granite,haiku,ministral,opus,qwen,rinna_jp_3_6b} | 81.6% (31/38) | 0.764 | 0.680 |
| 7 | {gemma_intro_v7_primed,granite,haiku,opus,qwen,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 81.6% (31/38) | 0.765 | 0.670 |
| 7 | {gemma_intro_v7_primed,granite,haiku,ministral,opus,rinna_bilingual_4b_jp,rinna_jp_3_6b} | 81.6% (31/38) | 0.765 | 0.679 |
| 9 | {bol,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_jp_3_6b} | 81.6% (31/38) | 0.764 | 0.673 |
| 6 | {gemma_intro_v7_primed,granite,haiku,opus,rinna_bilingual_4b,rinna_jp_3_6b_jp} | 81.6% (31/38) | 0.765 | 0.705 |

