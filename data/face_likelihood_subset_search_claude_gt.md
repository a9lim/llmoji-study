# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 9  (bol, gemma, gemma_intro_v7_primed, gpt_oss_20b, granite, haiku, ministral, opus, qwen)
**Faces (overlap):** 68
**GT subset (Claude empirical, total ≥ 3):** 40
**Subsets evaluated:** 511

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **gemma_intro_v7_primed** at **face-uniform similarity = 0.790** (emit-weighted 0.798)
- Best ensemble subset: **{gemma,gemma_intro_v7_primed,opus}** at **face-uniform similarity = 0.793** (emit-weighted 0.812); size 3; Δ vs best solo (face-uniform) = +0.003

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| gemma_intro_v7_primed | 0.790 | 0.798 | 0.1452 |
| gemma | 0.754 | 0.742 | 0.1705 |
| opus | 0.736 | 0.781 | 0.1829 |
| haiku | 0.675 | 0.702 | 0.2250 |
| gpt_oss_20b | 0.588 | 0.643 | 0.2855 |
| bol | 0.549 | 0.455 | 0.3129 |
| ministral | 0.537 | 0.623 | 0.3211 |
| granite | 0.520 | 0.575 | 0.3326 |
| qwen | 0.494 | 0.546 | 0.3511 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ gemma_intro_v7_primed | 0.732 |
| haiku ↔ opus | 0.623 |
| gemma_intro_v7_primed ↔ opus | 0.540 |
| gemma ↔ opus | 0.513 |
| gemma ↔ gpt_oss_20b | 0.502 |
| gemma_intro_v7_primed ↔ haiku | 0.486 |
| gemma ↔ haiku | 0.445 |
| gpt_oss_20b ↔ opus | 0.417 |
| gemma_intro_v7_primed ↔ gpt_oss_20b | 0.408 |
| gpt_oss_20b ↔ haiku | 0.399 |
| granite ↔ ministral | 0.311 |
| granite ↔ opus | 0.265 |
| gpt_oss_20b ↔ granite | 0.260 |
| haiku ↔ qwen | 0.255 |
| bol ↔ opus | 0.254 |
| gemma_intro_v7_primed ↔ qwen | 0.252 |
| bol ↔ gpt_oss_20b | 0.239 |
| bol ↔ haiku | 0.232 |
| bol ↔ gemma | 0.226 |
| gemma_intro_v7_primed ↔ granite | 0.223 |
| bol ↔ gemma_intro_v7_primed | 0.221 |
| gemma ↔ granite | 0.206 |
| gemma ↔ ministral | 0.157 |
| gemma ↔ qwen | 0.154 |
| gemma_intro_v7_primed ↔ ministral | 0.127 |
| opus ↔ qwen | 0.123 |
| ministral ↔ opus | 0.121 |
| gpt_oss_20b ↔ ministral | 0.119 |
| gpt_oss_20b ↔ qwen | 0.114 |
| granite ↔ haiku | 0.113 |
| bol ↔ granite | 0.109 |
| ministral ↔ qwen | 0.087 |
| bol ↔ ministral | 0.063 |
| granite ↔ qwen | 0.043 |
| haiku ↔ ministral | 0.013 |
| bol ↔ qwen | 0.004 |

## Top 25 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 3 | {gemma,gemma_intro_v7_primed,opus} | 0.793 | 0.812 |
| 2 | 2 | {gemma_intro_v7_primed,opus} | 0.792 | 0.820 |
| 3 | 1 | {gemma_intro_v7_primed} | 0.790 | 0.798 |
| 4 | 2 | {gemma,gemma_intro_v7_primed} | 0.783 | 0.781 |
| 5 | 3 | {gemma,gemma_intro_v7_primed,haiku} | 0.783 | 0.800 |
| 6 | 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.782 | 0.807 |
| 7 | 2 | {gemma,opus} | 0.778 | 0.802 |
| 8 | 2 | {gemma_intro_v7_primed,haiku} | 0.774 | 0.798 |
| 9 | 3 | {gemma_intro_v7_primed,haiku,opus} | 0.771 | 0.802 |
| 10 | 4 | {bol,gemma,gemma_intro_v7_primed,opus} | 0.769 | 0.774 |
| 11 | 4 | {gemma,gemma_intro_v7_primed,granite,opus} | 0.767 | 0.793 |
| 12 | 4 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,opus} | 0.766 | 0.799 |
| 13 | 5 | {gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.764 | 0.795 |
| 14 | 5 | {bol,gemma,gemma_intro_v7_primed,haiku,opus} | 0.763 | 0.777 |
| 15 | 4 | {gemma,gemma_intro_v7_primed,granite,haiku} | 0.763 | 0.787 |
| 16 | 3 | {gemma,haiku,opus} | 0.762 | 0.792 |
| 17 | 3 | {gemma,gemma_intro_v7_primed,gpt_oss_20b} | 0.762 | 0.789 |
| 18 | 4 | {bol,gemma,gemma_intro_v7_primed,haiku} | 0.761 | 0.765 |
| 19 | 2 | {gemma,haiku} | 0.760 | 0.780 |
| 20 | 5 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,haiku,opus} | 0.759 | 0.793 |
| 21 | 3 | {bol,gemma,gemma_intro_v7_primed} | 0.759 | 0.747 |
| 22 | 4 | {gemma,gemma_intro_v7_primed,ministral,opus} | 0.758 | 0.792 |
| 23 | 4 | {gemma,gemma_intro_v7_primed,gpt_oss_20b,haiku} | 0.758 | 0.788 |
| 24 | 3 | {gemma,gemma_intro_v7_primed,granite} | 0.756 | 0.769 |
| 25 | 5 | {gemma,gemma_intro_v7_primed,haiku,ministral,opus} | 0.755 | 0.790 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {gemma_intro_v7_primed} | 0.790 | 0.798 |
| 2 | {gemma_intro_v7_primed,opus} | 0.792 | 0.820 |
| 3 | {gemma,gemma_intro_v7_primed,opus} | 0.793 | 0.812 |
| 4 | {gemma,gemma_intro_v7_primed,haiku,opus} | 0.782 | 0.807 |
| 5 | {gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.764 | 0.795 |
| 6 | {bol,gemma,gemma_intro_v7_primed,granite,haiku,opus} | 0.752 | 0.771 |
| 7 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,opus} | 0.739 | 0.765 |
| 8 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus} | 0.724 | 0.756 |
| 9 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen} | 0.707 | 0.742 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| gemma | 70.0% (28/40) | 0.628 |
| gemma_intro_v7_primed | 70.0% (28/40) | 0.624 |
| opus | 70.0% (28/40) | 0.612 |
| haiku | 57.5% (23/40) | 0.451 |
| gpt_oss_20b | 50.0% (20/40) | 0.387 |
| bol | 45.0% (18/40) | 0.266 |
| granite | 37.5% (15/40) | 0.221 |
| qwen | 27.5% (11/40) | 0.108 |
| ministral | 22.5% (9/40) | 0.070 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 6 | {bol,gemma_intro_v7_primed,granite,haiku,ministral,opus} | 82.5% (33/40) | 0.775 | 0.717 |
| 7 | {bol,gemma_intro_v7_primed,granite,haiku,ministral,opus,qwen} | 82.5% (33/40) | 0.775 | 0.696 |
| 7 | {bol,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus} | 82.5% (33/40) | 0.775 | 0.706 |
| 8 | {bol,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus,qwen} | 82.5% (33/40) | 0.775 | 0.688 |
| 4 | {gemma,gpt_oss_20b,haiku,opus} | 80.0% (32/40) | 0.744 | 0.735 |
| 5 | {gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,opus} | 80.0% (32/40) | 0.745 | 0.729 |
| 8 | {bol,gemma,gemma_intro_v7_primed,gpt_oss_20b,granite,haiku,ministral,opus} | 80.0% (32/40) | 0.745 | 0.724 |
| 6 | {bol,gemma_intro_v7_primed,granite,haiku,opus,qwen} | 80.0% (32/40) | 0.742 | 0.711 |
| 5 | {gemma,gpt_oss_20b,haiku,ministral,opus} | 80.0% (32/40) | 0.745 | 0.710 |
| 3 | {gemma_intro_v7_primed,granite,haiku} | 80.0% (32/40) | 0.745 | 0.745 |

